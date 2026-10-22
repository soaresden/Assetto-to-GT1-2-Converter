"""Gran Turismo (PS1) archives: "@(#)GT-ARC" containers and their LZSS compression.

GT-ARC layout:
  0x00  "@(#)GT-ARC" + u16 0
  0x0C  u8 1, u8 flags (0x80: some entries are compressed), u16 entry count
  0x10  entries of 12 bytes: u32 offset, u32 stored size, u32 size (compressed when they differ)
No names: entries are known by their index. A whole file (CARINF.DAT) may itself be compressed.

LZSS: a flag byte for each 8 items, low bit first; 0 = a literal byte, 1 = a reference made of a
length byte (+3) and a distance (one byte + 1, or, if its top bit is set, 15 bits on two bytes,
+ 1) back into what was already written.
"""
import struct

import numpy as np

MAGIC = b'@(#)GT-ARC'


def decompress(src, size=None):
    out = bytearray()
    i = 0
    n = len(src)
    while i < n and (size is None or len(out) < size):
        flags = src[i]
        i += 1
        for bit in range(8):
            if i >= n or (size is not None and len(out) >= size):
                break
            if flags >> bit & 1:
                length = src[i] + 3
                d = src[i + 1]
                if d & 0x80:
                    d = ((d & 0x7F) << 8 | src[i + 2]) + 1
                    i += 3
                else:
                    d += 1
                    i += 2
                start = len(out) - d
                for k in range(length):
                    out.append(out[start + k])
            else:
                out.append(src[i])
                i += 1
    return bytes(out)


def literal(data):
    """'Compression' with literals only (9/8 of the size): always accepted by the game's decoder."""
    out = bytearray()
    for i in range(0, len(data), 8):
        out.append(0)
        out += data[i:i + 8]
    return bytes(out)


def _matches(data, window, max_len, tries, block=1 << 15):
    """For each position: (longest match length, its distance, longest match length within 128
    bytes, its distance), among the `tries` latest earlier positions starting with the same 3
    bytes and within `window` (nearest first on equal lengths): hash chains, walked with numpy
    for all the positions of a block at once."""
    n = len(data)
    ml = np.zeros(n, np.int32)
    md = np.zeros(n, np.int32)
    sl = np.zeros(n, np.int32)
    sd = np.zeros(n, np.int32)
    m = n - 2
    if m <= 0:
        return ml, md, sl, sd
    a = np.frombuffer(data, np.uint8)
    key = (a[:m].astype(np.int32) << 16) | (a[1:m + 1].astype(np.int32) << 8) | a[2:m + 2]
    # the chains: positions sorted by their 3 bytes, then by position; the k-th earlier position
    # with the same bytes is k places before in that order (if still in the same group)
    order = np.argsort(key, kind='stable').astype(np.int32)
    skey = key[order]
    rank = np.empty(m, np.int32)
    rank[order] = np.arange(m, dtype=np.int32)
    first = np.ones(m, bool)
    first[1:] = skey[1:] != skey[:-1]
    gstart = np.maximum.accumulate(np.where(first, np.arange(m, dtype=np.int32), 0)).astype(np.int32)
    # the 8 bytes at every offset (little endian), zero padded: matches stop at the limit anyway
    pad = np.zeros(n + max_len + 16, np.uint8)
    pad[:n] = a
    words = np.lib.stride_tricks.as_strided(pad, (n + max_len + 8, 8), (1, 1)).copy().view('<u8').ravel()
    one = np.uint64(1)
    for b0 in range(0, m, block):
        pos = np.arange(b0, min(m, b0 + block), dtype=np.int32)
        r = rank[pos]
        g = gstart[r]
        limit = np.minimum(max_len, n - pos)
        bl = np.zeros(len(pos), np.int32)
        bd = np.zeros(len(pos), np.int32)
        sbl = np.zeros(len(pos), np.int32)
        sbd = np.zeros(len(pos), np.int32)
        live = np.arange(len(pos))
        for k in range(1, tries + 1):
            # the k-th latest earlier position, for the positions still searching (a match as
            # long as the limit ends the search: the farther ones cannot change anything)
            live = live[r[live] - k >= g[live]]
            if not len(live):
                break
            p = order[r[live] - k]
            d = pos[live] - p
            w = d <= window
            live, p, d = live[w], p[w], d[w]
            if not len(live):
                break
            i = pos[live]
            lim = limit[live]
            ln = np.full(len(live), 3, np.int32)
            act = np.nonzero(ln < lim)[0]
            while len(act):
                # 8 bytes at a time; the first differing byte is the lowest set byte of the xor
                x = words[i[act] + ln[act]] ^ words[p[act] + ln[act]]
                same = x == 0
                low = np.where(same, one, x & (~x + one)).astype(np.float64)
                nb = np.where(same, 8, np.log2(low).astype(np.int32) >> 3)
                nl = np.minimum(ln[act] + nb, lim[act])
                ln[act] = nl
                act = act[same & (nl < lim[act])]
            up = ln > bl[live]
            bl[live[up]] = ln[up]
            bd[live[up]] = d[up]
            up = (d <= 128) & (ln > sbl[live])
            sbl[live[up]] = ln[up]
            sbd[live[up]] = d[up]
            live = live[bl[live] < lim]
        ml[pos], md[pos], sl[pos], sd[pos] = bl, bd, sbl, sbd
    return ml, md, sl, sd


def compress(data, window=0x7FFF, max_len=258, tries=48):
    """LZSS with an optimal parse of the matches found (costs in bits: literal 9, match within
    128 bytes 17, farther 25, flag bits included); output decodes with decompress()."""
    data = bytes(data)
    n = len(data)
    ml, md, sl, sd = (v.tolist() for v in _matches(data, window, max_len, tries))
    cost = [0] * (n + 1)
    choice = [0] * n   # 0: literal, else (length << 1) | short
    for i in range(n - 1, -1, -1):
        # the cheapest of: a literal, a match of 3..S within 128 bytes, a farther one of S+1..L
        # (on equal costs: the literal, then the shortest match)
        best = 9 + cost[i + 1]
        ch = 0
        L = ml[i]
        if L >= 3:
            S = sl[i]
            if S >= 3:
                c = min(cost[i + 3:i + S + 1])
                if 17 + c < best:
                    best = 17 + c
                    ch = (cost.index(c, i + 3, i + S + 1) - i) << 1 | 1
            lo = max(S + 1, 3)
            if lo <= L:
                c = min(cost[i + lo:i + L + 1])
                if 25 + c < best:
                    best = 25 + c
                    ch = (cost.index(c, i + lo, i + L + 1) - i) << 1
        cost[i] = best
        choice[i] = ch
    out = bytearray()
    i = 0
    while i < n:
        flag_pos = len(out)
        out.append(0)
        flags = 0
        for bit in range(8):
            if i >= n:
                break
            ch = choice[i]
            if ch:
                l, short = ch >> 1, ch & 1
                d = (sd[i] if short else md[i]) - 1
                flags |= 1 << bit
                out.append(l - 3)
                if d < 0x80:
                    out.append(d)
                else:
                    out += bytes((0x80 | d >> 8, d & 0xFF))
                i += l
            else:
                out.append(data[i])
                i += 1
        out[flag_pos] = flags
    return bytes(out)


class Arc:
    def __init__(self, data):
        if data[1:9] == MAGIC[:8] and data[:1] == b'\0':
            data = decompress(data)  # the whole archive is compressed
        if data[:10] != MAGIC:
            raise ValueError('not a GT-ARC')
        self.data = data
        self.version, self.flags, count = struct.unpack_from('<BBH', data, 12)
        self.entries = [struct.unpack_from('<III', data, 16 + 12 * i) for i in range(count)]

    def __len__(self):
        return len(self.entries)

    def raw(self, i):
        off, stored, _ = self.entries[i]
        return self.data[off:off + stored]

    def __getitem__(self, i):
        off, stored, size = self.entries[i]
        b = self.data[off:off + stored]
        return b if stored == size else decompress(b, size)

    def compressed(self, i):
        return self.entries[i][1] != self.entries[i][2]


def replace(arc, new, align=0x800):
    """The archive with some entries replaced ({index: uncompressed bytes}, stored compressed);
    the other entries are copied as they are stored. Entries start on `align` boundaries."""
    count = len(arc)
    off = (16 + 12 * count + align - 1) // align * align
    table = bytearray()
    body = bytearray()
    done = {}          # the same bytes are compressed once (CAR.DAT: day and night copies)
    for i in range(count):
        if i in new:
            size = len(new[i])
            # compressed if the entry was: the game decompresses some files whatever the sizes
            # say (a menu page stored 21102 bytes compressed for 20276); stored raw only if it was
            if arc.compressed(i):
                key = bytes(new[i])
                if key not in done:
                    done[key] = compress(key)
                stored = done[key]
            else:
                stored = new[i]
            if len(stored) == size and arc.compressed(i):
                stored += b'\0'          # stored == size would read as "not compressed"
        else:
            stored, size = arc.raw(i), arc.entries[i][2]
        table += struct.pack('<III', off + len(body), len(stored), size)
        body += stored
        body += b'\0' * (-len(body) % align)
    head = bytearray(arc.data[:16]) + table
    head += b'\0' * (off - len(head))
    head += body           # one copy less of MENU_IMG.ARC (120 MB) in the browser's memory
    del body
    return bytes(head)


def build(files, compressed, align=0x800, version=1):
    """A GT-ARC from a list of (uncompressed) files; compressed[i] says which ones to compress."""
    count = len(files)
    off = 16 + 12 * count
    off = (off + align - 1) // align * align
    table = bytearray()
    body = bytearray()
    flags = 0
    for f, c in zip(files, compressed):
        stored = compress(f) if c else f
        if c:
            flags = 0x80
        table += struct.pack('<III', off + len(body), len(stored), len(f))
        body += stored
        body += b'\0' * (-len(body) % align)
    head = MAGIC + b'\0\0' + struct.pack('<BBH', version, flags, count) + table
    head += b'\0' * ((16 + 12 * count + align - 1) // align * align - len(head))
    return bytes(head + body)
