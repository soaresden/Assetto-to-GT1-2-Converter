"""Write a single-track CD image (.bin of raw 2352-byte sectors) as a CHD v5, as `chdman createcd`
does: hunks of 8 frames (sector + 96 bytes of zero subcode), each compressed with cdlz (LZMA) or
cdzl (deflate), whichever is smaller, or stored; the Huffman-coded map; the CHT2 track metadata;
the raw and overall SHA-1. Read back by chd.py, libchdr (DuckStation, RetroArch, Batocera) and MAME.

python chdmake.py <disc.bin> <out.chd> [MODE2_RAW|MODE1_RAW]

The sync and ECC of the sectors are kept (chdman strips the ones it can rebuild): a slightly bigger
file, the same data.
"""
import hashlib
import lzma
import os
import struct
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chd  # noqa: E402

FRAMES_PER_HUNK = 8
HUNK = FRAMES_PER_HUNK * chd.FRAME          # 19584
CODECS = (b'cdlz', b'cdzl', b'cdfl', b'\0\0\0\0')
CD_TRACK_PADDING = 4
META_CHECKSUM = 0x01


def _lzma(data):
    filt = [{'id': lzma.FILTER_LZMA1, 'dict_size': chd._lzma_props(len(data)), 'lc': 3, 'lp': 0, 'pb': 2,
             'preset': 8}]
    return lzma.compress(data, format=lzma.FORMAT_RAW, filters=filt)


def _deflate(data):
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush()


def compress_hunk(raw):
    """(type, bytes): type 0 cdlz, 1 cdzl, 4 stored; raw is HUNK bytes (frames with subcode)."""
    sect = b''.join(raw[i * chd.FRAME:i * chd.FRAME + chd.SECTOR] for i in range(FRAMES_PER_HUNK))
    sub = b''.join(raw[i * chd.FRAME + chd.SECTOR:(i + 1) * chd.FRAME] for i in range(FRAMES_PER_HUNK))
    ecc = bytes((FRAMES_PER_HUNK + 7) // 8)      # no frame has its ECC removed
    zsub = _deflate(sub)
    best = None
    for kind, fn in ((0, _lzma), (1, _deflate)):
        base = fn(sect)
        if len(base) >= 1 << 16:
            continue
        out = ecc + struct.pack('>H', len(base)) + base + zsub
        if best is None or len(out) < len(best[1]):
            best = (kind, out)
    if best is None or len(best[1]) >= HUNK:
        return 4, raw
    return best


def _work(args):
    path, first, count, frames = args
    out = []
    with open(path, 'rb') as f:
        for h in range(first, first + count):
            raw = bytearray(HUNK)
            for i in range(FRAMES_PER_HUNK):
                lba = h * FRAMES_PER_HUNK + i
                if lba < frames:
                    f.seek(lba * chd.SECTOR)
                    raw[i * chd.FRAME:i * chd.FRAME + chd.SECTOR] = f.read(chd.SECTOR)
            raw = bytes(raw)
            kind, data = compress_hunk(raw)
            out.append((kind, data, chd.crc16(raw), raw))
    return [(k, d, c) for k, d, c, _ in out], b''.join(r for *_, r in out)


class BitWriter:
    def __init__(self):
        self.bits = []

    def write(self, value, n):
        for i in range(n - 1, -1, -1):
            self.bits.append((value >> i) & 1)

    def bytes(self):
        b = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int(''.join(map(str, b[i:i + 8])), 2) for i in range(0, len(b), 8))


def huffman_lengths(freq, maxbits=8):
    """Code lengths (<= maxbits) for the symbols with a non-zero count."""
    import heapq
    used = [s for s, f in enumerate(freq) if f]
    lens = [0] * len(freq)
    if len(used) == 1:
        lens[used[0]] = 1
        return lens
    heap = [(freq[s], i, [s]) for i, s in enumerate(used)]
    heapq.heapify(heap)
    k = len(heap)
    while len(heap) > 1:
        a, _, sa = heapq.heappop(heap)
        b, _, sb = heapq.heappop(heap)
        for s in sa + sb:
            lens[s] += 1
        heapq.heappush(heap, (a + b, k, sa + sb))
        k += 1
    if max(lens) > maxbits:                       # flat codes: 16 symbols fit in 4 bits
        lens = [4 if f else 0 for f in freq]
    return lens


def canonical(lens):
    """The decoder's code assignment (libchdr huffman_assign_canonical_codes): {sym: (code, len)}."""
    histo = [0] * 33
    for n in lens:
        histo[n] += 1
    start = 0
    for n in range(32, 0, -1):
        nxt = (start + histo[n]) >> 1
        histo[n] = start
        start = nxt
    codes = {}
    for s, n in enumerate(lens):
        if n:
            codes[s] = (histo[n], n)
            histo[n] += 1
    return codes


def export_tree(bw, lens, nb=4):
    """libchdr import_tree_rle in reverse: 1 escapes (1,1 = a literal 1; 1,L,c-3 = L repeated c times)."""
    i = 0
    while i < len(lens):
        n = lens[i]
        run = 1
        while i + run < len(lens) and lens[i + run] == n and run < 18:
            run += 1
        if n == 1:
            bw.write(1, nb)
            bw.write(1, nb)
            i += 1
        elif run >= 3:
            bw.write(1, nb)
            bw.write(n, nb)
            bw.write(run - 3, nb)
            i += run
        else:
            bw.write(n, nb)
            i += 1


def compress_map(entries, first_offset):
    """The v5 map: entries [(type, offset, length, crc)], types 0..4 only."""
    raw = b''.join(struct.pack('>B', t) + struct.pack('>I', ln)[1:] + struct.pack('>Q', off)[2:] + struct.pack('>H', crc)
                   for t, off, ln, crc in entries)
    symbols = []
    last = 0
    i, n = 0, len(entries)
    while i < n:
        t = entries[i][0]
        if t != last:
            symbols.append(t)
            last = t
            i += 1
            continue
        run = 1
        while i + run < n and entries[i + run][0] == last:
            run += 1
        i += run
        while run:
            if run >= 19:
                k = min(run, 274)
                symbols += [chd.RLE_LARGE, (k - 19) >> 4, (k - 19) & 15]
            elif run >= 3:
                k = run
                symbols += [chd.RLE_SMALL, k - 3]
            else:
                k = 1
                symbols.append(last)
            run -= k
    freq = [0] * 16
    for s in symbols:
        freq[s] += 1
    lens = huffman_lengths(freq)
    codes = canonical(lens)
    bw = BitWriter()
    export_tree(bw, lens)
    for s in symbols:
        c, ln = codes[s]
        bw.write(c, ln)
    maxlen = max([ln for t, _, ln, _ in entries if t < 4] or [1])
    lengthbits = max(1, maxlen.bit_length())
    for t, off, ln, crc in entries:
        if t < 4:
            bw.write(ln, lengthbits)
            bw.write(crc, 16)
        elif t == 4:
            bw.write(crc, 16)
    body = bw.bytes()
    head = struct.pack('>I', len(body)) + struct.pack('>Q', first_offset)[2:] + struct.pack('>H', chd.crc16(raw)) + \
        bytes([lengthbits, 0, 0, 0])
    return head + body


def write(bin_path, out_path, track_type='MODE2_RAW', log=print, workers=None):
    size = os.path.getsize(bin_path)
    if size % chd.SECTOR:
        raise SystemExit('%s : taille qui n\'est pas un multiple de 2352' % bin_path)
    frames = size // chd.SECTOR
    pad = (-frames) % CD_TRACK_PADDING
    total = frames + pad
    hunks = (total + FRAMES_PER_HUNK - 1) // FRAMES_PER_HUNK
    logical = total * chd.FRAME
    meta = ('TRACK:1 TYPE:%s SUBTYPE:NONE FRAMES:%d PREGAP:0 PGTYPE:MODE1 PGSUB:NONE POSTGAP:0' % (track_type, frames)).encode() + b'\0'
    meta_entry = struct.pack('>4sIQ', b'CHT2', (META_CHECKSUM << 24) | len(meta), 0) + meta
    first = 124 + len(meta_entry)
    tmp = out_path + '.part'
    raw_sha = hashlib.sha1()
    entries = []
    step = 256
    jobs = [(bin_path, h, min(step, hunks - h), frames) for h in range(0, hunks, step)]
    with open(tmp, 'wb') as o, ProcessPoolExecutor(max_workers=workers) as ex:
        o.write(bytes(124))
        o.write(meta_entry)
        pos = first
        done = 0
        for res, raw in ex.map(_work, jobs):
            for kind, data, crc in res:
                o.write(data)
                entries.append((kind, pos, len(data) if kind != 4 else HUNK, crc))
                pos += len(data)
            raw_sha.update(raw)
            done += len(res)
            if done % (step * 20) < step:
                log('  CHD : %d %%' % (100 * done // hunks))
        mapoff = pos
        o.write(compress_map(entries, first))
        rawsha1 = raw_sha.digest()
        overall = hashlib.sha1(rawsha1 + b'CHT2' + hashlib.sha1(meta).digest()).digest()
        header = b'MComprHD' + struct.pack('>II', 124, 5) + b''.join(CODECS) + \
            struct.pack('>QQQII', logical, mapoff, 124, HUNK, chd.FRAME) + rawsha1 + overall + bytes(20)
        o.seek(0)
        o.write(header)
    os.replace(tmp, out_path)
    log('  CHD : %s, %d Mo (bin %d Mo)' % (os.path.basename(out_path), os.path.getsize(out_path) >> 20, size >> 20))


def verify(bin_path, chd_path, log=print):
    """Every sector of the CHD read back with chd.py, against the .bin."""
    c = chd.Chd(chd_path)
    frames = os.path.getsize(bin_path) // chd.SECTOR
    if c.frames != frames:
        raise SystemExit('nombre de secteurs différent')
    with open(bin_path, 'rb') as f:
        for h in range(c.hunks):
            b = c.hunk(h, check=True)
            for i in range(FRAMES_PER_HUNK):
                lba = h * FRAMES_PER_HUNK + i
                if lba < frames and b[i * chd.FRAME:i * chd.FRAME + chd.SECTOR] != f.read(chd.SECTOR):
                    raise SystemExit('secteur %d différent' % lba)
    log('  CHD relu : %d secteurs identiques au .bin' % frames)


if __name__ == '__main__':
    write(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'MODE2_RAW')
    verify(sys.argv[1], sys.argv[2])
