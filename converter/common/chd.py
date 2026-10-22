"""Read CD images in MAME's CHD format (version 5), as made by `chdman createcd`.

The standard library (zlib, lzma) and numpy, so it also runs in the browser (Pyodide). Gives the raw
2352-byte sectors of the disc; `Chd.sector(lba)` and `Chd.user_data(lba)` (the 2048 bytes of a
mode 1 / mode 2 form 1 sector).

Layout (from MAME's chd.cpp / libchdr): a 124-byte header, metadata entries (CHT2 lines describe
the tracks), a Huffman-coded map giving for each hunk its codec, offset, length and CRC16, and the
hunks. A CD hunk holds `hunkbytes / 2448` frames of 2352 bytes of sector + 96 bytes of subcode;
the cdlz / cdzl / cdfl codecs compress the sectors and the subcodes separately, and may strip the
sync pattern and the ECC of sectors whose ECC could be rebuilt (flag bit per frame).
"""
import binascii
import lzma
import struct
import zlib

import numpy as np

SECTOR = 2352
SUBCODE = 96
FRAME = SECTOR + SUBCODE
SYNC = b'\x00' + b'\xff' * 10 + b'\x00'

# map entry types
C0, C1, C2, C3, NONE, SELF, PARENT, RLE_SMALL, RLE_LARGE, SELF_0, SELF_1, PARENT_SELF, PARENT_0, PARENT_1 = range(14)


class Bits:
    """MSB-first bit reader; reads past the end give zeros (as in libchdr)."""

    def __init__(self, data):
        self.data = data
        self.pos = 0  # in bits

    def read(self, n):
        if n == 0:
            return 0
        v = 0
        for _ in range(n):
            i = self.pos >> 3
            b = self.data[i] if i < len(self.data) else 0
            v = (v << 1) | ((b >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def peek(self, n):
        p = self.pos
        v = self.read(n)
        self.pos = p
        return v


class Huffman:
    """libchdr's huffman_decoder: canonical codes, tree imported in its RLE form."""

    def __init__(self, numcodes, maxbits, bits):
        self.maxbits = maxbits
        nb = 5 if maxbits >= 16 else 4 if maxbits >= 8 else 3
        lens = []
        while len(lens) < numcodes:
            n = bits.read(nb)
            if n != 1:
                lens.append(n)
            else:
                n = bits.read(nb)
                if n == 1:
                    lens.append(n)
                else:
                    lens += [n] * (bits.read(nb) + 3)
        lens = lens[:numcodes]
        histo = [0] * 33
        for n in lens:
            histo[n] += 1
        start = 0
        for n in range(32, 0, -1):
            nxt = (start + histo[n]) >> 1
            histo[n] = start
            start = nxt
        self.table = [None] * (1 << maxbits)
        for sym, n in enumerate(lens):
            if n:
                code = histo[n]
                histo[n] += 1
                shift = maxbits - n
                for k in range(code << shift, (code + 1) << shift):
                    self.table[k] = (sym, n)

    def decode(self, bits):
        sym, n = self.table[bits.peek(self.maxbits)]
        bits.pos += n
        return sym


def crc16(data):
    """CRC-16/CCITT (poly 0x1021, init 0xFFFF), used for the hunks: binascii's crc_hqx, in C."""
    return binascii.crc_hqx(data, 0xFFFF)


def _lzma_props(size):
    # chdman's LZMA: level 8 defaults (lc 3, lp 0, pb 2), dictionary reduced to the hunk size
    d = 1 << 24
    if d > size:
        for i in range(11, 31):
            if size <= 2 << i:
                d = 2 << i
                break
            if size <= 3 << i:
                d = 3 << i
                break
    return d


def inflate(data, size):
    return zlib.decompressobj(-15).decompress(data, size)


def unlzma(data, size):
    filt = [{'id': lzma.FILTER_LZMA1, 'dict_size': _lzma_props(size), 'lc': 3, 'lp': 0, 'pb': 2}]
    d = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=filt)
    out = d.decompress(data + b'\x00' * 16, size)
    return out


class Chd:
    def __init__(self, f):
        self.f = open(f, 'rb') if isinstance(f, str) else f
        h = self._at(0, 124)
        if h[:8] != b'MComprHD' or struct.unpack('>I', h[12:16])[0] != 5:
            raise ValueError('not a CHD v5 file')
        self.codecs = [h[16 + 4 * i:20 + 4 * i] for i in range(4)]
        self.logical, mapoff, metaoff = struct.unpack('>QQQ', h[32:56])
        self.hunkbytes, self.unitbytes = struct.unpack('>II', h[56:64])
        if any(h[104:124]):
            raise ValueError('CHD with a parent: not supported')
        self.hunks = (self.logical + self.hunkbytes - 1) // self.hunkbytes
        self.frames_per_hunk = self.hunkbytes // FRAME
        self.meta = []
        o = metaoff
        while o:
            tag, fl, nxt = struct.unpack('>IIQ', self._at(o, 16))
            self.meta.append((struct.pack('>I', tag).decode('latin-1'), self._at(o + 16, fl & 0xFFFFFF)))
            o = nxt
        self.tracks = []
        for tag, v in self.meta:
            if tag in ('CHT2', 'CHTR'):
                t = dict(kv.split(':', 1) for kv in v.rstrip(b'\0').decode().split())
                self.tracks.append(t)
        self._read_map(mapoff)
        self._cache = (None, None)

    def _at(self, off, n):
        self.f.seek(off)
        return self.f.read(n)

    def _read_map(self, mapoff):
        mh = self._at(mapoff, 16)
        mapbytes, = struct.unpack('>I', mh[:4])
        first = int.from_bytes(mh[4:10], 'big')
        lengthbits, selfbits, parentbits = mh[12], mh[13], mh[14]
        bits = Bits(self._at(mapoff + 16, mapbytes))
        huff = Huffman(16, 8, bits)
        types = []
        last = 0
        rep = 0
        for _ in range(self.hunks):
            if rep:
                types.append(last)
                rep -= 1
                continue
            v = huff.decode(bits)
            if v == RLE_SMALL:
                types.append(last)
                rep = 2 + huff.decode(bits)
            elif v == RLE_LARGE:
                types.append(last)
                rep = 2 + 16 + (huff.decode(bits) << 4)
                rep += huff.decode(bits)
            else:
                types.append(v)
                last = v
        self.map = []
        cur = first
        last_self = last_parent = 0
        for n, t in enumerate(types):
            off, length, crc = cur, 0, None
            if t in (C0, C1, C2, C3):
                length = bits.read(lengthbits)
                cur += length
                crc = bits.read(16)
            elif t == NONE:
                length = self.hunkbytes
                cur += length
                crc = bits.read(16)
            elif t == SELF:
                last_self = off = bits.read(selfbits)
            elif t == PARENT:
                last_parent = off = bits.read(parentbits)
            elif t in (SELF_0, SELF_1):
                if t == SELF_1:
                    last_self += 1
                t, off = SELF, last_self
            elif t == PARENT_SELF:
                t = PARENT
                last_parent = off = n * self.hunkbytes // self.unitbytes
            elif t in (PARENT_0, PARENT_1):
                if t == PARENT_1:
                    last_parent += self.hunkbytes // self.unitbytes
                t, off = PARENT, last_parent
            self.map.append((t, off, length, crc))

    def hunk(self, n, check=False):
        if self._cache[0] == n:
            return self._cache[1]
        t, off, length, crc = self.map[n]
        if t == SELF:
            return self.hunk(off, check)
        if t == PARENT:
            raise ValueError('hunk %d is in the parent CHD' % n)
        src = self._at(off, length)
        if t == NONE:
            out = src
        else:
            out = self._decode(self.codecs[t], src)
        if check and crc16(out) != crc:
            raise ValueError('hunk %d: bad CRC' % n)
        self._cache = (n, out)
        return out

    def _decode(self, codec, src):
        frames = self.frames_per_hunk
        dest = self.hunkbytes
        if codec not in (b'cdlz', b'cdzl', b'cdfl'):
            raise ValueError('codec %r not supported' % codec)
        if codec == b'cdfl':
            # FLAC is only chosen for audio; the sectors are FLAC, the subcodes deflate
            raise NotImplementedError('cdfl (FLAC) hunk: CD audio tracks are not supported')
        cb = 2 if dest < 65536 else 3
        eb = (frames + 7) // 8
        hb = eb + cb
        base = int.from_bytes(src[eb:eb + cb], 'big')
        size = frames * SECTOR
        sect = (unlzma if codec == b'cdlz' else inflate)(src[hb:hb + base], size)
        sub = inflate(src[hb + base:], frames * SUBCODE)
        sect = np.frombuffer(sect.ljust(size, b'\0')[:size], np.uint8).reshape(frames, SECTOR).copy()
        flags = np.unpackbits(np.frombuffer(src[:eb], np.uint8), bitorder='little')[:frames].astype(bool)
        if flags.any():
            # sync pattern and ECC removed by chdman: put back
            fixed = sect[flags]
            fixed[:, 0:12] = np.frombuffer(SYNC, np.uint8)
            ecc_generate_many(fixed)
            sect[flags] = fixed
        out = np.zeros(dest, np.uint8)
        fr = out[:frames * FRAME].reshape(frames, FRAME)
        fr[:, :SECTOR] = sect
        sub = sub.ljust(frames * SUBCODE, b'\0')[:frames * SUBCODE]
        fr[:, SECTOR:] = np.frombuffer(sub, np.uint8).reshape(frames, SUBCODE)
        return out.tobytes()

    @property
    def frames(self):
        return sum(int(t['FRAMES']) for t in self.tracks)

    def sector(self, lba):
        """The raw 2352 bytes of frame `lba` (first track starts at 0)."""
        h, i = divmod(lba, self.frames_per_hunk)
        b = self.hunk(h)
        return b[i * FRAME:i * FRAME + SECTOR]

    def user_data(self, lba):
        s = self.sector(lba)
        if s[15] == 2:
            if s[18] & 0x20:  # mode 2 form 2: 2324 bytes
                return s[24:24 + 2324]
            return s[24:24 + 2048]
        return s[16:16 + 2048]

    def read(self, lba, count):
        return b''.join(self.user_data(lba + i) for i in range(count))


# ---- CD-ROM ECC (P and Q parity), as MAME's cdrom.cpp / ECM compute it ------------------------
_ECC_F = [((i << 1) ^ (0x11D if i & 0x80 else 0)) & 0xFF for i in range(256)]
_ECC_B = [0] * 256
for _i in range(256):
    _ECC_B[_i ^ _ECC_F[_i]] = _i


def _ecc_plan(major_count, minor_count, major_mult, minor_inc):
    """Which byte (from offset 12) goes in each step of each parity byte, and the multiplication
    by the power of alpha it ends up with (each step: a ^= t, b ^= t, a = F[a])."""
    size = major_count * minor_count
    idx = np.empty((major_count, minor_count), np.int32)
    for major in range(major_count):
        k = (major >> 1) * major_mult + (major & 1)
        for j in range(minor_count):
            idx[major, j] = k
            k += minor_inc
            if k >= size:
                k -= size
    f = np.array(_ECC_F, np.uint8)
    mul = np.empty((minor_count, 256), np.uint8)
    t = f.copy()                       # multiplication by alpha^1
    for j in range(minor_count - 1, -1, -1):
        mul[j] = t                     # step j is followed by minor_count - j multiplications
        t = f[t]
    return idx, mul, major_count


_ECC_P = _ecc_plan(86, 24, 2, 86)
_ECC_Q = _ecc_plan(52, 43, 86, 88)
_ECC_FA = np.array(_ECC_F, np.uint8)
_ECC_BA = np.array(_ECC_B, np.uint8)


def _ecc_parity(data, plan):
    """The parity bytes (a of each major, then a ^ b) of sectors data[:, 12:], all at once."""
    idx, mul, count = plan
    v = data[:, 12:][:, idx]                                   # (sectors, major, minor)
    a = np.bitwise_xor.reduce(mul[np.arange(idx.shape[1]), v], axis=2)
    b = np.bitwise_xor.reduce(v, axis=2)
    a = _ECC_BA[_ECC_FA[a] ^ b]
    return np.concatenate([a, a ^ b], 1)


def ecc_generate_many(s):
    """Rebuild the P and Q parity of raw sectors (uint8 array, sectors x 2352) in place. For mode
    2 form 1 the parity is computed with the 4 header bytes taken as zero."""
    hdr = s[:, 12:16].copy()
    s[s[:, 15] == 2, 12:16] = 0
    s[:, 2076:2076 + 172] = _ecc_parity(s, _ECC_P)
    s[:, 2248:2248 + 104] = _ecc_parity(s, _ECC_Q)
    s[:, 12:16] = hdr


def ecc_generate(s):
    """Rebuild the P and Q parity of one raw sector (bytearray) in place."""
    a = np.frombuffer(bytes(s), np.uint8).reshape(1, SECTOR).copy()
    ecc_generate_many(a)
    s[:] = a.tobytes()


def edc(data):
    """CD-ROM EDC (CRC32, poly 0xD8018001 reflected)."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0xD8018001 if crc & 1 else 0)
    return crc


if __name__ == '__main__':
    import sys
    c = Chd(sys.argv[1])
    print(c.codecs, c.hunks, 'hunks of', c.hunkbytes, c.tracks)
    from collections import Counter
    print(Counter(c.codecs[t][:4] if t < 4 else t for t, *_ in c.map))
