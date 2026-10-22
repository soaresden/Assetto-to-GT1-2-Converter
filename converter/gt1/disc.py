"""The PlayStation disc: files of its ISO 9660 file system, and a .bin/.cue copy of it with some
files replaced in place (same sectors), made from the CHD the player has.

Sectors are mode 2 form 1 (2048 bytes of data): a replaced sector keeps its header and subheader
and gets its EDC and ECC computed again.
"""
import os
import struct

import numpy as np

import binimage
import chd

SECTOR = chd.SECTOR


def _edc_table():
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ (0xD8018001 if c & 1 else 0)
        t.append(c)
    return t


_EDC_TABLE = _edc_table()
_EDC_LEN = 2056          # bytes 16..2071 of a mode 2 form 1 sector
_EDC_POS = None


def _edc_pos():
    """The EDC is linear in the data (it starts at 0): what byte value v at position k adds,
    for every k and v, so that a sector's EDC is the xor of 2056 table entries."""
    t = np.array(_EDC_TABLE, np.uint32)
    pos = np.empty((_EDC_LEN, 256), np.uint32)
    c = t.copy()                 # the last byte: one step
    for k in range(_EDC_LEN - 1, -1, -1):
        pos[k] = c
        c = t[c & 0xFF] ^ (c >> 8)   # one more (zero) byte after it
    return pos


def edc_many(data):
    """CD-ROM EDC (CRC-32 with the polynomial 0xD8018001, reflected) of rows of 2056 bytes."""
    global _EDC_POS
    if _EDC_POS is None:
        _EDC_POS = _edc_pos()
    return np.bitwise_xor.reduce(_EDC_POS[np.arange(_EDC_LEN), data], axis=1)


def edc(data):
    """CD-ROM EDC: CRC-32 with the polynomial 0xD8018001 (reflected), table driven."""
    crc = 0
    t = _EDC_TABLE
    for b in data:
        crc = t[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc


def make_sectors(originals, datas):
    """Mode 2 form 1 sectors with the header and subheader of the originals and 2048 bytes of
    data each (shorter data is padded with zeros), EDC and ECC computed again."""
    n = len(originals)
    s = np.zeros((n, SECTOR), np.uint8)
    s[:, :24] = np.frombuffer(b''.join(o[:24] for o in originals), np.uint8).reshape(n, 24)
    s[:, 24:2072] = np.frombuffer(b''.join(bytes(d).ljust(2048, b'\0') for d in datas), np.uint8).reshape(n, 2048)
    s[:, 2072:2076] = edc_many(s[:, 16:2072]).astype('<u4').view(np.uint8).reshape(n, 4)
    chd.ecc_generate_many(s)
    return [r.tobytes() for r in s]


def make_sector(original, data):
    """Mode 2 form 1 sector with the header and subheader of `original` and 2048 bytes of data."""
    return make_sectors([original], [data])[0]


class Disc:
    def __init__(self, chd_file):
        # the player's image: .chd, or .bin / .cue
        self.c = binimage.open_image(chd_file)
        self.files = {}
        pvd = self.c.user_data(16)
        root = pvd[156:190]
        self._walk(struct.unpack_from('<I', root, 2)[0], struct.unpack_from('<I', root, 10)[0], '')

    def _walk(self, lba, size, path):
        d = self.c.read(lba, (size + 2047) // 2048)
        o = 0
        while o < len(d):
            n = d[o]
            if n == 0:
                o = (o // 2048 + 1) * 2048
                continue
            e_lba, = struct.unpack_from('<I', d, o + 2)
            e_size, = struct.unpack_from('<I', d, o + 10)
            flags, nl = d[o + 25], d[o + 32]
            name = d[o + 33:o + 33 + nl]
            if name not in (b'\0', b'\1'):
                nm = name.decode('ascii').split(';')[0]
                full = path + '/' + nm
                if flags & 2:
                    self._walk(e_lba, e_size, full)
                else:
                    # where the size is written in the directory, to change it if needed
                    self.files[full.lstrip('/')] = (e_lba, e_size, lba + (o // 2048), o % 2048)
            o += n

    def read(self, name):
        lba, size = self.files[name][:2]
        return self.c.read(lba, (size + 2047) // 2048)[:size]

    def room(self, name):
        """Bytes the file can grow to without moving anything: up to the next file's first sector."""
        lba = self.files[name][0]
        after = min((f[0] for f in self.files.values() if f[0] > lba), default=self.c.frames)
        return (after - lba) * 2048

    def bin_chunks(self, replace, chunk=2048, progress=None):
        """The .bin (raw 2352-byte sectors), `chunk` sectors at a time; replace = {file name: new
        bytes} (each must fit its room)."""
        sectors = {}
        dir_patch = {}
        for name, data in replace.items():
            lba, size, dlba, doff = self.files[name]
            if len(data) > self.room(name):
                raise ValueError('%s: %d bytes, room for %d' % (name, len(data), self.room(name)))
            for i in range((len(data) + 2047) // 2048):
                sectors[lba + i] = (data, i * 2048)
            if len(data) != size:
                dir_patch.setdefault(dlba, []).append((doff, len(data)))
        total = self.c.frames
        for start in range(0, total, chunk):
            raws = [self.c.sector(lba) for lba in range(start, min(total, start + chunk))]
            todo, datas = [], []
            for k, raw in enumerate(raws):
                lba = start + k
                if lba in sectors:
                    data, o = sectors[lba]
                    datas.append(data[o:o + 2048])
                elif lba in dir_patch:
                    d = bytearray(raw[24:24 + 2048])
                    for off, n in dir_patch[lba]:
                        struct.pack_into('<I', d, off + 10, n)
                        struct.pack_into('>I', d, off + 14, n)
                    datas.append(bytes(d))
                else:
                    continue
                todo.append(k)
            if todo:
                for k, new in zip(todo, make_sectors([raws[k] for k in todo], datas)):
                    raws[k] = new
            yield b''.join(raws)
            if progress:
                progress(start + len(raws), total)

    @staticmethod
    def cue(bin_name):
        """The .cue of a .bin called `bin_name` (one data track)."""
        return 'FILE "%s" BINARY\r\n  TRACK 01 MODE2/2352\r\n    INDEX 01 00:00:00\r\n' % bin_name

    def write_bin(self, path, replace, progress=None):
        """Write the .bin (raw 2352-byte sectors) and its .cue; replace = {file name: new bytes}
        (each must fit its room)."""
        # written under a temporary name and renamed once complete: a half-written image never
        # looks like a finished one
        part = path + '.part'
        with open(part, 'wb') as f:
            for b in self.bin_chunks(replace, progress=progress):
                f.write(b)
        os.replace(part, path)
        with open(os.path.splitext(path)[0] + '.cue', 'w', newline='') as f:
            f.write(self.cue(os.path.basename(path)))
