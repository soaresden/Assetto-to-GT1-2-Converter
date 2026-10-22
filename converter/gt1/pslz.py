"""Gran Turismo (PS1) executables (GTMENU.EXE, GTMAIN.EXE...): PS-X EXE files whose code is
compressed with the GT-ARC LZSS, written backwards ("PSLZ"), and unpacked in place by a small
loader at the entry point.

The 0x1C-byte PSLZ header sits just before the loader (layout from pez2k's GT1ExecutableTool):
"PSLZ", end of the unpacked data (RAM address), unpacked size, end of the packed data (RAM
address), loader address, loader size, start of the packed data (RAM address).
"""
import struct

import gtarc

HEADER = 0x800


class Exe:
    def __init__(self, data):
        if data[:8] != b'PS-X EXE':
            raise ValueError('not a PS-X EXE')
        self.data = bytearray(data)
        self.entry, = struct.unpack_from('<I', data, 0x10)
        self.load, = struct.unpack_from('<I', data, 0x18)
        self.base = self.load - HEADER         # RAM address of file offset 0
        self.hdr = self.entry - self.base - 0x1C
        if data[self.hdr:self.hdr + 4] != b'PSLZ':
            raise ValueError('no PSLZ header')
        (self.out_end, self.out_size, self.in_end, self.loader, self.loader_size,
         self.in_start) = struct.unpack_from('<IiIIII', data, self.hdr + 4)

    def unpack(self):
        """The unpacked code (to load at out_end - out_size + 1... as the loader writes it)."""
        a, b = self.in_start - self.base, self.in_end - self.base + 1
        packed = bytes(self.data[a:b])[::-1]
        return gtarc.decompress(packed, self.out_size)[::-1]

    def repack(self, code):
        """The executable with new unpacked code of the same size: the packed data ends where it
        ended before and must start no earlier than it did (so the loader and the rest stay)."""
        if len(code) != self.out_size:
            raise ValueError('code size changed')
        packed = gtarc.compress(code[::-1])[::-1]
        old = self.in_end - self.in_start + 1
        if len(packed) > old:
            raise ValueError('packed code %d bytes, room for %d' % (len(packed), old))
        d = bytearray(self.data)
        a = self.in_end - self.base + 1 - len(packed)
        d[a:a + len(packed)] = packed
        struct.pack_into('<I', d, self.hdr + 0x18, self.in_start + (old - len(packed)))
        return bytes(d)
