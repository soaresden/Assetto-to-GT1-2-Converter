"""PlayStation TIM pictures (4, 8, 16 bits per pixel, with or without CLUT): read to RGBA, and
write a new 4- or 8-bit picture of the same size and VRAM place.

  u32 0x10, u32 flags (bits 0-1: 0 = 4 bpp, 1 = 8 bpp, 2 = 16 bpp; bit 3: has a CLUT)
  CLUT block: u32 size, u16 x, y, w, h, the colours      image block: u32 size, u16 x, y, w (in
  16-bit words), h, the pixels
Colour 0x0000 is transparent; bit 15 is the semi-transparency flag.
"""
import struct

import numpy as np


def rgba(words):
    w = np.asarray(words, np.uint32)
    r = (w & 31) << 3
    g = (w >> 5 & 31) << 3
    b = (w >> 10 & 31) << 3
    a = np.where(w == 0, 0, 255)
    return np.stack([r, g, b, a], -1).astype(np.uint8)


class Tim:
    def __init__(self, data):
        magic, flags = struct.unpack_from('<II', data, 0)
        if magic != 0x10:
            raise ValueError('not a TIM')
        self.data = bytes(data)
        self.bpp = {0: 4, 1: 8, 2: 16, 3: 24}[flags & 3]
        self.has_clut = bool(flags & 8)
        o = 8
        self.clut = None
        if self.has_clut:
            n, cx, cy, cw, ch = struct.unpack_from('<I4H', data, o)
            self.clut_at = (cx, cy, cw, ch)
            self.clut = np.frombuffer(data, '<u2', cw * ch, o + 12).reshape(ch, cw).copy()
            self.clut_off = o + 12
            o += n
        n, x, y, w, h = struct.unpack_from('<I4H', data, o)
        self.img_at = (x, y, w, h)
        self.pix_off = o + 12
        raw = np.frombuffer(data, np.uint8, w * h * 2, o + 12)
        if self.bpp == 4:
            self.index = np.stack([raw & 15, raw >> 4], -1).reshape(h, w * 4)
        elif self.bpp == 8:
            self.index = raw.reshape(h, w * 2)
        else:
            self.index = None
            self.words = raw.view('<u2').reshape(h, w)
        self.end = o + n

    @property
    def size(self):
        return (self.index if self.index is not None else self.words).shape[::-1]

    def image(self, clut_row=0):
        if self.index is None:
            return rgba(self.words)
        return rgba(self.clut[clut_row])[self.index]

    def with_pixels(self, index, clut=None):
        """Same TIM with new indices (and CLUT rows), same sizes."""
        d = bytearray(self.data)
        if clut is not None:
            c = np.asarray(clut, '<u2')
            d[self.clut_off:self.clut_off + c.nbytes] = c.tobytes()
        idx = np.asarray(index, np.uint8)
        if self.bpp == 4:
            raw = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
        else:
            raw = idx
        d[self.pix_off:self.pix_off + raw.nbytes] = raw.tobytes()
        return bytes(d)
