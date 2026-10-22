"""Gran Turismo (PS1) menu pages (.slt in MENU_IMG.ARC): a GT-ARC of a 16-colour CLUT and six
4-bit blocks, the pieces of a 640x480 picture as they go in VRAM:

  top row (256 high):    256 wide | 256 wide | 128 wide
  bottom row (224 high): 256 wide | 256 wide | 128 wide
"""
import numpy as np

import gtarc

BLOCKS = [(0, 0, 256, 256), (256, 0, 256, 256), (512, 0, 128, 256),
          (0, 256, 256, 224), (256, 256, 256, 224), (512, 256, 128, 224)]


class Page:
    def __init__(self, data):
        self.arc = gtarc.Arc(data)
        self.clut = np.frombuffer(self.arc[0], '<u2').copy()
        self.index = np.zeros((480, 640), np.uint8)
        for k, (x, y, w, h) in enumerate(BLOCKS):
            raw = np.frombuffer(self.arc[k + 1], np.uint8)
            if raw.size * 2 != w * h:
                raise ValueError('block %d: %d bytes for %dx%d' % (k, raw.size, w, h))
            self.index[y:y + h, x:x + w] = np.stack([raw & 15, raw >> 4], -1).reshape(h, w)

    def write(self):
        files = [self.clut.astype('<u2').tobytes()]
        for x, y, w, h in BLOCKS:
            b = self.index[y:y + h, x:x + w]
            files.append((b[:, 0::2] | (b[:, 1::2] << 4)).astype(np.uint8).tobytes())
        new = {k: f for k, f in enumerate(files) if f != self.arc[k]}
        return gtarc.replace(self.arc, new, align=4) if new else self.arc.data

    def rgb(self):
        w = self.clut.astype(np.uint32)
        pal = np.stack([(w & 31) << 3, (w >> 5 & 31) << 3, (w >> 10 & 31) << 3], -1).astype(np.uint8)
        return pal[self.index]
