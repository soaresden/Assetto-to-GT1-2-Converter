"""The car name pictures of the Gran Turismo (PS1) menus (MENU_IMG.ARC, one picture per car in each
language: the big name of the car's page, white letters on transparent, 256x48, 4 bits).

New names are written with the game's own letters: every name picture of the disc is cut into
letters (columns with nothing between them) and matched with the car's name in CARINF.DAT; a
letter seen several times keeps its most common drawing.
"""
from collections import Counter, defaultdict

import numpy as np

import tim


def blank(picture):
    """Index of the transparent colour (0x0000) of a picture's CLUT (15 in the name pictures)."""
    z = np.nonzero(picture.clut[0] == 0)[0]
    return int(z[0]) if len(z) else 0


def columns(index, empty=15):
    """(start, end) of the runs of columns with something drawn."""
    used = (index != empty).any(0)
    runs, x = [], 0
    w = len(used)
    while x < w:
        if used[x]:
            s = x
            while x < w and used[x]:
                x += 1
            runs.append((s, x))
        else:
            x += 1
    return runs


class Font:
    def __init__(self):
        self.glyphs = defaultdict(Counter)   # char -> Counter of glyph bytes
        self.shapes = {}
        self.gaps = []
        self.spaces = []
        self.height = None
        self.blank = 15

    def learn(self, picture, text):
        idx = picture.index
        self.blank = blank(picture)
        runs = columns(idx, self.blank)
        chars = [c for c in text if c != ' ']
        if len(runs) != len(chars) or not runs:
            return False
        self.height = idx.shape[0]
        for (s, e), c in zip(runs, chars):
            g = idx[:, s:e]
            key = g.tobytes() + bytes([e - s])
            self.glyphs[c][key] += 1
            self.shapes[key] = g.copy()
        # spacing: gaps between letters, and the gap where the text has a space
        k = 0
        words = text.split(' ')
        pos = 0
        for wi, wd in enumerate(words):
            for ci in range(len(wd)):
                if ci + 1 < len(wd):
                    self.gaps.append(runs[pos + 1][0] - runs[pos][1])
                pos += 1
            if wi + 1 < len(words) and pos < len(runs):
                self.spaces.append(runs[pos][0] - runs[pos - 1][1])
        return True

    def glyph(self, c):
        if c not in self.glyphs:
            raise KeyError('letter %r not found in the game' % c)
        return self.shapes[self.glyphs[c].most_common(1)[0][0]]

    def has(self, text):
        return all(c == ' ' or c in self.glyphs for c in text)

    def render(self, text, width, x0=0):
        gap = int(np.median(self.gaps)) if self.gaps else 1
        space = int(np.median(self.spaces)) if self.spaces else 8
        out = np.full((self.height, width), self.blank, np.uint8)
        x = x0
        for i, c in enumerate(text):
            if c == ' ':
                x += space - gap
                continue
            g = self.glyph(c)
            w = g.shape[1]
            if x + w > width:
                raise ValueError('%r does not fit in %d pixels' % (text, width))
            out[:, x:x + w] = np.where(g != self.blank, g, out[:, x:x + w])
            x += w + gap
        return out


def learn_font(pictures, names):
    """pictures: {index: TIM bytes}; names: {index: text}."""
    f = Font()
    n = 0
    for i, b in pictures.items():
        if i in names:
            try:
                t = tim.Tim(b)
            except ValueError:
                continue
            if t.bpp == 4 and f.learn(t, names[i]):
                n += 1
    return f, n


def luminance(clut_row):
    w = np.asarray(clut_row, np.uint32)
    return (((w & 31) + (w >> 5 & 31) + (w >> 10 & 31)) / 93.0).astype(np.float32)


def emblem_title(picture, emblem, font, text, height=34, gap=10):
    """A name picture like `picture` (same size and CLUT): the emblem (RGBA array, white on
    transparent), then `text` in the game's letters at the emblem's height, centred."""
    from PIL import Image
    W, H = picture.size
    lum = luminance(picture.clut[0])
    empty = blank(picture)
    if emblem.ndim == 2:          # coverage 0..1 already
        a = np.asarray(emblem, np.float32)
    else:                         # RGBA picture, white on transparent
        a = emblem[..., 3].astype(np.float32) / 255 * (emblem[..., :3].mean(2) / 255)
    ys, xs = np.nonzero(a > 0.05)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    eh = height
    ew = int(round(a.shape[1] * eh / a.shape[0]))
    parts = [np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((ew, eh), Image.LANCZOS), np.float32) / 255]
    if text:
        t = font.render(text, 256)
        tl = np.where(t == font.blank, 0.0, lum[t])
        cols = np.nonzero(tl.max(0) > 0)[0]
        rows = np.nonzero(tl.max(1) > 0)[0]
        tl = tl[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        th = int(round(height * 0.72))
        tw = int(round(tl.shape[1] * th / tl.shape[0]))
        parts.append(np.asarray(Image.fromarray((tl * 255).astype(np.uint8)).resize((tw, th), Image.LANCZOS), np.float32) / 255)
    total = sum(p.shape[1] for p in parts) + gap * (len(parts) - 1)
    if total > W:  # too wide: everything smaller
        k = W / total * 0.98
        parts = [np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize((max(1, int(p.shape[1] * k)), max(1, int(p.shape[0] * k))), Image.LANCZOS), np.float32) / 255 for p in parts]
        total = sum(p.shape[1] for p in parts) + int(gap * k) * (len(parts) - 1)
        gap = int(gap * k)
    canvas = np.zeros((H, W), np.float32)
    x = (W - total) // 2
    base = (H + parts[0].shape[0]) // 2          # bottoms aligned on the emblem's
    for p in parts:
        y = base - p.shape[0]
        canvas[y:y + p.shape[0], x:x + p.shape[1]] = np.maximum(canvas[y:y + p.shape[0], x:x + p.shape[1]], p)
        x += p.shape[1] + gap
    # nearest grey of the CLUT; transparent where nothing is drawn
    choices = [i for i in range(16) if i != empty]
    cl = lum[choices]
    idx = np.array(choices, np.uint8)[np.abs(canvas[..., None] - cl[None, None]).argmin(2)]
    idx[canvas < 0.06] = empty
    return idx


def plain_font(picture, text):
    """The letters of one name picture in the plain font (e.g. 'EUNOS ROADSTER'), with an F made
    from the E (its lower bar taken off) when the name has no F."""
    f = Font()
    f.blank = blank(picture)
    f.height = picture.index.shape[0]
    idx = picture.index
    lum = luminance(picture.clut[0])
    lum[f.blank] = 0
    used = (lum[idx] > 0.3).any(0)
    runs, x = [], 0
    while x < len(used):
        if used[x]:
            s0 = x
            while x < len(used) and used[x]:
                x += 1
            runs.append((s0, x))
        else:
            x += 1
    chars = [c for c in text if c != ' ']
    widths = np.array([e - s0 for s0, e in runs])
    wide = 1.6 * np.median(widths)
    # letters that touch make one wide run: take the single letters from both ends only
    pairs = []
    i = 0
    while i < min(len(runs), len(chars)) and widths[i] < wide:
        pairs.append((runs[i], chars[i]))
        i += 1
    j = 1
    while j <= min(len(runs), len(chars)) - i and widths[-j] < wide:
        pairs.append((runs[-j], chars[-j]))
        j += 1
    if not pairs:
        raise ValueError('cannot cut %r into letters' % text)
    for (s0, e), c in pairs:
        g = np.where(lum[idx[:, s0:e]] > 0, idx[:, s0:e], f.blank).astype(np.uint8)
        key = g.tobytes() + bytes([e - s0])
        f.glyphs[c][key] += 1
        f.shapes[key] = g
    f.gaps = [runs[k + 1][0] - runs[k][1] for k in range(min(3, len(runs) - 1))]
    if 'F' not in f.glyphs and 'E' in f.glyphs:
        e = f.glyph('E').copy()
        drawn = e != f.blank
        rows = np.nonzero(drawn.any(1))[0]
        stem = drawn[rows].mean(0) > 0.8           # the columns of the upright
        last = stem.nonzero()[0].max() + 1 if stem.any() else 2
        # the lower bar: the drawn rows at the bottom that go past the upright
        bottom = rows.max()
        r = bottom
        while r >= rows.min() and drawn[r, last:].any():
            e[r, last:] = f.blank
            r -= 1
        key = e.tobytes() + bytes([e.shape[1]])
        f.glyphs['F'][key] += 1
        f.shapes[key] = e
    return f
