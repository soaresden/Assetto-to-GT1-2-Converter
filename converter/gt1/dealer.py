"""The car tiles of the Gran Turismo (PS1) dealer pages (mazda-new.slt...): orange boxes with a
picture of the car, its name and trim in black letters, and its price. The pages are pictures
(slt.Page), the same layout in the six languages but with their own CLUT order.

Tile layout (pixels from the tile's top left corner, measured on the Mazda page):
  picture  x 8..66,  y 8..57          name  x 74..247, y 9..23
  trim     y 28..41                   price x 94..247, y 47..61 (after the Cr coin)
"""
import numpy as np

COLUMNS = (94, 350)       # left edge of the tiles of each column
TILE_W, TILE_H = 248, 63
PICTURE = (8, 8, 67, 58)
NAME = (74, 9, 247, 24)
TRIM = (74, 28, 247, 42)
PRICE = (94, 46, 247, 62)


def colours(page):
    """(orange, black) indices of the page's CLUT."""
    w = page.clut.astype(np.int64)
    r, g, b = w & 31, w >> 5 & 31, w >> 10 & 31
    orange = int(np.argmin((r - 23) ** 2 + (g - 15) ** 2 + (b - 0) ** 2))
    black = int(np.argmin(np.where(w == 0, 99, r + g + b)))
    return orange, black


def tiles(page):
    """Top left corners of the tiles, column by column, top to bottom."""
    orange, _ = colours(page)
    out = []
    for x0 in COLUMNS:
        col = page.index[:, x0 + 3] == orange
        y = 0
        while y < len(col):
            if col[y]:
                s = y
                while y < len(col) and col[y]:
                    y += 1
                if y - s >= TILE_H - 4:
                    out.append((x0, s))
            else:
                y += 1
    return out


NONE = 255   # no pixel, in a glyph patch


def glyphs(page, x0, y0, box):
    """Black letters of a box of a tile, from left to right: list of (x, patch), the patch being
    the letter's pixels with their anti-aliasing (indices of the page's CLUT, NONE elsewhere)."""
    orange, black = colours(page)
    bx0, by0, bx1, by1 = box
    sub = page.index[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1]
    m = sub == black
    used = m.any(0)
    out, x = [], 0
    while x < len(used):
        if used[x]:
            s = x
            while x < len(used) and used[x]:
                x += 1
            a, b = max(s - 1, 0), min(x + 1, sub.shape[1])
            patch = np.where(sub[:, a:b] == orange, NONE, sub[:, a:b]).astype(np.uint8)
            # keep only the edge columns' pixels that touch this letter
            out.append((a, patch))
        else:
            x += 1
    return out


# ---------------------------------------------------------------------------------------------
# The Mazda new car page (mazda-new.slt, number 529 of each language of MENU_IMG.ARC) and its second
# page (530): their tiles, column by column, and the texts the letters and digits are taken from.
MAZDA_NEW, MAZDA_NEW2 = 529, 530
PAGE_CARS = ['adlxn', 'adgln', 'ademn', 'altsn', 'aminn', 'amisn', 'amivn', 'afb7n']
TRIMS = {4: 'NORMAL', 5: 'S-SPECIAL', 2: 'GL-X'}
EIGHT = (0, 1)       # page 2, tile 0 (RX-7 Touring X, 38 150): its second digit is an 8


def tight(p):
    rows = np.nonzero((p != NONE).any(1))[0]
    return p[rows.min():rows.max() + 1], rows.max()


class Ink:
    """The black letters and digits of the page, as masks (bottom row = baseline)."""

    def __init__(self, page, page2, prices):
        t = tiles(page)
        self.letters = {}
        for k, txt in TRIMS.items():
            g = glyphs(page, *t[k], TRIM)
            if len(g) == len(txt):
                for (_, m), c in zip(g, txt):
                    self.letters.setdefault(c, tight(m)[0])
        self.digits = {}
        self.sep = None
        self.base = None
        gaps = []
        for k, pr in enumerate(prices):
            g = glyphs(page, *t[k], PRICE)
            if len(g) == len(pr) + 1:                    # a comma between the thousands
                comma = g[len(pr) - 3]
                self.sep = tight(comma[1])[0], comma
                g = g[:len(pr) - 3] + g[len(pr) - 2:]
            if len(g) != len(pr):
                continue
            for (x, m), c in zip(g, pr):
                mm, bottom = tight(m)
                self.digits.setdefault(c, mm)
                self.base = bottom if self.base is None else max(self.base, bottom)
            for (x, m), (x2, _) in zip(g, g[1:]):
                gaps.append(x2 - x - m.shape[1])
        g2 = glyphs(page2, *tiles(page2)[EIGHT[0]], PRICE)
        self.digits['8'] = tight(g2[EIGHT[1]][1])[0]
        if '9' not in self.digits and '6' in self.digits:
            self.digits['9'] = self.digits['6'][::-1, ::-1]   # the 9 of this font: the 6 upside down
        gaps = sorted(gaps)
        self.gap = gaps[len(gaps) // 3] if gaps else 1      # between two digits
        self.group = gaps[-1] if gaps else 6                # between the thousands (space)
        # F: the E of 'S-SPECIAL' without its lower bar
        if 'E' in self.letters:
            e = self.letters['E'].copy()
            drawn = e != NONE
            stem = drawn.mean(0) > 0.8
            last = stem.nonzero()[0].max() + 1 if stem.any() else 2
            r = e.shape[0] - 1
            while r >= 0 and drawn[r, last:].any():
                e[r, last:] = NONE
                r -= 1
            self.letters['F'] = e

    def price(self, value):
        """Masks and gaps of a price written like the page's prices."""
        s = '%d' % value
        out = []
        for i, c in enumerate(s):
            out.append(self.digits[c])
            left = len(s) - i - 1
            if left and left % 3 == 0:
                if self.sep is not None:
                    out.append(('sep', self.sep[0]))
                else:
                    out.append(('space', None))
        return out


def _clear(page, x0, y0, box, colour, grow=(0, 0)):
    bx0, by0, bx1, by1 = box
    page.index[y0 + by0 - grow[0]:y0 + by1 + grow[1], x0 + bx0:x0 + bx1] = colour


def _stamp(page, x, y_bottom, patch, colour=None):
    """Copy a glyph patch (or a boolean mask in `colour`) with its bottom row at y_bottom."""
    if patch.dtype == bool:
        patch = np.where(patch, colour, NONE).astype(np.uint8)
    h, w = patch.shape
    sub = page.index[y_bottom - h + 1:y_bottom + 1, x:x + w]
    p = patch[:sub.shape[0], :sub.shape[1]]
    sub[p != NONE] = p[p != NONE]


def write_price(page, x0, y0, value, ink):
    orange, black = colours(page)
    _clear(page, x0, y0, PRICE, orange, grow=(2, 1))
    x = x0 + PRICE[0] + 1
    yb = y0 + PRICE[1] + ink.base
    for item in ink.price(value):
        if isinstance(item, tuple):
            kind, m = item
            if kind == 'space':
                x += ink.group
                continue
            _stamp(page, x, yb + 1, m)                   # the comma hangs below the line
            x += m.shape[1] + ink.gap
            continue
        _stamp(page, x, yb, item)
        x += item.shape[1] + ink.gap


def write_name(page, x0, y0, emblem, ink, text='RF'):
    """The emblem (alpha 0..1, any size: scaled to the line) in black, then `text` in the page's
    letters."""
    from PIL import Image
    orange, black = colours(page)
    _clear(page, x0, y0, NAME, orange, grow=(4, 2))
    bx0, by0, bx1, by1 = NAME
    h = by1 - by0 - 1
    letters = [ink.letters[c] for c in text if c in ink.letters]
    lw = sum(m.shape[1] for m in letters)
    room = bx1 - bx0 - (lw + 6 if letters else 0)
    eh = min(h, int(round(emblem.shape[0] * room / emblem.shape[1])))
    ew = int(round(emblem.shape[1] * eh / emblem.shape[0]))
    em = np.asarray(Image.fromarray((emblem * 255).astype(np.uint8)).resize((ew, eh), Image.LANCZOS)) > 110
    yb = y0 + by1 - 2
    x = x0 + bx0
    _stamp(page, x, yb - (h - eh) // 2, em, black)
    x += ew + 6
    for m in letters:
        _stamp(page, x, yb, m)
        x += m.shape[1]


def write_picture(page, x0, y0, rgb):
    """A picture of the car (rgb, any size: fitted to the tile's picture box), in the page's
    colours (the nearest of its 16)."""
    from PIL import Image
    bx0, by0, bx1, by1 = PICTURE
    w, h = bx1 - bx0, by1 - by0
    im = np.asarray(Image.fromarray(rgb).resize((w, h), Image.LANCZOS), np.float32)
    c = page.clut.astype(np.int64)
    pal = np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1).astype(np.float32)
    idx = ((im[:, :, None] - pal[None, None]) ** 2).sum(3).argmin(2)
    page.index[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1] = idx
