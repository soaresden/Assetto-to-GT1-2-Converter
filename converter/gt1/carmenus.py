"""A converted car in the menu pictures of Gran Turismo (PS1, PAL), MENU_IMG.ARC: a GT-ARC of the six
languages (US English, European English, French, German, Italian, Spanish), each a GT-ARC of
pictures numbered as the list of names in entry 1 of MENU_RAW.ARC ("alts.tim", "mazda-new.slt",
"cialtsn.slt"...: checked, amin.tim = 20, mazda-new.slt = 529, cian16n.slt = 408).

For the car replaced (code `altsn`...):
- its name picture `<code[:4]>.tim` (256x48, white on transparent, greys of its CLUT): the
  emblem, the name in big letters and the version in small ones, as the game's own names;
- its tile of its dealer's new car page (`<dealer>-new.slt`): picture, name with the emblem,
  version and price, in the page's black on orange (dealer.py's layout);
- its INFO page `ci<code>.slt`: title and history text (infopage.write), in each language.

Letters: Liberation Sans Bold (fonts/, SIL Open Font License), the widths of the game's bold
Helvetica.
"""
import os
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import dealer
import gtarc
import infopage
import slt
import tim
import titles

FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'LiberationSans-Bold.ttf')


def picture_names(menu_raw):
    return gtarc.Arc(menu_raw)[1].decode('latin-1').split('\n')


def _text(text, size, spacing=0, stretch=1.0):
    """Coverage (0..1) of a line of text, cropped to its ink."""
    f = ImageFont.truetype(FONT, size)
    w = int(f.getlength(text) + spacing * len(text)) + 8
    im = Image.new('L', (w, size + 10), 0)
    d = ImageDraw.Draw(im)
    x = 2
    for ch in text:
        d.text((x, 2), ch, font=f, fill=255)
        x += f.getlength(ch) + spacing
    a = np.asarray(im, np.float32) / 255
    ys, xs = np.nonzero(a > 0.05)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if stretch != 1.0:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (max(1, int(a.shape[1] * stretch)), a.shape[0]), Image.LANCZOS), np.float32) / 255
    return a


def _fit(a, h=None, w=None):
    if h is not None:
        w = max(1, int(round(a.shape[1] * h / a.shape[0])))
    elif w is not None:
        h = max(1, int(round(a.shape[0] * w / a.shape[1])))
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS),
                      np.float32) / 255


def emblem_from_rgba(rgba):
    """An emblem picture (HxWx4 uint8 or float) as levels 0..1 for the white-on-transparent names:
    its brightness where it is drawn (alpha), cropped."""
    a = np.asarray(rgba, np.float32)
    if a.max() > 1.5:
        a = a / 255
    lev = a[..., :3].mean(2) * (a[..., 3] if a.shape[2] == 4 else 1)
    cov = a[..., 3] if a.shape[2] == 4 else (lev > 0.02)
    ys, xs = np.nonzero(cov > 0.05)
    return lev[ys.min():ys.max() + 1, xs.min():xs.max() + 1], cov[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def name_picture(old, emblem, main, sub=None):
    """A name picture like `old` (tim.Tim, 256x48 4 bits, same CLUT and place in VRAM): the emblem
    (levels, coverage), `main` in big letters, `sub` in small ones under it."""
    W, H = old.size
    lum = titles.luminance(old.clut[0])
    empty = titles.blank(old)
    canvas = np.zeros((H, W), np.float32)
    lev, cov = emblem
    eh = H - 6
    E = _fit(lev, h=eh)
    C = _fit(cov, h=eh)
    m = _text(main, 30, spacing=2, stretch=1.08)
    mh = 24 if sub else 28
    M = _fit(m, h=mh)
    S = _fit(_text(sub, 12, spacing=1), h=9) if sub else None
    gap = 8
    width = E.shape[1] + gap + max(M.shape[1], S.shape[1] if S is not None else 0)
    if width > W - 4:                              # too wide: the text narrower
        k = (W - 4 - E.shape[1] - gap) / max(M.shape[1], S.shape[1] if S is not None else 0)
        M = _fit(M, w=max(1, int(M.shape[1] * k))) if M.shape[1] * k < M.shape[1] else M
        M = np.asarray(Image.fromarray((M * 255).astype(np.uint8)).resize((max(1, int(M.shape[1])), mh)), np.float32) / 255
        if S is not None and S.shape[1] > W - 4 - E.shape[1] - gap:
            S = _fit(S, w=W - 4 - E.shape[1] - gap)
        width = E.shape[1] + gap + max(M.shape[1], S.shape[1] if S is not None else 0)
    x = max(2, (W - width) // 2)
    y = (H - eh) // 2
    # the emblem: its own levels where it is drawn (a dark ring stays visible: at least 0.25)
    canvas[y:y + eh, x:x + E.shape[1]] = np.where(C > 0.5, np.maximum(E, 0.25), E * C)
    tx = x + E.shape[1] + gap
    ty = 4 if sub else (H - mh) // 2
    canvas[ty:ty + mh, tx:tx + M.shape[1]] = np.maximum(canvas[ty:ty + mh, tx:tx + M.shape[1]], M)
    if S is not None:
        sy = ty + mh + 4
        canvas[sy:sy + S.shape[0], tx:tx + S.shape[1]] = np.maximum(canvas[sy:sy + S.shape[0], tx:tx + S.shape[1]], S)
    choices = [i for i in range(16) if i != empty]
    cl = lum[choices]
    idx = np.array(choices, np.uint8)[np.abs(canvas[..., None] - cl[None, None]).argmin(2)]
    idx[canvas < 0.06] = empty
    return old.with_pixels(idx)


# ---------------------------------------------------------------------------------------------
def _page_rgb(page):
    c = page.clut.astype(np.int64)
    return np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1).astype(np.float32)


def _paint(page, x, y, cov, colour, box=None):
    """Coverage drawn in `colour` (RGB) over the page, each pixel the nearest colour of the page's
    CLUT to the mix with what is under it."""
    pal = _page_rgb(page)
    h, w = cov.shape
    sub = page.index[y:y + h, x:x + w]
    cov = cov[:sub.shape[0], :sub.shape[1]]
    under = pal[sub]
    want = under * (1 - cov[..., None]) + np.asarray(colour, np.float32) * cov[..., None]
    pick = ((want[:, :, None] - pal[None, None]) ** 2).sum(3).argmin(2)
    m = cov > 0.08
    sub[m] = pick[m]


def _line(text, size):
    """Coverage of a line of text drawn at its baseline: (coverage, rows above the baseline)."""
    f = ImageFont.truetype(FONT, size)
    w = int(f.getlength(text)) + 4
    asc, desc = f.getmetrics()
    im = Image.new('L', (w, asc + desc + 2), 0)
    ImageDraw.Draw(im).text((1, asc + 1), text, font=f, fill=255, anchor='ls')
    return np.asarray(im, np.float32) / 255, asc + 1


# baselines and sizes of the tile's lines, measured on the Mazda page (dealer.py's boxes): the
# name 15 px high (Liberation Sans Bold 20), the trim and the price 14 px (17)
NAME_BASE, TRIM_BASE, PRICE_BASE = 20, 39, 59
NAME_SIZE, TRIM_SIZE, PRICE_SIZE = 20, 17, 17


def price_text(value, lang):
    """The price as the pages write it: 29,900 in English, Italian and Spanish, 29 900 in French,
    29.900 in German (languages in MENU_IMG.ARC's order)."""
    return '{:,}'.format(value).replace(',', {2: ' ', 3: '.'}.get(lang, ','))


def write_tile(page, x0, y0, emblem_cov, name, trim, price, thumb, lang=0):
    """One tile of a dealer page: picture (thumb, RGB), emblem + name, trim, price."""
    orange, black = dealer.colours(page)
    ink = _page_rgb(page)[black]
    # the old texts (their anti-aliasing goes one or two pixels past the boxes on the left; the
    # price's starts right after the Cr coin)
    for box, left in ((dealer.NAME, 2), (dealer.TRIM, 2), (dealer.PRICE, 2)):
        bx0, by0, bx1, by1 = box
        page.index[y0 + by0 - 3:y0 + by1 + 2, x0 + bx0 - left:x0 + bx1] = orange
    dealer.write_picture(page, x0, y0, thumb)

    def put(text, size, x, base, right):
        cov, up = _line(text, size)
        if cov.shape[1] > right - x:            # too long: narrower
            cov = np.asarray(Image.fromarray((cov * 255).astype(np.uint8)).resize((right - x, cov.shape[0]), Image.LANCZOS), np.float32) / 255
        _paint(page, x, y0 + base - up, cov, ink)

    bx0, by0, bx1, by1 = dealer.NAME
    x = x0 + bx0
    if emblem_cov is not None:
        e = _fit(emblem_cov, h=15)
        _paint(page, x, y0 + NAME_BASE - 14, e, ink)
        x += e.shape[1] + 4
    put(name, NAME_SIZE, x, NAME_BASE, x0 + bx1)
    if trim:
        put(trim, TRIM_SIZE, x0 + dealer.TRIM[0], TRIM_BASE, x0 + dealer.TRIM[2])
    put(price_text(price, lang), PRICE_SIZE, x0 + dealer.PRICE[0] + 1, PRICE_BASE, x0 + dealer.PRICE[2])


# ---------------------------------------------------------------------------------------------
def patch(menu_img, menu_raw, cars, log=print):
    """MENU_IMG.ARC with the cars' names, tiles and INFO pages. cars: list of dict(code, emblem
    (levels, coverage), main, sub, dealer ('mazda-new.slt'), tile (its number on the page), tile_name,
    tile_trim, price, thumb (RGB), info_title, info (six texts or None))."""
    names = picture_names(menu_raw)
    num = {n.lower(): i for i, n in enumerate(names)}
    outer = gtarc.Arc(menu_img)
    new_outer = {}
    for lang in range(len(outer)):
        inner = gtarc.Arc(outer[lang])
        new = {}
        for c in cars:
            k = num[c['code'][:4] + '.tim']
            old = tim.Tim(new.get(k, inner[k]))
            new[k] = name_picture(old, c['emblem'], c['main'], c.get('sub'))
            if c.get('dealer'):
                p = num[c['dealer']]
                page = slt.Page(new.get(p, inner[p]))
                t = dealer.tiles(page)
                x0, y0 = t[c['tile']]
                write_tile(page, x0, y0, c['emblem'][0], c['tile_name'], c.get('tile_trim'), c['price'], c['thumb'], lang)
                new[p] = page.write()
            if c.get('info'):
                p = num['ci%s.slt' % c['code']]
                page = slt.Page(new.get(p, inner[p]))
                infopage.write(page, c['info'][lang], title=c['info_title'])
                new[p] = page.write()
        new_outer[lang] = gtarc.replace(inner, new, align=4)
    log('  MENU_IMG.ARC : noms, vignettes du concessionnaire et pages INFO (%s), 6 langues'
        % ', '.join(c['code'] for c in cars))
    return gtarc.replace(outer, new_outer)
