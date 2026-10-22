"""The MX-5 RF in the menu pictures of Gran Turismo (PS1), MENU_IMG.ARC (a GT-ARC of the six
languages, each a GT-ARC of pictures, the same numbering in all six):

- the name picture of each replaced car (its page in the dealer, the garage...): the MX-5 emblem
  and "RF" in the game's letters (titles.py);
- the three Eunos Roadster tiles of the Mazda new car page: name, picture and price (dealer.py);
- the Roadsters' INFO page, with the MX-5 RF's history (infopage.py).
"""
import struct

import numpy as np

import dealer
import gt1data
import gtarc
import infopage
import preview
import slt
import tim
import titles

# name picture of each car code (numbers of the language archives)
TITLES = {'a-ian': 3, 'amian': 19, 'aminn': 20, 'amisn': 21, 'amivn': 22, 'an16n': 23, 'arodn': 24,
          'as16n': 25, 'av16n': 26}
BASE_TITLE = 20               # 'EUNOS ROADSTER', in the plain letters the "RF" is taken from
BASE_TEXT = 'EUNOS ROADSTER'
TILE_OF = {'aminn': 4, 'amisn': 5, 'amivn': 6}
# the cars whose INFO page is the Roadsters' one (CIAN16N.slt), which infopage.py draws again
INFO_CARS = {'aminn', 'amisn', 'amivn', 'an16n', 'as16n', 'av16n'}
THUMB_PAINT = {'aminn': 1, 'amisn': 2, 'amivn': 0}   # Soul Red, Machine Grey, Deep Crystal Blue


def emblem_alpha(path, cut=717):
    """The MX-5 part of the badge picture (white on transparent) as coverage 0..1."""
    from PIL import Image
    e = np.asarray(Image.open(path).convert('RGBA'), np.float32)
    a = (e[..., 3] / 255 * e[..., :3].mean(2) / 255)[:, :cut]
    ys, xs = np.nonzero(a > 0.05)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def spec_prices(carinf):
    s = gt1data.Table(gtarc.Arc(carinf)[13])
    return {bytes(r[:6]).rstrip(b'\0').decode(): struct.unpack_from('<I', r, 0x184)[0] // 100 for r in s.records}


def patch(menu_img, carinf, codes, badge_path, car, tex, price, log=print):
    """MENU_IMG.ARC with the MX-5's names, pictures and price (Cr). carinf: the disc's CARINF.DAT
    (the prices the page's digits are learnt from); car, tex: the MX-5's model and texture."""
    outer = gtarc.Arc(menu_img)
    emb = emblem_alpha(badge_path)
    old_prices = spec_prices(carinf)
    page_prices = ['%d' % old_prices[c] for c in dealer.PAGE_CARS]
    # car, tex: the MX-5's model and texture, or a dict code -> (model, texture) (one roof per car)
    made = car if isinstance(car, dict) else {c: (car, tex) for c in codes}
    thumbs = {c: preview.render(*made[c], THUMB_PAINT[c], (236, 200), yaw=215, pitch=-20, bg=(66, 66, 66))
              for c in codes if c in TILE_OF}
    new_outer = {}
    for lang in range(len(outer)):
        inner = gtarc.Arc(outer[lang])
        new = {}
        base = tim.Tim(inner[BASE_TITLE])
        font = titles.plain_font(base, BASE_TEXT)
        idx = titles.emblem_title(base, emb, font, 'RF')
        picture = base.with_pixels(idx)
        for c in codes:
            if c not in TITLES:
                continue
            k = TITLES[c]
            old = tim.Tim(inner[k])
            if old.size != base.size or old.bpp != 4:
                raise ValueError('name picture %d: %s %d bpp' % (k, old.size, old.bpp))
            # the new picture, at the old one's place in VRAM
            b = bytearray(picture)
            struct.pack_into('<4H', b, 12, *old.clut_at)
            struct.pack_into('<4H', b, base.pix_off - 8, *old.img_at)
            new[k] = bytes(b)
        tiles_done = [c for c in codes if c in TILE_OF]
        if tiles_done:
            page = slt.Page(inner[dealer.MAZDA_NEW])
            page2 = slt.Page(inner[dealer.MAZDA_NEW2])
            ink = dealer.Ink(page, page2, page_prices)
            t = dealer.tiles(page)
            if len(t) != len(dealer.PAGE_CARS):
                raise ValueError('Mazda page: %d tiles' % len(t))
            for c in tiles_done:
                x0, y0 = t[TILE_OF[c]]
                dealer.write_name(page, x0, y0, emb, ink)
                dealer.write_picture(page, x0, y0, thumbs[c])
                dealer.write_price(page, x0, y0, price, ink)
            new[dealer.MAZDA_NEW] = page.write()
        if any(c in INFO_CARS for c in codes):
            new.update(infopage.patch(inner, lang))
        new_outer[lang] = gtarc.replace(inner, new, align=4)
    log('  MENU_IMG.ARC : noms (%s) et vignettes du concessionnaire, 6 langues'
        % ', '.join(c for c in codes if c in TITLES))
    if any(c in INFO_CARS for c in codes):
        log("  MENU_IMG.ARC : page INFO (historique de la MX-5 RF), 6 langues")
    return gtarc.replace(outer, new_outer)
