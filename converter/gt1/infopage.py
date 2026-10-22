"""The INFO page of the Roadsters (MENU_IMG.ARC, picture 408, CIAN16N.slt): the history text is
drawn into the page itself, one picture per language. All the Roadsters (new and used:
ciaminn.htm, ciamisn.htm, ciamivn.htm, cian16n.htm, cias16n.htm, ciav16n.htm) show this one page.

The page is drawn again with the MX-5 RF's title and text, in the letters of the game (a bold
Helvetica: Liberation Sans Bold, which has the widths of Arial Bold, at the same sizes, 25 px for
the title and 14 px for the text, measured on the originals) and the page's own greys. The layout
is measured on each page: left edge, the gap between the two columns, title and first line, 16 px
between lines.
"""
import collections
import os
import re

import numpy as np

PAGE = 408
# Liberation Sans Bold (SIL Open Font License, fonts/LICENSE-LiberationSans.txt): free to ship
# with the tools, so the web page draws the same page (a browser has no Windows fonts)
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'LiberationSans-Bold.ttf')
FALLBACK = r'C:\Windows\Fonts\arialbd.ttf'
TITLE_SIZE, TEXT_SIZE, PITCH = 25, 14, 16
TOP = 190            # everything below is the title and the text (the icons are above)

TITLE = 'Mazda MX-5 RF'
# the languages of the menus, in their order in MENU_IMG.ARC
TEXTS = [
    # English (USA): hp and lb ft, as the original
    "The Mazda MX-5 RF was unveiled in 2016 as the retractable hardtop version of the fourth-"
    "generation Mazda roadster. RF stands for Retractable Fastback: the three-piece roof folds away "
    "behind the seats in about 13 seconds, while the rear buttresses stay in place. In 2018 the ND2 "
    "update brought a 2.0-liter SKYACTIV-G engine producing 181 hp / 7000 rpm and 151 lb ft / 4000 "
    "rpm, and revving to 7500 rpm. Rear-wheel drive, a six-speed manual gearbox, about 2,450 lb and a "
    "50/50 weight distribution: the roadster recipe has not changed. A car you enjoy from the very "
    "first corner, and never tire of on the track.",
    # English (Europe): PS and kgm
    "The Mazda MX-5 RF was unveiled in 2016 as the retractable hardtop version of the fourth-"
    "generation Mazda roadster. RF stands for Retractable Fastback: the three-piece roof stows behind "
    "the seats in around 13 seconds, while the rear buttresses stay in place. In 2018 the ND2 update "
    "brought a 2.0-litre SKYACTIV-G engine producing 184 PS / 7000 rpm and 20.9 kgm / 4000 rpm, and "
    "revving to 7500 rpm. Rear-wheel drive, a six-speed manual gearbox, just over 1100 kg and a 50/50 "
    "weight distribution: the roadster recipe has not changed. A car you enjoy from the very first "
    "corner, and never tire of on the track.",
    # French
    "La MX-5 RF a \u00e9t\u00e9 pr\u00e9sent\u00e9e en 2016 : c'est la version \u00e0 toit rigide "
    "de la quatri\u00e8me g\u00e9n\u00e9ration du roadster Mazda. RF signifie \u00ab Retractable "
    "Fastback \u00bb : le toit en trois parties se range derri\u00e8re les si\u00e8ges en 13 "
    "secondes environ, tandis que les arcs-boutants arri\u00e8re restent en place. En 2018, la "
    "version ND2 re\u00e7oit un 2.0 SKYACTIV-G de 184 ch / 7 000 tr/mn et 20,9 mkg / 4 000 tr/mn, "
    "qui monte jusqu'\u00e0 7 500 tr/mn. Propulsion, bo\u00eete manuelle \u00e0 6 rapports, \u00e0 "
    "peine plus de 1 100 kg et une r\u00e9partition des masses de 50/50 : la recette du roadster "
    "n'a pas chang\u00e9. Une voiture que l'on appr\u00e9cie d\u00e8s les premiers m\u00e8tres, et "
    "dont on ne se lasse pas sur circuit.",
    # German
    "Der Mazda MX-5 RF wurde 2016 vorgestellt: die Version mit versenkbarem Hardtop der vierten "
    "Roadster-Generation von Mazda. RF steht f\u00fcr \u201eRetractable Fastback\u201c: Das "
    "dreiteilige Dach verschwindet in rund 13 Sekunden hinter den Sitzen, w\u00e4hrend die hinteren "
    "Dachpfosten stehen bleiben. 2018 erhielt der ND2 einen 2,0-Liter-SKYACTIV-G-Motor mit 184 "
    "PS/7000 UpM und 20,9 kgm/4000 UpM, der bis 7500 UpM dreht. Hinterradantrieb, "
    "Sechsgang-Schaltgetriebe, knapp \u00fcber 1100 kg und eine Gewichtsverteilung von 50:50: Am "
    "Rezept des Roadsters hat sich nichts ge\u00e4ndert. Ein Auto, das vom ersten Meter an Freude "
    "macht und auf der Rennstrecke nie langweilig wird.",
    # Italian
    "La Mazda MX-5 RF \u00e8 stata presentata nel 2016: \u00e8 la versione con tetto rigido "
    "retrattile della quarta generazione della roadster Mazda. RF sta per \u00abRetractable "
    "Fastback\u00bb: il tetto in tre parti si ripone dietro i sedili in circa 13 secondi, mentre i "
    "montanti posteriori restano al loro posto. Nel 2018 la ND2 riceve un 2.0 SKYACTIV-G da 184 CV "
    "a 7000 giri/min e 20,9 kgm a 4000 giri/min, che sale fino a 7500 giri/min. Trazione "
    "posteriore, cambio manuale a 6 marce, poco pi\u00f9 di 1100 kg e una distribuzione dei pesi "
    "50/50: la ricetta della roadster non \u00e8 cambiata. Un'auto che si apprezza fin dai primi "
    "metri e che non stanca mai in pista.",
    # Spanish
    "El Mazda MX-5 RF se present\u00f3 en 2016: es la versi\u00f3n con techo r\u00edgido "
    "retr\u00e1ctil de la cuarta generaci\u00f3n del roadster de Mazda. RF significa "
    "\u00abRetractable Fastback\u00bb: el techo de tres piezas se guarda detr\u00e1s de los asientos "
    "en unos 13 segundos, mientras los montantes traseros permanecen en su sitio. En 2018, el ND2 "
    "recibe un motor 2.0 SKYACTIV-G de 184 CV a 7.000 rpm y 20,9 kg-m a 4.000 rpm, que sube hasta "
    "7.500 rpm. Propulsi\u00f3n trasera, cambio manual de 6 velocidades, poco m\u00e1s de 1.100 kg y "
    "un reparto de pesos 50/50: la receta del roadster no ha cambiado. Un coche que se disfruta "
    "desde el primer metro y que nunca cansa en el circuito.",
]


def _layout(page):
    """(background index, text colour index, greys {index: level}, left x, column width, right
    column x, title top, first line top, last line top) measured on the original page."""
    idx = page.index
    c = page.clut.astype(np.int64)
    pal = np.stack([(c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3], -1)
    bg = collections.Counter(idx[TOP:].ravel().tolist()).most_common(1)[0][0]
    ink = idx[TOP:] != bg
    rows = np.nonzero(ink.any(1))[0] + TOP
    runs, start = [], rows[0]
    for a, b in zip(rows, rows[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, rows[-1]))
    title_top, first, last = runs[0][0], runs[1][0], runs[-1][0]
    body = ink[first - TOP:]
    cols = body.any(0)
    xs = np.nonzero(cols)[0]
    left = xs[0]
    # the gap between the columns: the widest empty run inside the text
    best, x = (0, 0), left
    while x < xs[-1]:
        if not cols[x]:
            s = x
            while not cols[x]:
                x += 1
            if x - s > best[1] - best[0]:
                best = (s, x)
        x += 1
    greys = {k: int(pal[k][0]) for k in range(len(pal))
             if pal[k][0] == pal[k][1] == pal[k][2] and (k == bg or c[k] & 0x7FFF)}
    greys[bg] = 0
    vals = idx[first:][body]
    text = collections.Counter(v for v in vals.tolist() if v in greys).most_common(1)[0][0]
    return bg, text, greys, left, best[0] - left, best[1], title_top, first, last


def _keep_together(text):
    """No line break between a number and its unit or its next group of digits, nor before the
    French : ; ! ? and inside the French quotes."""
    nbsp = '\u00a0'
    text = re.sub(r'(\d) (?=[A-Za-z%]|\d{3}\b)', r'\1' + nbsp, text)
    text = re.sub(r' ([:;!?\u00bb])', nbsp + r'\1', text)
    return text.replace('\u00ab ', '\u00ab' + nbsp)


def _wrap(text, font, width):
    words, lines, cur = _keep_together(text).split(' '), [], ''
    for w in words:
        t = w if not cur else cur + ' ' + w
        if font.getlength(t) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _draw(page, x, y, text, font, ink, greys):
    """Text with its top-left at (x, y): coverage to the page's greys, up to the text's level."""
    from PIL import Image, ImageDraw
    w = int(font.getlength(text)) + 4
    h = font.size + 8
    im = Image.new('L', (w, h), 0)
    ImageDraw.Draw(im).text((0, 0), text, font=font, fill=255)
    cov = np.asarray(im, np.float32) / 255
    top = font.getbbox('E')[1]          # rows above the capitals
    keys = np.array(sorted(greys, key=greys.get))
    levels = np.array([greys[k] for k in keys], np.float32)
    want = cov * greys[ink]
    pick = keys[np.abs(want[..., None] - levels).argmin(-1)]
    sub = page.index[y - top:y - top + h, x:x + w]
    m = cov[:sub.shape[0], :sub.shape[1]] > 0.02
    sub[m] = pick[:sub.shape[0], :sub.shape[1]][m]


def write(page, text, title=TITLE):
    """Page with the title and the text in place of the originals (ValueError if it does not fit)."""
    from PIL import ImageFont
    bg, ink, greys, left, width, right, title_top, first, last = _layout(page)
    font = FONT if os.path.exists(FONT) else FALLBACK
    tf = ImageFont.truetype(font, TITLE_SIZE)
    bf = ImageFont.truetype(font, TEXT_SIZE)
    lines = _wrap(text, bf, width)
    per_col = (last - first) // PITCH + 1
    if len(lines) > 2 * per_col:
        raise ValueError('%d lines, room for %d' % (len(lines), 2 * per_col))
    page.index[TOP:, left:] = bg
    _draw(page, left, title_top, title, tf, ink, greys)
    for k, line in enumerate(lines):
        col, row = divmod(k, per_col)
        _draw(page, right if col else left, first + row * PITCH, line, bf, ink, greys)
    return page


def patch(menu_img_inner, lang, log=None):
    """{PAGE: new page bytes} for one language archive of MENU_IMG.ARC."""
    import slt
    page = slt.Page(menu_img_inner[PAGE])
    write(page, TEXTS[lang])
    return {PAGE: page.write()}
