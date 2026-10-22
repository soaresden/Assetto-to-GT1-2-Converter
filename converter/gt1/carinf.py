"""The MX-5 RF in CARINF.DAT: name, specifications and paint names of the cars it replaces.

SPEC record (0x1A8 bytes; layout from pez2k's GT1DataSplitter, values checked on the disc):
  0x00 code (6 bytes)        0x0C u16 width, height, wheelbase (mm)   0x14 u16 front, rear track
  0x18 9 x u16 gear ratios x1000 (reverse, 1st..7th, final)           0x2A u8 gear count
  0x30 u8 power multiplier (100)   0x31 u8 shift limit, 0x32 rev limit (x100 rpm)
  0x34 16 x u8 rpm of the torque curve points (x100)                  0x56 u8 point count
  0x5A u16 weight (kg)       0x66 u8 wheel diameter (inches) front, rear, tyre width / 10 front,
  rear, profile / 5 front, rear
  0x13E 16 x u16 torque (kgf.m x 100)                0x184 u32 price (yen; PAL shows / 100)
  0x188 name (u16 string, u16 table), 0x18C second line of the name (u16 string, u16 table)
  0x194 u16 length (mm), displacement (cc), power (PS), power rpm, torque (kgf.m x 100), torque rpm
  0x1A3 u8 aspiration (0 normal, 1 turbo, 2 supercharger), valvetrain (0 SOHC, 1 DOHC, 2 OHV,
  3 rotary), cylinders (0 L4, 1 L6, 2 V6, 3 V8, 4 V10, 5 rotary, 6 boxer 4, 7 boxer 6), front and
  rear suspension (0 strut, 1 trailing twist beam, 2 double wishbone, 3 multilink, 4 parallel-link
  strut, 5 multilink beam, 6 MacPherson strut, 7 semi-trailing arm, 8 torque arm): found by
  comparing the cars of the disc, the names are lists of MESSAGES.DAT
COLOR record (0x54 bytes): u16 car (SPEC index + 1), 16 x u8 paint letters (those of the car's
texture), u16, 16 x (u16 string, u16 table) paint names.
"""
import struct

import gt1data
import gtarc

SPEC_ENTRY, COLOR_ENTRY = 13, 38

NAME = ('MX-5 RF', 'ND2 2.0')
MX5_SPEC = {
    'dims': (1735, 1235, 2310), 'tracks': (1495, 1505),
    # 6-speed: reverse, 1st..6th, 7th, final drive
    'gears': (3603, 3709, 2190, 1536, 1177, 1000, 832, 0, 3454), 'n_gears': 6,
    'shift': 76, 'rev': 82,
    # Skyactiv-G 2.0 (ND2): 205 Nm at 4000 rpm, 184 PS at 7000 rpm (kgf.m x 100, 1000..8500 rpm)
    'torque': (1450, 1650, 1800, 1920, 2000, 2060, 2090, 2080, 2060, 2030, 1990, 1940, 1880, 1760, 1550, 1300),
    'weight': 1125, 'tyres': (17, 17, 20, 20, 9, 9),
    'display': (3915, 1997, 187, 7000, 2090, 4000),   # 187 PS shows as 184 in the PAL menus
}


def patch(carinf, codes, paints, price=None, log=print, spec_values=None, name=None):
    """CARINF.DAT with the cars of `codes` renamed and given the specifications spec_values (a dict
    like MX5_SPEC; default: the MX-5's), name (two lines; default: the MX-5's), and their
    paints renamed (paints: names, in the order of the texture's palettes); price in Cr (the menus
    show yen / 100), None: unchanged."""
    arc = gtarc.Arc(carinf)
    spec = gt1data.Table(arc[SPEC_ENTRY])
    color = gt1data.Table(arc[COLOR_ENTRY])
    index = {bytes(r[:6]).rstrip(b'\0').decode(): i for i, r in enumerate(spec.records)}
    name = name or NAME
    n1 = spec.add_string(0, name[0])
    n2 = spec.add_string(1, name[1])
    names = [color.add_string(0, p) for p in paints]
    s = spec_values or MX5_SPEC
    done = []
    for code in codes:
        if code not in index:
            continue  # racing versions have no record of their own
        i = index[code]
        r = spec.records[i]
        struct.pack_into('<3H', r, 0x0C, *s['dims'])
        struct.pack_into('<2H', r, 0x14, *s['tracks'])
        struct.pack_into('<9H', r, 0x18, *s['gears'])
        r[0x2A] = s['n_gears']
        r[0x31], r[0x32] = s['shift'], s['rev']
        r[0x34:0x44] = bytes(range(10, 90, 5))
        r[0x56] = 16
        struct.pack_into('<H', r, 0x5A, s['weight'])
        r[0x66:0x6C] = bytes(s['tyres'])
        struct.pack_into('<16H', r, 0x13E, *s['torque'])
        struct.pack_into('<4H', r, 0x188, n1, 0, n2, 1)
        struct.pack_into('<6H', r, 0x194, *s['display'])
        if s.get('type'):
            r[0x1A3:0x1A8] = bytes(s['type'])
        if price is not None:
            struct.pack_into('<I', r, 0x184, int(price) * 100)
        for c in color.records:
            if struct.unpack_from('<H', c, 0)[0] == i + 1:
                k = sum(1 for b in c[2:18] if b)
                for j in range(k):
                    struct.pack_into('<2H', c, 20 + 4 * j, names[j % len(names)], 0)
        done.append(code)
    log('  CARINF.DAT : %s -> %s %s' % (', '.join(done), *name))
    new = {SPEC_ENTRY: spec.write(), COLOR_ENTRY: color.write()}
    # the whole file is one compressed archive
    files = [new.get(k, arc[k]) for k in range(len(arc))]
    return gtarc.compress(gtarc.build(files, [False] * len(files), align=4))


ASPIRATION = {'normal': 0, 'na': 0, 'turbo': 1, 'supercharger': 2, 'compresseur': 2}
VALVETRAIN = {'sohc': 0, 'dohc': 1, 'ohv': 2, 'rotary': 3}
CYLINDERS = {'l4': 0, 'l6': 1, 'v6': 2, 'v8': 3, 'v10': 4, 'rotary': 5, 'boxer4': 6, 'b4': 6, 'boxer6': 7, 'b6': 7}
SUSPENSION = {'strut': 0, 'twist beam': 1, 'trailing twist beam': 1, 'double wishbone': 2, 'multilink': 3,
              'parallel-link strut': 4, 'multilink beam': 5, 'macpherson': 6, 'mcpherson': 6,
              'semi-trailing arm': 7, 'torque arm': 8}


def type_bytes(aspiration, valvetrain, cylinders, front, rear):
    """The five bytes at 0x1A3 from names (see the table above)."""
    return (ASPIRATION[aspiration.lower()], VALVETRAIN[valvetrain.lower()], CYLINDERS[cylinders.lower()],
            SUSPENSION[front.lower()], SUSPENSION[rear.lower()])


def set_types(carinf, codes, types):
    """CARINF.DAT with the engine and suspension types of `codes` changed (five bytes, type_bytes)."""
    arc = gtarc.Arc(carinf)
    spec = gt1data.Table(arc[SPEC_ENTRY])
    for r in spec.records:
        if bytes(r[:6]).rstrip(b'\0').decode() in codes:
            r[0x1A3:0x1A8] = bytes(types)
    files = [spec.write() if k == SPEC_ENTRY else arc[k] for k in range(len(arc))]
    return gtarc.compress(gtarc.build(files, [False] * len(files), align=4))
