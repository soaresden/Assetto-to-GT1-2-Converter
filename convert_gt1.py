"""A 3D car model (Assetto Corsa .kn5 or glTF / .glb) in Gran Turismo (PS1, PAL SCES-00984), in
place of one of the game's cars, as a patched disc image (.bin + .cue).

python convert_gt1.py --model car.glb --specs car.json --disc "Gran Turismo.chd" --out "GT1 car.bin"

The specification file (JSON) gives the car's name, dimensions, engine, gears, tyres, paints, price
and the game car it replaces (see examples/). Nothing of the game or of the model is kept here.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'converter', d) for d in ('source', 'gt1', 'common')]

import build_car as B     # noqa: E402
import carinf             # noqa: E402
import disc               # noqa: E402
import gtarc              # noqa: E402
import sysnames           # noqa: E402
import wheelpic           # noqa: E402
import wheels as WH       # noqa: E402

BOOT = 'SCES_009.84'
NIGHT = 688
# CAR.DAT entry of the day texture of some cars (model = +1, night = +688), checked on the disc
TARGETS = {
    'slgan': (532, 'Subaru Legacy Touring Wagon GT-B'),
    'aminn': (70, 'Mazda Eunos Roadster NORMAL'),
    'altsn': (64, 'Mazda Lantis 2000 Type-R'),
}
# the car's tile on its dealer's new car page (MENU_IMG.ARC picture, tile number column by column)
TILES = {'slgan': ('subaru-new.slt', 5), 'aminn': ('mazda-new.slt', 4), 'altsn': ('mazda-new.slt', 3)}
RPM = list(range(1000, 9000, 500))      # the 16 points of the game's torque curves


def torque_curve(points):
    """16 values (kgf.m x 100) at 1000..8500 rpm from [[rpm, Nm], ...] (linear, 0 past the end)."""
    p = np.array(sorted(points), float)
    nm = np.interp(RPM, p[:, 0], p[:, 1], left=p[0, 1], right=0.0)
    return tuple(int(round(v / 9.80665 * 100)) for v in nm)


def gt1_spec(s):
    """The CARINF.DAT values of a specification file."""
    e = s['engine']
    gears = [g for g in s['gears']]
    if len(gears) > 7:                   # the game has 7 forward gears at most: drop the top ones
        gears = gears[:7]
    g = [int(round(x * 1000)) for x in [s.get('reverse', 3.5)] + gears] + [0] * (8 - 1 - len(gears))
    g = g[:8] + [int(round(s['final'] * 1000))]
    t = s['tyres']
    peak_nm = max(n for _, n in e['torque'])
    return {
        'dims': (s['width'], s['height'], s['wheelbase']), 'tracks': tuple(s['tracks']),
        'gears': tuple(g), 'n_gears': len(gears),
        'shift': int(e['shift_rpm'] // 100), 'rev': int(e['rev_limit'] // 100),
        'torque': torque_curve(e['torque']),
        'weight': s['weight'],
        'tyres': (t['rim'], t['rim'], t['width'] // 10, t['width'] // 10, t['profile'] // 5, t['profile'] // 5),
        # the PAL menus show the power x 0.986 (PS to hp): 187 shows as 184
        'display': (s['length'], e['cc'], int(np.ceil(e['power_ps'] / 0.986)), e['power_rpm'],
                    int(round(peak_nm / 9.80665 * 100)), e['torque_rpm']),
        # shown on the SPEC page (None: those of the car replaced)
        'type': carinf.type_bytes(e['aspiration'], e.get('valvetrain', 'DOHC'), e['cylinders'],
                                  *s['suspension']) if 'cylinders' in e and 'suspension' in s else None,
    }


def load_model(path, s, log):
    if path.lower().endswith(('.glb', '.gltf')):
        import glbscene
        sp, st, scene, wheels, info = glbscene.load(path, length=s['length'], front=s.get('front'),
                                                    paint=s.get('paint_materials'), log=log)
    else:
        import kn5scene
        tyre_r = None
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if os.path.isfile(os.path.join(folder, 'data.acd')):
            import acd
            try:
                ini = acd.read(os.path.join(folder, 'data.acd'))['tyres.ini'].decode('latin-1')
                import re
                m = re.search(r'^\s*RADIUS\s*=\s*([0-9.]+)', ini, re.M)
                tyre_r = float(m.group(1)) if m else None
            except Exception as e:
                log('  data.acd illisible (%s) : pas de contrôle avec tyres.ini' % e)
        sp, st, scene, wheels, info = kn5scene.load(path, skin=s.get('skin'), paint=s.get('paint_materials'),
                                                    tyre_radius=tyre_r, log=log)
        L = (sp[np.unique(st)][:, 2].max() - sp[np.unique(st)][:, 2].min()) * 1000
        if abs(L - s['length']) > 0.03 * s['length']:
            log('  ! longueur du modèle %.0f mm, fiche %d mm' % (L, s['length']))
    # what sticks out past the body's width (mirrors) stays out of the shell
    half = s['width'] / 2000.0 + 0.01
    keep = (np.abs(sp[st][:, :, 0]) <= half).all(1)
    log('  coque : %d triangles (%d hors largeur, rétroviseurs : laissés à la texture)' % (keep.sum(), (~keep).sum()))
    st = st[keep]
    tyre_r = (s['tyres']['rim'] * 0.0254 + 2 * s['tyres']['width'] / 1000 * s['tyres']['profile'] / 100) / 2
    problems = WH.check(wheels, spec={'wheelbase': s['wheelbase'], 'tyre_radius': tyre_r}, body=(sp, st), log=log)
    if problems:
        raise ValueError('roues : %d problème(s), conversion arrêtée' % len(problems))
    zc = (wheels[0].centre[2] + wheels[2].centre[2]) / 2
    sp = sp - [0, 0, zc]
    for P in scene.parts:
        P['pos'] = P['pos'] - np.float32([0, 0, zc])
    W = [w.centre - [0, 0, zc] for w in wheels]
    info['wheel_scene'] = _shift(info['wheel_scene'], zc)
    for w in wheels:
        w.centre = w.centre - [0, 0, zc]
    return sp, st, scene, W, wheels, info


def _shift(scene, zc):
    for P in scene.parts:
        P['pos'] = P['pos'] - np.float32([0, 0, zc])
    return scene


def generate(model, s, disc_path, log=print, files=None):
    """({file name: new bytes}, Disc, (car, tex)). files: files already changed (another car
    converted before, the MX-5...): this car is added to them."""
    d = disc_path if isinstance(disc_path, disc.Disc) else disc.Disc(disc_path)
    files = dict(files or {})
    read = lambda name: files[name] if name in files else d.read(name)
    if BOOT not in d.read('SYSTEM.CNF').decode('latin-1'):
        raise ValueError("ce n'est pas Gran Turismo PAL (%s)" % BOOT)
    code = s['target']
    i = TARGETS[code][0]
    log('1/3 le modèle (%s)' % os.path.basename(model))
    sp, st, scene, W, wheels, info = load_model(model, s, log)
    lods = [B.make_shell(sp, st, n, (W[0][2], W[2][2]), log=log) for n in B.LODS]
    hv, hf, _ = lods[0]
    ht = np.array([q for f in hf for q in ([(f[0], f[1], f[2]), (f[0], f[2], f[3])] if len(f) == 4 else [tuple(f)])])
    atlas = B.Atlas(scene, sp, (hv, ht))
    paints = [(p['name'], tuple(p['rgb'])) for p in s['paints']]
    log('2/3 CAR.DAT (%s, entrées %d/%d)' % (TARGETS[code][1], i, i + 1))
    arc = gtarc.Arc(read('CAR.DAT'))
    wp = wheelpic.picture(info['wheel_scene'], wheels[1], arc[i], log=log)
    arches = [(w.centre, w.radius) for w in wheels]
    # as the disc's cars: the faces over the arches and black wells behind the wheels drawn
    # before the wheels (the wheels show in front of them, the PlayStation has no depth buffer)
    tex, ftex, _, _ = B.build_texture(atlas, lods, arc[i], paints=paints, wheel_pic=wp)
    before = B.over_wheels(lods, arches)
    lods = B.wheel_walls(lods, ftex, [(w.centre, w.radius, w.width) for w in wheels])
    log('  passages de roue : %d faces dessinées avant les roues, 4 parois noires par niveau' % len(before))
    menu = (wheels[0].width, wheels[0].radius, wheels[2].width, wheels[2].radius)
    car, _ = B.build_car(arc[i + 1], lods, ftex, W, sp[:, 1].min(), menu_wheels=menu, before_wheels=before)
    new = {i: tex, i + NIGHT: tex, i + 1: car, i + 1 + NIGHT: car}
    files['CAR.DAT'] = gtarc.replace(arc, new)
    log('3/3 CARINF.DAT (nom, caractéristiques, peintures, prix)')
    files['CARINF.DAT'] = carinf.patch(read('CARINF.DAT'), [code], [p for p, _ in paints], s.get('price'),
                                       log=log, spec_values=gt1_spec(s), name=tuple(s['name']))
    # the menus: name picture, dealer tile, INFO page (when the specification file has an emblem)
    if s.get('emblem'):
        import carmenus
        import preview
        from PIL import Image
        em = carmenus.emblem_from_rgba(np.asarray(Image.open(s['emblem']).convert('RGBA')))
        info = s.get('info')
        if isinstance(info, str):
            info = [info] * 6
        dealer, tile = TILES.get(code, (None, None))
        car_menu = dict(code=code, emblem=em, main=s.get('title', s['name'][0]), sub=s.get('title_sub', s['name'][1]),
                        dealer=dealer, tile=tile, tile_name=s.get('tile_name', s['name'][0]),
                        tile_trim=s.get('tile_trim', s['name'][1]), price=s.get('price', 0),
                        thumb=preview.render(car, tex, 0, (236, 200), yaw=215, pitch=-20, bg=(66, 66, 66)),
                        info_title=s.get('info_title', ' '.join(s['name'])), info=info)
        files['MENU_IMG.ARC'] = carmenus.patch(read('MENU_IMG.ARC'), d.read('MENU_RAW.ARC'), [car_menu], log=log)
    # the short name of the race screens (SYSTEM.DAT), at the old name's length
    short = s.get('short_name', ' '.join(s['name']))
    files['SYSTEM.DAT'] = sysnames.rename(read('SYSTEM.DAT'), i, short)
    log('  SYSTEM.DAT : nom court %r' % sysnames.name_of(files['SYSTEM.DAT'], i))
    for name, data in files.items():
        if len(data) > d.room(name):
            raise ValueError('%s : %d octets de trop' % (name, len(data) - d.room(name)))
    return files, d, (car, tex)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--specs', required=True)
    ap.add_argument('--disc', required=True, help='Gran Turismo PAL (SCES-00984): .chd, .cue or .bin')
    ap.add_argument('--out', required=True, help='.bin to write (a .cue is written next to it)')
    a = ap.parse_args()
    with open(a.specs, encoding='utf-8') as f:
        s = json.load(f)
    t0 = time.time()
    files, d, (car, tex) = generate(a.model, s, a.disc)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    d.write_bin(a.out, files, lambda k, n: print('\r  %3d %%' % (100 * k // n), end='', flush=True))
    print('\r%s (%.0f s)' % (a.out, time.time() - t0))
    with open(os.path.splitext(a.out)[0] + '.car', 'wb') as f:
        f.write(car)
    with open(os.path.splitext(a.out)[0] + '.tex', 'wb') as f:
        f.write(tex)


if __name__ == '__main__':
    main()
