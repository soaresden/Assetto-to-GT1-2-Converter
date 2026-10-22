"""The part shared by every model format: from the model's pieces (already in Gran Turismo's axes,
x right, y up, z forward, metres) to what the Gran Turismo converters use:

- the four wheels (given by the format, as Assetto Corsa's WHEEL_LF... nodes, or found on the
  geometry by wheels.detect), the car put on the ground (tyres at y = 0);
- the outer shell (every triangle but the wheels'), for the low-poly model;
- a coloured scene (views.Scene) for the texture pictures: paint, glass, lamps, black, other;
- a scene of the front right wheel alone, for the game's wheel picture.

A piece is a dict: name, path, material (name, rgb, alpha, blend, tex HxWx3|4 or None, shader),
pos Nx3, nrm Nx3, uv Nx2 or None, tri Mx3.
"""
import re

import numpy as np

import views as V
import wheels as WH

PAINT_RE = re.compile(r'car_?paint|body_?paint|paint|carrosserie|exterior_?colou?r|ext_?body|^body', re.I)
NOT_PAINT_RE = re.compile(r'black|insides?|int_|interior|chrome|glass', re.I)
LIGHT_RE = re.compile(r'light|lamp|phare|feu|headl|taill|led|(^|_)hl|signal|fog|bulb|(^|_)rt_|rlight', re.I)
INTERIOR_RE = re.compile(r'(^|[_/ ])int[_ ]|interior|cockpit|roof_?light|cluster|dash', re.I)


def _tex_mean(m):
    t = m.get('tex')
    if t is None:
        return np.asarray(m['rgb'], np.float32)
    return t[..., :3].reshape(-1, 3).mean(0) * np.asarray(m['rgb'], np.float32)


def _red(c):
    return c[0] > 0.08 and c[0] > 2.2 * max(c[1], c[2], 0.01)


def build(parts, wheels=None, member=None, paint=None, log=print):
    """(shell_pos, shell_tri, scene, wheels [FL, FR, RL, RR], info). wheels/member: the format's
    own wheels and, per triangle of the concatenated pieces, whether it is part of a wheel
    (None: found on the geometry)."""
    pos = np.concatenate([p['pos'] for p in parts])
    off = np.cumsum([0] + [len(p['pos']) for p in parts])
    tri = np.concatenate([p['tri'] + o for p, o in zip(parts, off)])
    tcount = np.cumsum([0] + [len(p['tri']) for p in parts])
    if wheels is None:
        wheels, member = WH.detect(pos, tri, log=log)
    else:
        for w in wheels:
            log('  roue ' + repr(w))
    # the car sits on its wheels: ground at y = 0
    ground = min(w.centre[1] - w.radius for w in wheels)
    for p in parts:
        p['pos'] = p['pos'] - [0, ground, 0]
    for w in wheels:
        w.centre = w.centre - [0, ground, 0]
    # paint: by name, else the biggest uniform, not black, not blended material
    names = {p['material']['name'] for p in parts}
    paint_mats = set(paint) if paint else {n for n in names if PAINT_RE.search(n) and not NOT_PAINT_RE.search(n)}
    if not paint_mats:
        area = {}
        for p in parts:
            m = p['material']
            if m['blend'] or m['tex'] is not None or max(m['rgb']) < 0.15:
                continue
            a = np.linalg.norm(np.cross(p['pos'][p['tri'][:, 1]] - p['pos'][p['tri'][:, 0]],
                                        p['pos'][p['tri'][:, 2]] - p['pos'][p['tri'][:, 0]]), axis=1).sum()
            area[m['name']] = area.get(m['name'], 0) + a
        if area:
            paint_mats = {max(area, key=area.get)}
    log('  peinture : %s' % ', '.join(sorted(paint_mats)) if paint_mats else '  ! pas de peinture trouvée')
    scene = V.Scene()
    wheel_scene = V.Scene()
    sp, st = [], []
    n = 0
    classes = {}
    for i, p in enumerate(parts):
        wm = member[tcount[i]:tcount[i + 1]]
        m = p['material']
        tex = None if m['tex'] is None or p['uv'] is None else m['tex'][..., :3]
        rgb = np.asarray(m['rgb'], np.float32)
        if wm.any():
            t = p['tri'][wm]
            t = t[wheels[1].inside(p['pos'][t].reshape(-1, 3), margin=0.05).reshape(-1, 3).all(1)]
            if len(t):
                wheel_scene.add(p['pos'], p['nrm'], t, rgb, V.OTHER, tex=tex, uv=p['uv'] if tex is not None else None)
        keep = ~wm
        if not keep.any():
            continue
        pos, nrm, t = p['pos'], p['nrm'], p['tri'][keep]
        uv = p['uv']
        mean = _tex_mean(m)
        cent = pos[t].mean(1)
        if m['name'] not in paint_mats:
            # in the wheel arches (liners, arms, springs: around the tyre, a little wider): black,
            # as on the game's cars, whatever colour the model gives them
            well = np.zeros(len(t), bool)
            for w in wheels:
                d = cent - w.centre
                well |= (np.hypot(d[:, 1], d[:, 2]) < w.radius + 0.09) & (np.abs(d[:, 0]) < w.width / 2 + 0.25)
            if well.any():
                scene.add(pos, nrm, t[well], 0.04, V.BLACK)
                sp.append(pos)
                st.append(t[well] + n)
                n += len(pos)
                t, cent = t[~well], cent[~well]
                if not len(t):
                    classes.setdefault(m['name'], 'passage de roue (noir)')
                    continue
        if m['name'] in paint_mats:
            cls = 'peinture'
            scene.add(pos, nrm, t, 1.0, V.PAINT)
        elif m['blend'] and _red(mean):
            cls = 'feux arrière (verre rouge)'
            scene.add(pos, nrm, t, np.clip(mean * 2.5, 0, 1), V.LAMP_R)
        elif m['blend'] and LIGHT_RE.search(m['name']):
            classes.setdefault(m['name'], 'verre de feu (transparent)')
            continue                       # clear lamp covers: the lamps inside show
        elif m['blend']:
            cls = 'vitre'
            scene.add(pos, nrm, t, V.GLASS_RGB, V.GLASS, glass=True)
        elif LIGHT_RE.search(m['name']) and not INTERIOR_RE.search(m['name'] + ' ' + p.get('path', '')):
            cls = 'feux'
            front = cent[:, 2] > 0
            for sel, c in ((front, V.LAMP_F), (~front, V.LAMP_R)):
                if sel.any():
                    if tex is not None:
                        scene.add(pos, nrm, t[sel], np.maximum(rgb, 0.6), c, tex=tex, uv=uv)
                    else:
                        scene.add(pos, nrm, t[sel], np.maximum(rgb, 0.6), c)
        elif tex is not None:
            cls = 'texture'
            scene.add(pos, nrm, t, rgb, V.OTHER, tex=tex, uv=uv)
        elif float(rgb.mean()) < 0.06:
            cls = 'noir'
            scene.add(pos, nrm, t, np.maximum(rgb, 0.03), V.BLACK)
        else:
            cls = 'couleur'
            scene.add(pos, nrm, t, rgb, V.OTHER)
        classes.setdefault(m['name'], cls)
        sp.append(pos)
        st.append(t + n)
        n += len(pos)
    sp, st = np.concatenate(sp), np.concatenate(st)
    # nothing of the body inside the tyres (axles, arms, liners): out of the shell (the game
    # draws its wheels there; those triangles stay in the pictures, where the tyres hide them)
    inside = np.zeros(len(st), bool)
    for w in wheels:
        v = w.inside(sp, margin=0.0) & (np.abs(sp[:, 0] - w.centre[0]) < w.width / 2 + 0.01)
        inside |= v[st].any(1)
    if inside.any():
        log('  coque : %d triangles dans les pneus retirés' % inside.sum())
    st = st[~inside]
    info = dict(paint=sorted(paint_mats), wheel_scene=wheel_scene, classes=classes)
    return sp, st, scene, wheels, info
