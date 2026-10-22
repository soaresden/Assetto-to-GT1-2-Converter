"""An Assetto Corsa car (its folder: the main .kn5, data.acd, skins/) as the Gran Turismo converters
see it (carscene.build), in Gran Turismo's axes: Assetto Corsa's are metres, y up, z forward and x
to the left (negated here).

- Wheels: the WHEEL_LF / RF / LR / RR nodes give the centres; everything under WHEEL_*, RIM_*,
  TYRE_*, DISC_* and SUSP_* is part of the wheels; radius and width are measured on the geometry
  under each WHEEL_* node and checked against tyres.ini (data.acd) when it can be read.
- Left out: the high-resolution cockpit (COCKPIT_HR: the low one is kept), shadows, light beams
  and flares, damage glass, the driver, hidden meshes.
- Textures: those of the .kn5, replaced by the files of the chosen skin (skins/<name>/) of the
  same name, as the game does.
"""
import io
import os
import re

import numpy as np
from PIL import Image

import acmodel
import carscene
import wheels as WH

DROP_RE = re.compile(r'shadow|beam|flare|damage|driver|_lod_?[bcd]\b|blurred|^lod', re.I)
WHEEL_RE = re.compile(r'(^|/)(WHEEL|RIM|TYRE|TIRE|DISC|SUSP|HUB)_?(LF|RF|LR|RR)', re.I)
KEYS = {'LF': 'FL', 'RF': 'FR', 'LR': 'RL', 'RR': 'RR'}


class Textures:
    def __init__(self, kn5_textures, skin_dir=None):
        self.t = dict(kn5_textures)
        if skin_dir and os.path.isdir(skin_dir):
            for f in os.listdir(skin_dir):
                p = os.path.join(skin_dir, f)
                if os.path.isfile(p) and f.lower().endswith(('.dds', '.png', '.jpg')):
                    with open(p, 'rb') as fh:
                        self.t[f] = fh.read()
        self.lower = {k.lower(): k for k in self.t}
        self.cache = {}

    def __call__(self, name):
        if not name:
            return None
        k = self.lower.get(name.lower())
        if k is None:
            return None
        if k not in self.cache:
            try:
                im = Image.open(io.BytesIO(self.t[k]))
                im.load()
                # RGB first: the alpha of Assetto Corsa's maps is often gloss, not coverage
                # (and Pillow premultiplies when resizing RGBA)
                im = im.convert('RGB')
                if max(im.size) > 512:
                    im.thumbnail((512, 512))
                self.cache[k] = np.asarray(im, np.float32) / 255.0
            except Exception:
                self.cache[k] = None
        return self.cache[k]


def main_kn5(folder):
    """The car's main .kn5 in its folder: the biggest one that is not a LOD or the collider."""
    ks = [f for f in os.listdir(folder) if f.lower().endswith('.kn5')
          and 'collider' not in f.lower() and not re.search(r'_lod_?[b-z]\.kn5$', f.lower())]
    if not ks:
        raise ValueError('%s : pas de .kn5' % folder)
    return os.path.join(folder, max(ks, key=lambda f: os.path.getsize(os.path.join(folder, f))))


def load(path, skin=None, paint=None, tyre_radius=None, log=print):
    """path: the car's folder or its .kn5. (shell_pos, shell_tri, scene, wheels, info)."""
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    kn5_path = main_kn5(folder) if os.path.isdir(path) else path
    textures, materials, meshes, nodes = acmodel.load(kn5_path)
    skins = sorted(os.listdir(os.path.join(folder, 'skins'))) if os.path.isdir(os.path.join(folder, 'skins')) else []
    if skin is None and skins:
        skin = skins[0]
    T = Textures(textures, os.path.join(folder, 'skins', skin) if skin else None)
    log('  modèle : %s, %d maillages, peinture du dossier skins : %s'
        % (os.path.basename(kn5_path), len(meshes), skin or '(aucune)'))
    parts = []
    wheel_tris = []
    dropped = 0
    has_lr = any('COCKPIT_LR' in m['path'] for m in meshes)
    for m in meshes:
        mat = m['mat']
        if not m.get('visible', 1) or DROP_RE.search(m['name']) or DROP_RE.search(mat['name']) \
                or (has_lr and 'COCKPIT_HR' in m['path']):
            dropped += 1
            continue
        tex = T(mat['tex'].get('txDiffuse'))
        # glass: the windscreen shader, or a name saying glass (not the plastic, rubber or chrome
        # around it); the blend flag alone also marks decals and badges
        blend = 'windscreen' in mat['shader'].lower() or bool(
            re.search(r'glass|window|vitr|_tran\b', mat['name'], re.I)
            and not re.search(r'plastic|rubber|frame|chrome|badge|exhaust|(^|_)int_', mat['name'], re.I))
        material = dict(name=mat['name'], rgb=(1.0, 1.0, 1.0) if tex is not None else (0.5, 0.5, 0.5),
                        alpha=1.0, blend=blend, tex=tex, shader=mat['shader'])
        uv = m['uv'].astype(np.float64)
        uv -= np.floor(uv.min(0))
        parts.append(dict(name=m['name'], path=m['path'], material=material,
                          pos=m['pos'] * [-1, 1, 1], nrm=m['nrm'] * [-1, 1, 1], uv=uv,
                          tri=m['tri'][:, ::-1]))
        wheel_tris.append(np.full(len(m['tri']), bool(WHEEL_RE.search(m['path']))))
    log('  %d maillages gardés, %d laissés (cockpit détaillé, ombres, faisceaux...)' % (len(parts), dropped))
    member = np.concatenate(wheel_tris)
    wheels = None
    if all('WHEEL_' + k in nodes for k in KEYS):
        wheels = []
        for k in ('LF', 'RF', 'LR', 'RR'):
            c = nodes['WHEEL_' + k][3, :3] * [-1, 1, 1]
            pts = np.concatenate([p['pos'] for p in parts if re.search(r'(^|/)WHEEL_' + k + r'(/|$)', p['path'])]
                                 or [np.zeros((0, 3))])
            if not len(pts):
                wheels = None
                break
            d = pts - c
            r = np.hypot(d[:, 1], d[:, 2])
            # the tyre's outline: its farthest points from the axle (a few stray points aside)
            radius = float(np.percentile(r, 99.8))
            near = r > 0.6 * radius
            width = float(np.percentile(d[near, 0], 99.5) - np.percentile(d[near, 0], 0.5))
            wheels.append(WH.Wheel(KEYS[k], c, radius, width))
        if wheels:
            fl, fr, rl, rr = wheels
            wheels = [fl, fr, rl, rr]
    if wheels is None:
        log('  pas de nœuds WHEEL_* utilisables : roues cherchées sur la géométrie')
        member = None
    elif tyre_radius:
        for w in wheels:
            if abs(w.radius - tyre_radius) > 0.04 * tyre_radius:
                log('  ! roue %s : rayon mesuré %.3f m, tyres.ini %.3f m' % (w.key, w.radius, tyre_radius))
    sp, st, scene, wheels, info = carscene.build(parts, wheels=wheels, member=member, paint=paint, log=log)
    info.update(kn5=kn5_path, skin=skin, skins=skins)
    return sp, st, scene, wheels, info
