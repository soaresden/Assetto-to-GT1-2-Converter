"""A glTF / GLB car model as the Gran Turismo converters see it: the outer shell (for the low-poly
model), a coloured scene (for the texture pictures) and the four wheels, in Gran Turismo's axes
(x right, y up, z forward, metres), scaled to the car's real length.

The model's own axes: up is the shortest extent, the length the longest; the front is the end
with the light-coloured lamps / glass, the rear the one with the red ones (front='+z' / '-z'...
forces it). Materials are sorted by their name and colour: paint (name with "paint" / "body",
else the biggest uniform colour), glass (blended), red lamp glass, front lamp parts, black, other.
"""
import numpy as np

import carscene


def _axes(pos, front=None):
    """Rotation (3x3) to Gran Turismo's axes: up = y, length = z, front = +z, x right."""
    s = pos.max(0) - pos.min(0)
    up = int(np.argmin(s))
    length = int(np.argmax(s))
    side = 3 - up - length
    R = np.zeros((3, 3))
    R[1, up] = 1.0
    R[2, length] = 1.0
    R[0, side] = 1.0
    return R


def load(path, length=None, front=None, paint=None, log=print):
    """(shell_pos, shell_tri, scene, wheels [FL, FR, RL, RR], info)."""
    import gltf
    parts, extras = gltf.load(path)
    allp = np.concatenate([p['pos'] for p in parts])
    R = _axes(allp)
    # glTF is right-handed (x = y cross z points left when z is the front): x is negated below
    for p in parts:
        p['pos'] = p['pos'] @ R.T
        p['nrm'] = p['nrm'] @ R.T
    allp = np.concatenate([p['pos'] for p in parts])
    # front: the end with the red lamps is the rear
    zmid = (allp[:, 2].min() + allp[:, 2].max()) / 2
    red_z = []
    for p in parts:
        m = p['material']
        r, g, b = m['rgb']
        if r > 0.08 and r > 3 * max(g, b, 0.01):
            red_z.append(p['pos'][:, 2].mean() - zmid)
    flip = False
    if front in ('-z', 'rear'):
        flip = True
    elif front is None and red_z:
        flip = np.mean(red_z) > 0
    for p in parts:
        if flip:
            p['pos'] = p['pos'] * [-1, 1, -1]
            p['nrm'] = p['nrm'] * [-1, 1, -1]
        p['pos'] = p['pos'] * [-1, 1, 1]
        p['nrm'] = p['nrm'] * [-1, 1, 1]
        p['tri'] = p['tri'][:, ::-1]
    allp = np.concatenate([p['pos'] for p in parts])
    lo, hi = allp.min(0), allp.max(0)
    centre = (lo + hi) / 2
    k = 1.0
    if length:
        k = length / 1000.0 / (hi[2] - lo[2])
    for p in parts:
        p['pos'] = (p['pos'] - [centre[0], lo[1], centre[2]]) * k
    log('  modèle : %d pièces, %d triangles, échelle x%.4g, avant %s'
        % (len(parts), sum(len(p['tri']) for p in parts), k, 'retourné' if flip else 'tel quel'))
    sp, st, scene, wheels, info = carscene.build(parts, paint=paint, log=log)
    info.update(extras=extras, scale=k, flipped=flip)
    return sp, st, scene, wheels, info
