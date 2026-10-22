"""The four wheels of a car model found from its geometry (for models without Assetto Corsa's
WHEEL_LF... nodes), and the checks that every converted car must pass.

Model in Gran Turismo's axes (x right, y up, z forward, metres, ground at y = 0 not assumed).

Detection: the triangles are cut into connected pieces (vertices welded by position); a piece is a
wheel candidate when it is round seen from the side (y and z extents within 15 % of each other,
0.45 to 1.1 m across), narrow (x extent under 0.45 m), low (centre under 45 % of the car's height)
and out at a side. The candidates fall into four corners; in each, the biggest one is the tyre:
its centre and size are the wheel's. Every piece inside the wheel's cylinder (rim, disc, caliper,
nuts) belongs to the wheel.
"""
import numpy as np


class Wheel:
    def __init__(self, key, centre, radius, width):
        self.key, self.centre, self.radius, self.width = key, np.asarray(centre, float), float(radius), float(width)

    def __repr__(self):
        c = self.centre
        return '%s: centre (%.3f, %.3f, %.3f) m, radius %.3f m, width %.3f m' % (self.key, c[0], c[1], c[2], self.radius, self.width)

    def inside(self, p, margin=0.02):
        """Points p (Nx3) inside the wheel's cylinder (with a margin)."""
        d = p - self.centre
        r = np.hypot(d[:, 1], d[:, 2])
        return (np.abs(d[:, 0]) <= self.width / 2 + margin + 0.03) & (r <= self.radius + margin)


def pieces(pos, tri, weld=1e-4):
    """Label of the connected piece of each triangle (vertices welded at `weld` metres)."""
    key = np.round(pos / weld).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.ravel()
    t = vid[tri]
    n = vid.max() + 1
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    # union by edges, vectorised in rounds (pointer jumping)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]]])
    while True:
        a, b = parent[e[:, 0]], parent[e[:, 1]]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        moved = lo != hi
        if not moved.any():
            break
        np.minimum.at(parent, hi[moved], lo[moved])
        for _ in range(4):
            parent = parent[parent]
    for _ in range(32):
        p2 = parent[parent]
        if (p2 == parent).all():
            break
        parent = p2
    return parent[t[:, 0]]


def detect(pos, tri, log=print):
    """[FL, FR, RL, RR] Wheel objects found in the model, and a boolean per triangle: part of a
    wheel. Raises ValueError with an explanation when the four wheels are not clear."""
    lab = pieces(pos, tri)
    lo, hi = pos.min(0), pos.max(0)
    H = hi[1] - lo[1]
    zmid = (lo[2] + hi[2]) / 2
    cands = []
    for L in np.unique(lab):
        sel = tri[lab == L].ravel()
        p = pos[sel]
        a, b = p.min(0), p.max(0)
        s = b - a
        c = (a + b) / 2
        dia = max(s[1], s[2])
        if not (0.45 <= dia <= 1.1) or s[0] > 0.45 or s[0] < 0.08:
            continue
        if c[1] - lo[1] > 0.45 * H or abs(c[0]) < 0.35:
            continue
        # a tyre is round seen from the side: every point of its outline at the same distance
        # from the centre (a wheel arch or a door is not, even with a square bounding box)
        r = np.hypot(p[:, 1] - c[1], p[:, 2] - c[2])
        if abs(s[1] - s[2]) > 0.03 * dia or r.max() > 0.52 * dia or np.percentile(r, 90) < 0.42 * dia:
            continue
        # and it stands on the ground (the lowest point of the model, within 6 cm)
        if a[1] - lo[1] > 0.06:
            continue
        cands.append((L, c, s, len(sel)))
    corners = {}
    for L, c, s, n in cands:
        k = ('F' if c[2] > zmid else 'R') + ('L' if c[0] < 0 else 'R')
        corners.setdefault(k, []).append((max(s[1], s[2]), L, c, s))
    missing = [k for k in ('FL', 'FR', 'RL', 'RR') if k not in corners]
    if missing:
        raise ValueError('roues : pas trouvé de pneu en %s (%d pièces rondes trouvées). Indiquer les '
                         'centres des roues à la main dans la fiche (wheels).' % (', '.join(missing), len(cands)))
    wheels = []
    for k in ('FL', 'FR', 'RL', 'RR'):
        dia, L, c, s = max(corners[k], key=lambda x: x[0])
        wheels.append(Wheel(k, c, dia / 2, s[0]))
    member = np.zeros(len(tri), bool)
    cent = pos[tri].mean(1)
    for L in np.unique(lab):
        m = lab == L
        p = pos[tri[m].ravel()]
        for w in wheels:
            if w.inside(p).all():
                member |= m
                break
    for w in wheels:
        log('  roue ' + repr(w))
    return wheels, member


def check(wheels, spec=None, body=None, log=print):
    """Problems (list of text) with four wheels; empty: good. spec: dict with wheelbase / tracks in
    mm and tyre radius in m if known; body: (pos, tri) of the shell, to check the arches."""
    fl, fr, rl, rr = wheels
    out = []

    def bad(t):
        out.append(t)

    for a, b in ((fl, fr), (rl, rr)):
        if abs(a.centre[0] + b.centre[0]) > 0.02:
            bad('roues %s/%s pas symétriques (x %.3f / %.3f m)' % (a.key, b.key, a.centre[0], b.centre[0]))
        if abs(a.centre[2] - b.centre[2]) > 0.02 or abs(a.centre[1] - b.centre[1]) > 0.02:
            bad('roues %s/%s pas sur le même essieu' % (a.key, b.key))
        if abs(a.radius - b.radius) > 0.01:
            bad('roues %s/%s de rayons différents (%.3f / %.3f m)' % (a.key, b.key, a.radius, b.radius))
    wb = (fl.centre[2] + fr.centre[2] - rl.centre[2] - rr.centre[2]) / 2
    if not 1.8 < wb < 3.6:
        bad('empattement de %.2f m : pas une voiture ?' % wb)
    for w in wheels:
        if not 0.25 < w.radius < 0.5:
            bad('roue %s : rayon %.3f m hors de 0,25..0,50' % (w.key, w.radius))
        if not 0.12 < w.width < 0.42:
            bad('roue %s : largeur %.3f m hors de 0,12..0,42' % (w.key, w.width))
    ground = min(w.centre[1] - w.radius for w in wheels)
    for w in wheels:
        if abs(w.centre[1] - w.radius - ground) > 0.02:
            bad('roue %s ne touche pas le sol comme les autres (%.3f m)' % (w.key, w.centre[1] - w.radius - ground))
    if spec:
        if spec.get('wheelbase') and abs(wb * 1000 - spec['wheelbase']) > 0.03 * spec['wheelbase']:
            bad('empattement du modèle %.0f mm, fiche %d mm' % (wb * 1000, spec['wheelbase']))
        if spec.get('tyre_radius'):
            for w in wheels:
                if abs(w.radius - spec['tyre_radius']) > 0.06 * spec['tyre_radius']:
                    bad('roue %s : rayon du modèle %.3f m, pneus de la fiche %.3f m'
                        % (w.key, w.radius, spec['tyre_radius']))
    if body is not None:
        bp, bt = body
        bp = bp[np.unique(bt)]
        for w in wheels:
            inside = w.inside(bp, margin=-0.02) & (np.abs(bp[:, 0] - w.centre[0]) < w.width / 2 - 0.01)
            if inside.any():
                bad('roue %s : %d sommets de la coque dans le pneu' % (w.key, int(inside.sum())))
    for t in out:
        log('  ! ' + t)
    return out
