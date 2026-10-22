"""A low-poly shell of the car, the way the cars of Gran Turismo (PS1) are built: rings of points
around the body at stations along its length, joined by quads, closed at both ends.

Each ring is found by cutting the detailed model with the plane of the station and casting rays
from a point inside the cut: the farthest crossing on each ray is a point of the ring. The rings
are symmetric (the left half mirrors the right one).
Coordinates: Gran Turismo's axes in metres (x right, y up, z forward).
"""
import numpy as np


def plane_cut(pos, tri, z):
    """Segments (Nx2x2, in x y) where the triangles cross the plane at z."""
    p = pos[tri]                        # T x 3 x 3
    d = p[:, :, 2] - z
    s = np.sign(d)
    cross = (s.min(1) < 0) & (s.max(1) > 0)
    p, d = p[cross], d[cross]
    segs = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[:, a], d[:, b]
        hit = (da < 0) != (db < 0)
        t = np.where(hit, da / np.where(hit, da - db, 1), 0)
        q = p[:, a, :2] + t[:, None] * (p[:, b, :2] - p[:, a, :2])
        segs.append((hit, q))
    # each crossing triangle has exactly two crossed edges
    hits = np.stack([h for h, _ in segs], 1)
    qs = np.stack([q for _, q in segs], 1)
    order = np.argsort(~hits, axis=1, kind='stable')[:, :2]
    return np.take_along_axis(qs, order[:, :, None], 1)


def ray_far(segs, c, dirs):
    """For each ray (c + t * dir), the largest t where it crosses a segment (nan if none)."""
    P, Q = segs[:, 0], segs[:, 1]
    e = Q - P                                   # S x 2
    out = np.full(len(dirs), np.nan)
    for i, d in enumerate(dirs):
        # c + t d = P + s e  ->  t d - s e = P - c
        den = d[0] * (-e[:, 1]) - d[1] * (-e[:, 0])
        w = P - c
        ok = np.abs(den) > 1e-12
        t = (w[:, 0] * (-e[:, 1]) - w[:, 1] * (-e[:, 0])) / np.where(ok, den, 1)
        s = (d[0] * w[:, 1] - d[1] * w[:, 0]) / np.where(ok, den, 1)
        good = ok & (s >= -1e-6) & (s <= 1 + 1e-6) & (t > 0)
        if good.any():
            out[i] = t[good].max()
    return out


def ring_targets(n, half_width, y0, y1):
    """n points from the top middle to the bottom middle of the right half of the box (half_width,
    y0..y1), evenly spaced along its edges: the rays aim at them."""
    h = y1 - y0
    s = np.linspace(0, 2 * half_width + h, n)
    x = np.where(s < half_width, s, np.where(s < half_width + h, half_width, 2 * half_width + h - s))
    y = np.where(s < half_width, y1, np.where(s < half_width + h, y1 - (s - half_width), y0))
    return np.stack([x, y], 1)


def build(pos, tri, stations, per_side=7, centre_y=None, smooth=False):
    """Shell from the detailed mesh: (verts Vx3, quads Qx4 going round, ring of each vertex).

    stations: z of the rings, rear to front. A ring has 2 * per_side + 2 points: top, right side
    from top to bottom, bottom, left side from bottom to top."""
    rings = []
    for z in stations:
        segs = plane_cut(pos, tri, z)
        if len(segs) == 0:
            raise ValueError('nothing at z=%.3f' % z)
        pts = segs.reshape(-1, 2)
        y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        yc = (y0 + y1) / 2 if centre_y is None else centre_y
        c = np.array([0.0, yc])
        tg = ring_targets(per_side + 2, np.abs(pts[:, 0]).max(), y0, y1) - c
        right = tg / np.linalg.norm(tg, axis=1, keepdims=True)   # top .. bottom, right side
        r = ray_far(segs, c, right)
        if np.isnan(r).any():  # no crossing: keep the neighbours' radius
            good = ~np.isnan(r)
            r = np.interp(np.arange(len(r)), np.nonzero(good)[0], r[good])
        half = c + right * r[:, None]
        half[0, 0] = half[-1, 0] = 0.0
        # full ring: top, right side down, bottom, left side up (mirror of the right side)
        left = half[-2:0:-1] * [-1, 1]
        ring = np.concatenate([half, left])
        rings.append(np.c_[ring, np.full(len(ring), z)])
    rings = np.array(rings)                               # R x K x 3
    R, K = rings.shape[:2]
    if smooth and R > 2:
        # a point that sticks out of both its neighbours along the car (a headrest, an aerial
        # base...) goes back between them: median of three, end rings kept
        mid = np.median(np.stack([rings[:-2], rings[1:-1], rings[2:]]), 0)
        rings[1:-1, :, :2] = mid[:, :, :2]
    verts = rings.reshape(-1, 3)
    quads = []
    for i in range(R - 1):
        for k in range(K):
            a, b = i * K + k, i * K + (k + 1) % K
            quads.append((a, b, b + K, a + K))
    # end caps: quads across each end ring, pairing its right and left points
    # (the first and the last are triangles: a corner repeats)
    for i, flip in ((0, True), (R - 1, False)):
        base = i * K
        for k in range(per_side + 1):
            a, b = base + k, base + k + 1                        # right side, going down
            c_, d = base + (K - k - 1) % K, base + (K - k) % K   # their mirrors on the left
            q = (a, b, c_, d)
            quads.append(q[::-1] if flip else q)
    ring_of = np.repeat(np.arange(R), K)
    return verts, np.array(quads), ring_of
