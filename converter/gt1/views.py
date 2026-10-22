"""Orthographic pictures of the detailed model (side, top, front, rear), for the texture of the
low-poly shell: each face of the shell takes its texels from the view it faces, as the cars of
Gran Turismo (PS1) do (their textures are side, top, front and rear pictures of the car).

Rendering is by point splatting: every triangle is covered by random points (more for bigger
ones), and a z-buffer keeps the nearest point of each pixel. Only numpy: also runs in Pyodide.
Each pixel gets an RGB colour, a class (paint, glass...) and, for the paint, a shade.
Coordinates: Gran Turismo's axes in metres (x right, y up, z forward).
"""
import numpy as np

# classes of pixels
EMPTY, PAINT, BLACK, GLASS, LAMP_F, LAMP_R, OTHER = range(7)
GLASS_RGB = (0.10, 0.12, 0.14)

# view: (image u axis, image v axis (down), depth axis (towards the camera))
DENSITY = 12.0     # sample points per square pixel
CHUNK = 600000     # sample points at a time

VIEWS = {
    'side':  ((2, 1), (1, -1), (0, 1)),     # seen from the right: u = z
    'top':   ((2, 1), (0, 1), (1, 1)),      # seen from above: u = z, v = x (centre line at v=0)
    'front': ((0, 1), (1, -1), (2, 1)),     # seen from the front: u = x (centre at 0)
    'rear':  ((0, 1), (1, -1), (2, -1)),    # seen from behind: u = x (centre at 0)
}


class Scene:
    """Triangles with, per vertex: position, normal, colour, class, shade (paint)."""

    def __init__(self):
        self.parts = []

    def add(self, pos, nrm, tri, rgb, cls, shade=None, glass=False, tex=None, uv=None):
        """rgb: colour per vertex (or one); tex (HxWx3 float 0..1) and uv: multiplied in per point."""
        n = len(pos)
        rgb = np.broadcast_to(np.asarray(rgb, np.float32), (n, 3))
        shade = np.ones(n, np.float32) if shade is None else np.broadcast_to(np.asarray(shade, np.float32), (n,))
        self.parts.append(dict(pos=np.asarray(pos, np.float32), nrm=np.asarray(nrm, np.float32), tri=tri,
                               rgb=rgb, cls=cls, shade=shade, glass=glass, tex=tex,
                               uv=None if uv is None else np.asarray(uv, np.float32)))


def sample(img, uv):
    """Nearest texel of img (HxWxC) at uv (wrapped)."""
    h, w = img.shape[:2]
    x = ((uv[:, 0] % 1.0) * w).astype(np.int64).clip(0, w - 1)
    y = ((uv[:, 1] % 1.0) * h).astype(np.int64).clip(0, h - 1)
    return img[y, x]


def _chunks(p2, tri, density, limit):
    """The triangles in runs whose sample points (as _samples counts them) stay under limit."""
    a, b, c = p2[tri[:, 0]], p2[tri[:, 1]], p2[tri[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    n = np.minimum((area * density).astype(np.intp) + 3, 4000)   # intp: 32 bits in the browser
    cut = np.searchsorted(np.cumsum(n), np.arange(limit, n.sum() + limit, limit))
    start = 0
    for e in np.unique(np.r_[cut, len(tri)]):
        e = max(int(e), start + 1)
        if start >= len(tri):
            break
        yield tri[start:e]
        start = e


def _samples(p2, tri, density, rng):
    """Random barycentric points on each triangle; density = points per square pixel."""
    a, b, c = p2[tri[:, 0]], p2[tri[:, 1]], p2[tri[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    n = np.minimum((area * density).astype(np.intp) + 3, 4000)   # intp: 32 bits in the browser
    t = np.repeat(np.arange(len(tri)), n)
    r1 = np.sqrt(rng.random(len(t), dtype=np.float32))
    r2 = rng.random(len(t), dtype=np.float32)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    # the three corners themselves, so that thin triangles show
    first = np.r_[0, np.cumsum(n)[:-1]]
    for k in range(3):
        w[first + k] = np.eye(3, dtype=np.float32)[k]
    return t, w


def render(scene, view, res, box, light=(0.35, 0.85, 0.4), hull=None, seed=1):
    """Picture of the scene for one view: dict(rgb HxWx3 float 0..1, cls HxW, shade HxW, depth).

    res: pixels per metre. box: ((u0, u1), (v0, v1)) in metres along the view's u and v axes.
    hull: (pos, tri) of the low-poly shell: where the model is well behind it (or absent) inside its
    outline, the pixel is seen through a window (the model has no side windows): tinted glass."""
    (ua, us), (va, vs), (da, ds) = VIEWS[view]
    (u0, u1), (v0, v1) = box
    W, H = int(np.ceil((u1 - u0) * res)), int(np.ceil((v1 - v0) * res))
    rng = np.random.default_rng(seed)
    L = np.asarray(light, np.float32)
    L /= np.linalg.norm(L)

    def splat(parts):
        best = np.full(H * W, -1e9, np.float32)
        rgb = np.zeros((H * W, 3), np.float32)
        cls = np.zeros(H * W, np.int8)
        shade = np.zeros(H * W, np.float32)
        for P in parts:
            if len(P['tri']) == 0:
                continue
            pos = P['pos']
            px = (pos[:, ua] * us - u0) * res
            py = (pos[:, va] * vs - v0) * res
            # dense enough that the nearest surface covers every pixel: with fewer points some
            # pixels showed what is behind it (dark specks over the paint). The triangles go by
            # chunks of at most CHUNK points: the arrays of millions of points were too much for
            # the browser's Python (WebAssembly)
            p2 = np.stack([px, py], 1)
            for tri in _chunks(p2, P['tri'], DENSITY, CHUNK):
                t, w = _samples(p2, tri, DENSITY, rng)
                tv = tri[t]
                ix = (px[tv] * w).sum(1).astype(np.int64)
                iy = (py[tv] * w).sum(1).astype(np.int64)
                ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
                w, tv, ix, iy = w[ok], tv[ok], ix[ok], iy[ok]
                if not len(tv):
                    continue
                d = ((pos[tv, da] * ds) * w).sum(1)
                pix = iy * W + ix
                # keep the nearest point per pixel among these and what is already there
                order = np.lexsort((d, pix))
                pix, d, w, tv = pix[order], d[order], w[order], tv[order]
                last = np.r_[pix[1:] != pix[:-1], True]
                pix, d, w, tv = pix[last], d[last], w[last], tv[last]
                win = d > best[pix]
                pix, d, w, tv = pix[win], d[win], w[win], tv[win]
                best[pix] = d
                if P.get('nrm') is None:
                    continue
                n = (P['nrm'][tv] * w[:, :, None]).sum(1)
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
                lit = 0.55 + 0.45 * np.clip(n @ L, 0, 1)
                col = (P['rgb'][tv] * w[:, :, None]).sum(1)
                if P['tex'] is not None:
                    col = col * sample(P['tex'], (P['uv'][tv] * w[:, :, None]).sum(1))
                rgb[pix] = col * lit[:, None]
                shade[pix] = (P['shade'][tv] * w).sum(1) * lit
                cls[pix] = P['cls']
        return best, rgb, cls, shade

    best, rgb, cls, shade = splat([P for P in scene.parts if not P['glass']])
    _drop_outliers(best.reshape(H, W), rgb.reshape(H, W, 3), cls.reshape(H, W), shade.reshape(H, W))
    gbest, grgb, _, _ = splat([P for P in scene.parts if P['glass']])
    # glass in front of the rest: tinted mix of what is behind it and the glass colour
    front = gbest > best + 0.005
    mix = np.where(cls[front] == EMPTY, 0.0, 0.45)[:, None]
    rgb[front] = mix * rgb[front] + (1 - mix) * grgb[front]
    cls[front] = GLASS
    best[front] = gbest[front]
    if hull is not None:
        hbest = splat([dict(pos=np.asarray(hull[0], np.float32), tri=hull[1], nrm=None)])[0]
        seen_through = (hbest > -1e8) & (cls != GLASS) & ((cls == EMPTY) | (hbest - best > 0.12))
        mix = np.where(cls[seen_through] == EMPTY, 0.0, 0.4)[:, None]
        rgb[seen_through] = mix * rgb[seen_through] + (1 - mix) * np.asarray(GLASS_RGB, np.float32)
        cls[seen_through] = GLASS
    cls, rgb, shade = _fill_holes(cls.reshape(H, W), rgb.reshape(H, W, 3), shade.reshape(H, W))
    return dict(rgb=np.clip(rgb, 0, 1), cls=cls, shade=shade, depth=best.reshape(H, W), size=(W, H),
                res=res, box=box, view=view)


def _drop_outliers(depth, rgb, cls, shade, gap=0.03):
    """Pixels whose nearest point is well behind all four neighbours (a far surface showing
    through a gap between the points of the near one) take the nearest neighbour's values."""
    H, W = depth.shape
    nb = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    stack = np.stack([np.roll(np.roll(depth, dy, 0), dx, 1) for dy, dx in nb])
    lo = stack.min(0)
    bad = (cls != EMPTY) & (lo > depth + gap)
    ys, xs = np.nonzero(bad)
    if not len(ys):
        return
    k = stack[:, ys, xs].argmax(0)
    dy = np.array([d[0] for d in nb])[k]
    dx = np.array([d[1] for d in nb])[k]
    sy, sx = (ys - dy) % H, (xs - dx) % W
    depth[ys, xs] = depth[sy, sx]
    rgb[ys, xs] = rgb[sy, sx]
    cls[ys, xs] = cls[sy, sx]
    shade[ys, xs] = shade[sy, sx]


def _fill_holes(cls, rgb, shade, passes=2):
    """Pixels missed by the points but surrounded by filled ones take their neighbours' values."""
    for _ in range(passes):
        empty = cls == EMPTY
        nb = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb.append(np.roll(np.roll(~empty, dy, 0), dx, 1))
        count = sum(n.astype(np.int8) for n in nb)
        hole = empty & (count >= 3)
        if not hole.any():
            break
        ys, xs = np.nonzero(hole)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sy, sx = (ys - dy) % cls.shape[0], (xs - dx) % cls.shape[1]
            src = cls[sy, sx] != EMPTY
            take = src & (cls[ys, xs] == EMPTY)
            cls[ys[take], xs[take]] = cls[sy[take], sx[take]]
            rgb[ys[take], xs[take]] = rgb[sy[take], sx[take]]
            shade[ys[take], xs[take]] = shade[sy[take], sx[take]]
    return cls, rgb, shade
