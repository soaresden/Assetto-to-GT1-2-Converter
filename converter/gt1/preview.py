"""Pictures of a Gran Turismo (PS1) car from its .car and .tex, drawn like the game draws it:
4-bit texels through the face's CLUT, colour 0 transparent, back faces not drawn (the game's faces
have cross(p1 - p0, p2 - p0) pointing inwards). Used for the dealer thumbnails, and to check a
model before it goes on a disc.
"""
import numpy as np

import gt1car


def _look(yaw, pitch):
    y, p = np.radians(yaw), np.radians(pitch)
    fwd = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.stack([right, up, -fwd])


def texture(tex, paint=0):
    o = np.frombuffer(tex[0x60:0x8060], np.uint8)
    img = np.zeros((256, 256), np.int32)
    img[:, 0::2] = (o & 15).reshape(256, 128)
    img[:, 1::2] = (o >> 4).reshape(256, 128)
    n = (len(tex) - 0x8060) // 0x200
    o = 0x8060 + (paint % n) * 0x200
    w = np.frombuffer(tex[o:o + 0x200], '<u2').astype(np.int32).reshape(16, 16)
    clut = np.stack([(w & 31) << 3, (w >> 5 & 31) << 3, (w >> 10 & 31) << 3], 2).astype(np.float32)
    return img, clut


class _Face:
    """A plain coloured polygon (the wheels, which the game draws itself)."""
    textured = False

    def __init__(self, idx, rgb):
        self.v = list(idx) + [0] * (4 - len(idx))
        self.quad = len(idx) == 4
        self.tail = bytes(rgb) + b')'

    def tris(self):
        a, b, c, d = self.v
        return [(a, b, c), (a, c, d)] if self.quad else [(a, b, c)]


def _wheels(car, radius, width, n=14):
    """Tyre and rim polygons at the car's wheel positions (model units)."""
    verts, faces = [], []
    for x, y, z, _ in car.wheels:
        cx, cy, cz = x / 4.0, y / 4.0, z / 4.0          # wheel units are 1/4096 m, the model's 1/1024 m
        out = 1 if x > 0 else -1
        xo, xi = cx + out * width / 2, cx - out * width / 2
        base = len(verts)
        for k in range(n):
            a = 2 * np.pi * k / n
            verts.append((xo, cy + radius * np.sin(a), cz + radius * np.cos(a)))
            verts.append((xi, cy + radius * np.sin(a), cz + radius * np.cos(a)))
        for k in range(n):
            o0, i0 = base + 2 * k, base + 2 * k + 1
            o1, i1 = base + 2 * ((k + 1) % n), base + 2 * ((k + 1) % n) + 1
            faces.append(_Face((o0, o1, i1, i0) if out > 0 else (o0, i0, i1, o1), (22, 22, 24)))
        cen = len(verts)
        verts.append((xo + out * 2, cy, cz))
        for k in range(n):
            o0, o1 = base + 2 * k, base + 2 * ((k + 1) % n)
            faces.append(_Face((cen, o0, o1) if out > 0 else (cen, o1, o0), (52, 53, 58)))
    return verts, faces


def render(car, tex, paint=0, size=(320, 200), yaw=-35, pitch=-15, lod=0, bg=(40, 45, 60), margin=0.92,
           light=(0.4, 0.5, 0.75), wheel_radius=0.31, wheel_width=0.21):
    """RGB picture (H x W x 3, uint8). yaw/pitch: camera direction (degrees), as in common/render.
    The wheels (drawn by the game, not part of the model) are added as plain discs."""
    c = gt1car.Car.read(car)
    img, clut = texture(tex, paint)
    L = c.lods[lod]
    wv, wf = _wheels(c, wheel_radius * 1024, wheel_width * 1024)
    V = np.array([v[:3] for v in L.verts] + wv, float)
    faces = list(L.faces)
    off = len(L.verts)
    for f in wf:
        f.v = [i + off if i or j < (4 if f.quad else 3) else 0 for j, i in enumerate(f.v)]
        faces.append(f)
    P = np.stack([V[:, 2], -V[:, 0], V[:, 1]], 1)       # game axes -> X forward, Y left, Z up
    W, H = size
    R = _look(yaw, pitch)
    pc = (P - (P.min(0) + P.max(0)) / 2) @ R.T
    ext = max(np.ptp(pc[:, 0]), np.ptp(pc[:, 1]) * W / H)
    s = margin * W / ext
    sx = pc[:, 0] * s + W / 2
    sy = H / 2 - pc[:, 1] * s
    sz = pc[:, 2]
    out = np.zeros((H, W, 3), np.float32) + np.asarray(bg, np.float32)
    zb = np.full((H, W), -1e18)
    lt = np.asarray(light, float)
    lt /= np.linalg.norm(lt)
    for f in faces:
        k = 4 if f.quad else 3
        q = V[f.v[:k]]
        outw = -np.cross(q[1] - q[0], q[2] - q[0])
        nr = np.array([outw[2], -outw[0], outw[1]])
        nr = nr / max(np.linalg.norm(nr), 1e-12)
        if (nr @ R.T)[2] <= 0:
            continue                                     # back face
        shade = 0.7 + 0.3 * abs(nr @ lt)
        tris = f.tris()
        uvs = None
        if f.textured:
            u = f.uv
            uvs = [(u[0], u[1], u[2]), (u[0], u[2], u[3])] if f.quad else [(u[0], u[1], u[2])]
            cl = (f.pal >> 6) * 4 + (f.pal & 3)
        for ti, tri in enumerate(tris):
            x, y, z = sx[list(tri)], sy[list(tri)], sz[list(tri)]
            x0, x1 = int(max(0, np.floor(x.min()))), int(min(W - 1, np.ceil(x.max())))
            y0, y1 = int(max(0, np.floor(y.min()))), int(min(H - 1, np.ceil(y.max())))
            if x0 > x1 or y0 > y1:
                continue
            d = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            if abs(d) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
            w1 = ((gx - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (gy - y[0])) / d
            w2 = ((x[1] - x[0]) * (gy - y[0]) - (gx - x[0]) * (y[1] - y[0])) / d
            w0 = 1 - w1 - w2
            m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
            if uvs is not None:
                uv = np.array(uvs[ti], float)
                uu = np.clip(w0 * uv[0, 0] + w1 * uv[1, 0] + w2 * uv[2, 0], 0, 255).astype(int)
                vv = np.clip(w0 * uv[0, 1] + w1 * uv[1, 1] + w2 * uv[2, 1], 0, 255).astype(int)
                ci = img[vv, uu]
                m &= ci != 0
                col = clut[cl][ci] * shade
            else:
                col = np.broadcast_to(np.array(list(f.tail[:3]), np.float32) * shade, gx.shape + (3,))
            sub = zb[y0:y1 + 1, x0:x1 + 1]
            m &= zz > sub
            sub[m] = zz[m]
            out[y0:y1 + 1, x0:x1 + 1][m] = col[m]
    return np.clip(out, 0, 255).astype(np.uint8)
