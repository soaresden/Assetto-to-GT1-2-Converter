"""The MX-5 RF as a Gran Turismo (PS1) car: a .car model and a .tex texture made from the Assetto
Corsa model, on the template of the car it replaces (header, shadow and wheel picture kept).

- Shell: rings around the body (shell.py), three levels of detail.
- Texture: side, top, front and rear pictures of the detailed model (views.py) packed in the
  256x192 texels the faces can use; each face takes its texels from the view it faces (left side,
  left half of the top...: mirrored). 4 bits per texel, one 16-colour CLUT per face.
- CLUTs: 1-11 for the body (paint shades + fixed colours, chosen per group of faces), 14 for the
  faces of the rear lamps and 15 the same with the lamps lit (brakes), 0 (wheel) and 12, 13 kept
  from the template. Eight paints: the paint shades change, the rest stays.
"""
import struct

import numpy as np

import decimate
import gt1car
import shell
import views as V

UNIT = 1024.0          # model units per metre (LOD scale 18)
WHEEL_UNIT = 4096.0    # wheel positions: units per metre
TEX_V0, TEX_V1 = 32, 224   # rows the faces can use
WHEEL_TEXELS = (0, 32, 48, 80)  # u0, v0, u1, v1: the wheel picture the game draws the wheels with

# MX-5 RF paints (sRGB), in the order of the template's eight paints
PAINTS = [
    ('Deep Crystal Blue', (24, 40, 84)),
    ('Soul Red Crystal', (150, 16, 26)),
    ('Machine Grey', (88, 90, 94)),
    ('Jet Black', (14, 14, 16)),
    ('Snowflake White Pearl', (226, 226, 220)),
    ('Ceramic', (184, 186, 182)),
    ('Polymetal Grey', (96, 108, 114)),
    ('Platinum Quartz', (156, 150, 140)),
]

LODS = [200, 90, 40]   # vertices of the three levels of detail (the disc's cars: up to 272)


def paint_rgb(base, shade):
    """Colour of the paint for a shade (0 dark .. 1 lit), 0..1 floats."""
    b = np.asarray(base, np.float32) / 255
    s = np.asarray(shade, np.float32)[..., None]
    c = b * (0.25 + 1.05 * s) + 0.35 * np.clip(s - 0.8, 0, None) * 5 * (1 - b) * 0.4
    return np.clip(c, 0, 1)


def ps1(rgb):
    """0..1 RGB -> PS1 15-bit words (never 0, which is transparent)."""
    q = np.clip(np.round(np.asarray(rgb) * 31), 0, 31).astype(np.uint16)
    w = q[..., 0] | (q[..., 1] << 5) | (q[..., 2] << 10)
    return np.where(w == 0, 0x8000, w).astype(np.uint16)


# ---------------------------------------------------------------------------------------------
ARCH = 0.37   # half length of a wheel arch (m)


def station_list(z0, z1, wheel_z, n):
    """n stations from rear to front: both ends, the edges and the middle of the wheel arches
    (so that the shell tucks in under the arches only), the others in the longest gaps."""
    fixed = [z0 + 0.01, z1 - 0.01]
    for wz in wheel_z:
        fixed += [wz - ARCH, wz, wz + ARCH]
    st = sorted(z for z in fixed if z0 < z < z1)
    st = st[:n]
    while len(st) < n:
        gaps = np.diff(st)
        k = int(gaps.argmax())
        st.insert(k + 1, (st[k] + st[k + 1]) / 2)
    return np.array(st)


def make_shell(sp, st, target, wheel_z, smooth=False, stations=56, per_side=16, log=None):
    """Shell vertices (metres), faces (lists of 3 or 4 indices, going round; the game's winding:
    cross(p1 - p0, p2 - p0) points inwards), vertex normals (outwards).

    A detailed shell first (rings of 2 * per_side + 2 points at `stations` stations, the edges and
    middles of the wheel arches among them), its right half simplified to about target / 2
    vertices with the points of the centre line kept on it, then mirrored: closed, symmetric, and
    no face crosses the centre line (the textures are mirrored)."""
    z0, z1 = sp[:, 2].min(), sp[:, 2].max()
    zs = station_list(z0, z1, wheel_z, stations)
    v, _, _ = shell.build(sp, st, zs, per_side, smooth=smooth)
    K = 2 * per_side + 2
    R = len(zs)
    H = per_side + 2                       # right half of a ring: top (x=0) .. bottom (x=0)
    rings = v.reshape(R, K, 3)[:, :H].copy()
    rings[:, 0, 0] = rings[:, -1, 0] = 0.0
    pts = [tuple(x) for x in rings.reshape(-1, 3)]
    idx = lambda r, k: r * H + k
    tris = []
    for r in range(R - 1):
        for k in range(H - 1):
            a, b, c, d = idx(r, k), idx(r, k + 1), idx(r + 1, k + 1), idx(r + 1, k)
            tris += [(a, b, c), (a, c, d)]
    # end caps: the right half of each end ring, joined to its projection on the centre plane
    for r in (0, R - 1):
        cen = {}
        for k in range(H):
            if k in (0, H - 1):
                cen[k] = idx(r, k)
            else:
                q = list(rings[r, k])
                q[0] = 0.0
                cen[k] = len(pts)
                pts.append(tuple(q))
        for k in range(H - 1):
            a, b, c, d = idx(r, k), idx(r, k + 1), cen[k + 1], cen[k]
            f = [(a, b, c), (a, c, d)]
            if r == 0:
                f = [t[::-1] for t in f]
            tris += f
    pts = np.array(pts)
    tris = [t for t in tris if len(set(t)) == 3]
    # drop zero-area triangles where a ring point sits on the centre line
    on_plane = np.abs(pts[:, 0]) < 1e-9
    # orientation of the half: the same as the stations' quads; checked on the whole shell below
    hv, ht, hplane = decimate.simplify(pts, np.array(tris), max(8, target // 2), on_plane)
    # mirror: left half = right half with x negated, faces reversed; centre line points shared
    nh = len(hv)
    mirror = np.arange(nh) + nh
    mirror[hplane] = np.nonzero(hplane)[0]
    lv = hv * [-1, 1, 1]
    verts = np.vstack([hv, lv])
    faces = [tuple(t) for t in ht] + [tuple(mirror[t][::-1]) for t in ht]
    used = np.unique(np.array(faces))
    remap = -np.ones(len(verts), np.int64)
    remap[used] = np.arange(len(used))
    verts = verts[used]
    faces = np.array([[remap[i] for i in f] for f in faces])
    open_, dup, vol = decimate.check(verts, faces)
    if vol > 0:                            # the game's winding: cross product inwards
        faces = faces[:, ::-1]
        open_, dup, vol = decimate.check(verts, faces)
    if open_ or dup:
        raise ValueError('shell not closed: %d open edges, %d repeated' % (open_, dup))
    quads = decimate.pair_quads(verts, faces)
    # no quad across the centre line
    out = []
    for f in quads:
        x = verts[f, 0]
        if len(f) == 4 and x.min() < -1e-6 and x.max() > 1e-6:
            out += [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]
        else:
            out.append(list(f))
    vn = np.zeros_like(verts)
    for f in out:
        p = verts[f]
        n = -np.cross(p[1] - p[0], p[2] - p[0])
        if len(f) == 4:
            n = n - np.cross(p[2] - p[0], p[3] - p[0])
        for i in f:
            vn[i] += n
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    if log:
        log('    shell: %d vertices, %d faces (%d quads), closed, volume %.2f m3'
            % (len(verts), len(out), sum(len(f) == 4 for f in out), -vol))
    return verts, out, vn


# ---------------------------------------------------------------------------------------------
LIGHTS = {'side': (0.35, 0.85, 0.4), 'top': (0.35, 0.85, 0.4), 'front': (0.35, 0.85, 0.4),
          'rear': (0.35, 0.85, -0.4)}


class Atlas:
    """Where each view goes in the texture, and the pictures."""

    def __init__(self, scene, sp, hull, v0=TEX_V0, v1=TEX_V1, side_v=WHEEL_TEXELS[3]):
        """v0, v1: the rows the faces can use; side_v: first row of the side picture (under the
        wheel picture in GT1)."""
        self.v0, self.v1 = v0, v1
        lo, hi = sp.min(0), sp.max(0)
        hw = np.abs(sp[:, 0]).max()
        L, Hh = hi[2] - lo[2], hi[1] - lo[1]
        rows = v1 - v0
        # front and rear pictures in a column on the right, side and top pictures to their left
        r_fr = 0.68 * (rows / 2) / Hh
        col = int(np.ceil(hw * r_fr))
        r_main = min((256 - col) / L, (rows - (side_v - v0)) / (Hh + hw))
        self.boxes = {
            'side': ((lo[2], hi[2]), (-hi[1], -lo[1])),
            'top': ((lo[2], hi[2]), (0, hw)),
            'front': ((0, hw), (-hi[1], -lo[1])),
            'rear': ((0, hw), (-hi[1], -lo[1])),
        }
        side_h = int(np.ceil(Hh * r_main))
        self.place = {  # view: (u origin, v origin, texels per metre)
            'side': (0, side_v, r_main),
            'top': (0, side_v + side_h, r_main),
            'front': (256 - col, v0, r_fr),
            'rear': (256 - col, v0 + int(np.ceil(Hh * r_fr)), r_fr),
        }
        # pictures at 4x the texel density, reduced later. The light comes from in front of each
        # picture's camera (and from above): with one light from the front of the car, the rear
        # picture was lit by the ambient term only and the back of the car came out dark and dirty
        # (the game shades the faces again on top of it).
        self.pics = {v: V.render(scene, v, self.place[v][2] * 4, self.boxes[v], hull=hull, light=LIGHTS[v])
                     for v in self.boxes}

    def uv(self, view, p):
        """Texel coordinates of points p (Nx3) in a view (|x| for the mirrored halves)."""
        (ua, us), (va, vs), _ = V.VIEWS[view]
        q = np.array(p, np.float64)
        q[:, 0] = np.abs(q[:, 0])
        (u0, _), (v0, _) = self.boxes[view]
        ou, ov, r = self.place[view]
        return np.stack([ou + (q[:, ua] * us - u0) * r, ov + (q[:, va] * vs - v0) * r], 1)

    def miss(self, view, p):
        """How far (m) the points p are from what the view's picture shows at their place (large:
        something else hides them in that picture)."""
        pic = self.pics[view]
        (ua, us), (va, vs), (da, ds) = V.VIEWS[view]
        q = np.array(p, np.float64)
        q[:, 0] = np.abs(q[:, 0])
        (u0, _), (v0, _) = self.boxes[view]
        W, H = pic['size']
        r = pic['res']
        x = np.clip(((q[:, ua] * us - u0) * r).astype(int), 0, W - 1)
        y = np.clip(((q[:, va] * vs - v0) * r).astype(int), 0, H - 1)
        d = pic['depth'][y, x]
        return np.where(d < -1e8, 9.0, np.abs(q[:, da] * ds - d))

    def texels(self):
        """Per texel of the texture: class, rgb, shade (from the pictures, 4x4 reduced)."""
        cls = np.zeros((256, 256), np.int8)
        rgb = np.zeros((256, 256, 3), np.float32)
        shade = np.zeros((256, 256), np.float32)
        for v, pic in self.pics.items():
            W, H = pic['size']
            c = pic['cls'].reshape(H, W)
            col = pic['rgb'].reshape(H, W, 3)
            sh = pic['shade'].reshape(H, W)
            c, col, sh = _grow(c, col, sh)
            ou, ov, r = self.place[v]
            w, h = W // 4, H // 4
            blk = lambda a: a[:h * 4, :w * 4].reshape(h, 4, w, 4, *a.shape[2:])
            cb = blk(c)
            col4 = blk(col).mean((1, 3))
            # class of a texel: the most common of its 16 pixels; paint shade: mean over paint pixels
            flat = cb.transpose(0, 2, 1, 3).reshape(h, w, 16)
            counts = np.stack([(flat == k).sum(2) for k in range(7)], 2)
            tc = counts.argmax(2)
            pm = (flat == V.PAINT)
            shb = blk(sh).transpose(0, 2, 1, 3).reshape(h, w, 16)
            tsh = (shb * pm).sum(2) / np.maximum(pm.sum(2), 1)
            u1, v1 = min(256, ou + w), min(self.v1, ov + h)
            cls[ov:v1, ou:u1] = tc[:v1 - ov, :u1 - ou]
            rgb[ov:v1, ou:u1] = col4[:v1 - ov, :u1 - ou]
            shade[ov:v1, ou:u1] = tsh[:v1 - ov, :u1 - ou]
        return cls, rgb, shade


def _grow(cls, rgb, shade, steps=6):
    """Empty pixels take the values of a filled neighbour, a few pixels out from the car (faces
    of the shell stick out a little past the outline: no transparent holes)."""
    cls, rgb, shade = cls.copy(), rgb.copy(), shade.copy()
    for _ in range(steps * 4):
        empty = cls == V.EMPTY
        if not empty.any():
            break
        changed = False
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src_c = np.roll(np.roll(cls, dy, 0), dx, 1)
            take = empty & (src_c != V.EMPTY)
            if take.any():
                cls[take] = src_c[take]
                rgb[take] = np.roll(np.roll(rgb, dy, 0), dx, 1)[take]
                shade[take] = np.roll(np.roll(shade, dy, 0), dx, 1)[take]
                empty = cls == V.EMPTY
                changed = True
        if not changed:
            break
    return cls, rgb, shade


# ---------------------------------------------------------------------------------------------
AXES = {'side': (0, 1), 'top': (1, 1), 'front': (2, 1), 'rear': (2, -1)}


def face_view(p, n, atlas=None, tol=0.12):
    """The view a face takes its texels from (None: underside, plain black): among the views
    that see it (nothing in front of it in their picture), the one it faces most."""
    n = n / max(np.linalg.norm(n), 1e-9)
    if n[1] < -0.9:
        return None            # facing the ground
    m = n.copy()
    m[0] = abs(m[0])           # the left side uses the right side's pictures
    best, score = None, -9.0
    for view, (k, sgn) in AXES.items():
        c = m[k] * sgn
        if c <= 0.05:
            continue
        if atlas is not None:
            pts = np.vstack([p, p.mean(0)])
            if np.median(atlas.miss(view, pts)) > tol:
                c -= 2.0           # hidden in that picture: only if nothing better
        if c > score:
            best, score = view, c
    return best


def raster_mask(uv, size=256):
    """Texels covered by the polygon uv (Kx2)."""
    from PIL import Image, ImageDraw
    m = Image.new('L', (size, size), 0)
    ImageDraw.Draw(m).polygon([tuple(x) for x in uv], fill=1, outline=1)
    return np.array(m, bool)


def kmeans(x, k, iters=12, seed=0):
    x = np.asarray(x, np.float32)
    if len(x) <= k:
        return np.concatenate([x, np.repeat(x[-1:], k - len(x), 0)]) if len(x) else np.zeros((k,) + x.shape[1:], np.float32)
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        d = ((x[:, None] - c[None]) ** 2).reshape(len(x), k, -1).sum(2)
        a = d.argmin(1)
        for j in range(k):
            if (a == j).any():
                c[j] = x[a == j].mean(0)
    return c


def build_texture(atlas, lods, template_tex, night=False, paints=None, wheel_pic=None, arches=None):
    """(tex bytes, per LOD: per face (clut, uv list or None)). Faces: lists of (verts, faces, normals).
    arches: (centre, radius) of the wheels: their discs are transparent in the side picture."""
    paints = paints or PAINTS
    index, cluts, faces_out = texture_layout(atlas, lods, paints=paints)
    if arches:
        arch_windows(index, atlas, arches)
    # bytes: template header, index image, CLUTs (CLUT 0, 12 and 13 from the template)
    t = bytearray(template_tex)
    old = np.frombuffer(bytes(t[0x8060:]), '<u2').reshape(-1, 16, 16)
    tidx = np.zeros((256, 256), np.uint8)
    o = np.frombuffer(bytes(t[0x60:0x8060]), np.uint8)
    tidx[:, 0::2] = (o & 15).reshape(256, 128)
    tidx[:, 1::2] = (o >> 4).reshape(256, 128)
    u0, v0, u1, v1 = WHEEL_TEXELS
    index[v0:v1, u0:u1] = tidx[v0:v1, u0:u1]
    if wheel_pic is not None:          # the model's own wheel (wheelpic.py)
        index[v0:v0 + 48, u0:u0 + 48] = wheel_pic
    packed = (index[:, 0::2] | (index[:, 1::2] << 4)).astype(np.uint8)
    t[0x60:0x8060] = packed.tobytes()
    words = ps1(cluts)
    words[:, :, 0] = 0               # colour 0: transparent (no texel uses it)
    # as many palettes as the template has paints (the eight MX-5 paints over and over)
    words = words[np.arange(len(old)) % len(paints)].copy()
    for p in range(len(old)):
        for k in (0, 12, 13):
            words[p, k] = old[p, k]
    t[0x8060:0x8060 + len(old) * 0x200] = words.astype('<u2').tobytes()
    return bytes(t), faces_out, index, cluts


def texture_layout(atlas, lods, body_groups=11, front_lamps=None, paints=PAINTS):
    """The texture of the LODs (lists of (verts, faces, normals)) on the atlas: (index image 256 x
    256, CLUTs (paints x 16 x 16 x rgb 0..1), {(lod, face): (clut or None, uv)}).
    CLUTs: 1..body_groups for the body, 14 the rear lamps and 15 the same lit (brakes);
    front_lamps: a CLUT of its own for the faces of the front lamps (None: with the body);
    paints: (name, sRGB) of the paints, one set of CLUTs each."""
    cls, rgb, shade = atlas.texels()
    # every face of every LOD: its view, its uv, its texels
    info = []
    for li, (verts, faces, vn) in enumerate(lods):
        for fi, f in enumerate(faces):
            p = verts[f]
            n = -np.cross(p[1] - p[0], p[2] - p[0])
            view = face_view(p, n, atlas)
            if view is None:
                info.append((li, fi, None, None, None))
                continue
            uv = atlas.uv(view, p)
            uv = np.clip(np.round(uv), 0, 255)
            uv[:, 1] = np.clip(uv[:, 1], atlas.v0, atlas.v1 - 1)
            m = raster_mask(uv)
            info.append((li, fi, view, uv, m))
    # the CLUT of each face of LOD 0: rear lamp faces -> 14; the others in 11 groups by their
    # colours. A texel belongs to the group of the first LOD 0 face over it; the faces of the
    # other LODs take the group of most of their texels (same texture, same colours).
    groups = {}
    feats, body = [], []
    for idx, (li, fi, view, uv, m) in enumerate(info):
        if m is None or li != 0:
            continue
        c = cls[m]
        if not len(c):
            continue
        if (c == V.LAMP_R).mean() > 0.04:
            groups[idx] = 14
            continue
        if front_lamps is not None and (c == V.LAMP_F).mean() > 0.04:
            groups[idx] = front_lamps
            continue
        non = rgb[m][c != V.PAINT]
        feats.append(np.r_[non.mean(0) if len(non) else [0.5, 0.5, 0.5], (c == V.PAINT).mean() * 0.6,
                           (c == V.LAMP_F).mean() * 0.8, (c == V.GLASS).mean() * 0.5])
        body.append(idx)
    if body:
        cent = kmeans(np.array(feats), min(body_groups, len(body)))
        a_ = ((np.array(feats)[:, None] - cent[None]) ** 2).sum(2).argmin(1)
        for idx, g in zip(body, a_):
            groups[idx] = 1 + int(g)
    gmap = np.zeros((256, 256), np.int8)
    for idx in sorted(groups, reverse=True):
        gmap[info[idx][4]] = groups[idx]
    for idx, (li, fi, view, uv, m) in enumerate(info):
        if m is None or li == 0:
            continue
        g = gmap[m]
        g = g[g > 0]
        groups[idx] = int(np.bincount(g).argmax()) if len(g) else 1
        gmap[m & (gmap == 0)] = groups[idx]
    # palettes: per group, paint levels + fixed colours
    index = np.zeros((256, 256), np.uint8)
    cluts = np.zeros((len(paints), 16, 16, 3), np.float32)
    paint_slots = {}
    for g in range(1, 16):
        sel = gmap == g
        if g == 15 or not sel.any():
            continue
        c, col, sh = cls[sel], rgb[sel], shade[sel]
        is_p = c == V.PAINT
        np_ = int(np.clip(round(is_p.mean() * 15), 0 if not is_p.any() else 2, 13))
        nf = 15 - np_
        levels = np.sort(kmeans(sh[is_p][:, None], np_)[:, 0]) if np_ else np.zeros(0)
        fixed = kmeans(col[~is_p], nf) if nf else np.zeros((0, 3))
        if g == 14:
            fixed = kmeans(col[~is_p], nf)
        # indices: 1..np_ paint levels, then the fixed colours
        ix = np.zeros(sel.sum(), np.uint8)
        if np_:
            ix[is_p] = 1 + np.abs(sh[is_p][:, None] - levels[None]).argmin(1)
        if nf:
            dd = ((col[~is_p][:, None] - fixed[None]) ** 2).sum(2)
            ix[~is_p] = 1 + np_ + dd.argmin(1)
        index[sel] = ix
        for p, (_, base) in enumerate(paints):
            pal = np.zeros((16, 3), np.float32)
            if np_:
                pal[1:1 + np_] = paint_rgb(base, levels)
            pal[1 + np_:] = fixed
            cluts[p, g] = pal
        paint_slots[g] = np_
    # 15: the rear lamp CLUT with the lamps lit (red and amber brighter)
    lit = cluts[:, 14].copy()
    for p in range(len(paints)):
        npl = paint_slots.get(14, 0)
        f = lit[p, 1 + npl:]
        red = (f[:, 0] > f[:, 1] * 1.4) & (f[:, 0] > 0.15)
        f[red] = np.clip(f[red] * 1.9 + [0.15, 0.02, 0.02], 0, 1)
        lit[p, 1 + npl:] = f
    cluts[:, 15] = lit
    faces_out = {}
    for idx, (li, fi, view, uv, m) in enumerate(info):
        faces_out[(li, fi)] = (groups.get(idx), uv)
    return index, cluts, faces_out


def arch_windows(index, atlas, arches, margin=0.01):
    """Texels of the side picture over the wheels -> index 0 (transparent in every CLUT): the
    faces of the shell that pass in front of a tyre (the shell is a closed low-poly envelope,
    its faces cross the wheel arches) let the game's wheel show through, as on the disc's cars.
    """
    ou, ov, r = atlas.place['side']
    (z0, z1), (y0, y1) = atlas.boxes['side']
    u_end, v_end = int(ou + (z1 - z0) * r), int(ov + (y1 - y0) * r)
    vv, uu = np.mgrid[:256, :256] + 0.5
    inside_pic = (uu >= ou) & (uu < u_end) & (vv >= ov) & (vv < v_end)
    total = 0
    for c, rad in arches:
        cu, cv = atlas.uv('side', np.array([c]))[0]
        m = inside_pic & (np.hypot(uu - cu, vv - cv) <= (rad + margin) * r)
        total += int((m & (index != 0)).sum())
        index[m] = 0
    return total


def over_wheels(lods, wheels, margin=0.06, inner=0.3):
    """(lod, face) of the faces of the shell's sides that pass over a wheel seen from the side
    (a corner or the centre within the wheel's disc, plus a margin), for build_car's
    before_wheels. wheels: (centre, radius)."""
    out = set()
    for li, (verts, faces, vn) in enumerate(lods):
        verts = np.asarray(verts, float)
        for fi, f in enumerate(faces):
            p = verts[list(f)]
            q = np.vstack([p, p.mean(0)])
            if np.abs(q[-1, 0]) < inner:
                continue
            for c, rad in wheels:
                if (np.hypot(q[:, 1] - c[1], q[:, 2] - c[2]) < rad + margin).any():
                    out.add((li, fi))
                    break
    return out


def wheel_walls(lods, faces_tex, wheels, depth=0.03, around=0.12, above=0.08):
    """Black walls behind the wheels, facing out, in every LOD: the arches (transparent in the
    texture, arch_windows) show a dark wheel well, not the far side of the car. wheels: (centre,
    radius, width). Untextured faces (faces_tex entries (None, None))."""
    out = []
    for li, (verts, faces, vn) in enumerate(lods):
        verts = np.array(verts, float)
        vn = np.array(vn, float)
        faces = [list(f) for f in faces]
        lo = verts[:, 1].min()
        for c, rad, width in wheels:
            sgn = 1.0 if c[0] > 0 else -1.0
            x = c[0] - sgn * (width / 2 + depth)
            z0, z1 = c[2] - rad - around, c[2] + rad + around
            y0, y1 = lo, c[1] + rad + above
            k = len(verts)
            quad = [(x, y0, z0), (x, y0, z1), (x, y1, z1), (x, y1, z0)]
            verts = np.vstack([verts, quad])
            vn = np.vstack([vn, [[sgn, 0, 0]] * 4])
            # the game's winding: cross(p1 - p0, p2 - p0) points inwards (towards x = 0)
            f = [k, k + 1, k + 2, k + 3] if sgn > 0 else [k, k + 3, k + 2, k + 1]
            faces_tex[(li, len(faces))] = ('wall', None)
            faces.append(f)
        out.append((verts, faces, vn))
    return out


# ---------------------------------------------------------------------------------------------
def clut_word(k):
    """Palette word of CLUT k (as the disc's cars: (w >> 6) * 4 + (w & 3) = k)."""
    return ((k // 4) << 6) | (k % 4)


def build_car(template_car, lods, faces_tex, wheels, lo_y, menu_wheels=None, before_wheels=()):
    """The .car bytes: template header and shadow, our LODs and wheel positions. before_wheels:
    (lod, face) drawn before the wheels (bit 63), as the faces over the arches of the disc's cars."""
    car = gt1car.Car.read(template_car)
    tmpl_face = {}
    for L in car.lods:
        for f in L.faces:
            tmpl_face.setdefault((f.quad, f.textured), f)
    new_lods = []
    for li, (verts, faces, vn) in enumerate(lods):
        T = car.lods[min(li, len(car.lods) - 1)]
        L = gt1car.Lod()
        L.head = T.head
        L.scale = 18
        q = np.round((verts - [0, lo_y, 0]) * UNIT).astype(int)
        L.verts = [(int(x), int(y), int(z), 0) for x, y, z in q]
        nn = np.round(vn * 4000).astype(int)
        L.normals = [(int(x), int(y), int(z), 0) for x, y, z in nn]
        for fi, f in enumerate(faces):
            clut, uv = faces_tex[(li, fi)]
            quad = len(f) == 4
            wall = clut == 'wall'
            if wall:
                clut = None
            F = gt1car.Face()
            F.quad, F.textured = quad, clut is not None
            idx = [int(i) for i in f] + [0] * (4 - len(f))
            F.v = idx
            F.n = idx[:]
            # bits 27-31: 16 + 2 (lit, like most of the disc's faces)
            F.rest = (18 << 27)
            F.pad = b'\0\0'
            if F.textured:
                F.tail = b'\xff\xff\xff' + bytes([0x2D if quad else 0x25])
                uvl = [(int(a), int(b)) for a, b in uv] + [(0, 0)] * (4 - len(uv))
                F.uv = uvl
                F.pal = clut_word(clut)
            else:
                F.rest = 16 << 27  # unlit plain colour
                F.n = [0, 0, 0, 0]
                F.tail = b'\x10\x10\x10' + bytes([0x29 if quad else 0x21])
                F.uv = None
                F.pal = None
            if wall:
                # the wheel wells of the disc's cars: black, unlit
                F.tail = b'\x00\x00\x00' + bytes([0x29 if quad else 0x21])
            if wall or (li, fi) in before_wheels:
                # bit 63: drawn before the wheels, which then show in front (the disc's cars
                # have it on their wheel wells and on the faces of the body over the arches)
                F.rest |= 1 << 63
            L.faces.append(F)
        # header: counts are set on writing; bounding box and sphere here
        h = bytearray(L.head)
        mn, mx = q.min(0), q.max(0)
        struct.pack_into('<8h', h, 0x14, int(mn[0]), int(mn[1]), int(mn[2]), 0, int(mx[0]), int(mx[1]), int(mx[2]), 0)
        struct.pack_into('<H', h, 0x26, int(np.ceil(np.linalg.norm(q, axis=1).max())))
        L.head = bytes(h)
        new_lods.append(L)
    car.lods = new_lods
    # wheels: front left, front right, rear left, rear right
    if menu_wheels is not None:        # (width, radius) front, rear, metres
        car.menu_wheels = tuple(int(round(v * WHEEL_UNIT)) for v in menu_wheels)
    car.wheels = [(int(round(x * WHEEL_UNIT)), int(round((y - lo_y) * WHEEL_UNIT)), int(round(z * WHEEL_UNIT)), 0)
                  for x, y, z in wheels]
    return car.write(), car
