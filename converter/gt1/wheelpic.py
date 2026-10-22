"""The wheel picture of a Gran Turismo (PS1) car texture: texels u 0..47, v 32..79, through CLUT 0
(a ramp of greys from white, index 0, to black, index 14; index 15 is 0x0000: transparent). The
game draws the side of each wheel with it: the tyre's outer edge on the edge of the picture. In
its top left corner, outside the disc, the template has the tread strip of the tyre: kept.

picture(scene, wheel) draws the model's wheel seen from outside (its rim, the tyre's side) into
such a picture, in the template's greys. The disc is checked: round (its outline at the same
distance from the centre all round) and centred, or the template's picture is kept.
"""
import numpy as np

import views as V

SIZE = 48
U0, V0 = 0, 32


def template(tex):
    """The template's wheel picture (48x48 indices) and CLUT 0 (16 RGB 0..1)."""
    o = np.frombuffer(bytes(tex[0x60:0x8060]), np.uint8)
    idx = np.zeros((256, 256), np.uint8)
    idx[:, 0::2] = (o & 15).reshape(256, 128)
    idx[:, 1::2] = (o >> 4).reshape(256, 128)
    w = np.frombuffer(bytes(tex[0x8060:0x8080]), '<u2').astype(int)
    rgb = np.stack([w & 31, (w >> 5) & 31, (w >> 10) & 31], 1) / 31.0
    return idx[V0:V0 + SIZE, U0:U0 + SIZE].copy(), rgb, w


def picture(scene, wheel, tex, log=print):
    """48x48 indices of CLUT 0 for the wheel (a wheels.Wheel seen from its outer side), or None
    when the drawing does not pass the checks (the template's picture stays)."""
    old, clut, words = template(tex)
    c, r = wheel.centre, wheel.radius
    res = SIZE * 4 / (2 * r)
    box = ((c[2] - r, c[2] + r), (-(c[1] + r), -(c[1] - r)))
    view = 'side'
    pic = V.render(scene, view, res, box, light=(0.6, 0.6, 0.5))
    W, H = pic['size']
    cls = pic['cls'].reshape(H, W)
    rgb = pic['rgb'].reshape(H, W, 3)
    if wheel.centre[0] < 0:             # a left wheel seen from the right: mirror it
        cls, rgb = cls[:, ::-1], rgb[:, ::-1]
    # 4x4 reduction
    h, w = H // 4, W // 4
    full = (cls[:h * 4, :w * 4] != V.EMPTY).reshape(h, 4, w, 4).mean((1, 3))
    col = rgb[:h * 4, :w * 4].reshape(h, 4, w, 4, 3).mean((1, 3))
    full = np.pad(full, ((0, SIZE - h), (0, SIZE - w)))[:SIZE, :SIZE]
    col = np.pad(col, ((0, SIZE - h), (0, SIZE - w), (0, 0)))[:SIZE, :SIZE]
    yy, xx = np.mgrid[:SIZE, :SIZE] + 0.5
    rr = np.hypot(xx - SIZE / 2, yy - SIZE / 2)
    disc = rr <= SIZE / 2
    # checks: the drawn part fills the disc (round, centred) and nothing much outside it
    cover_in = full[disc & (rr < SIZE / 2 - 2)].mean()
    cover_out = full[rr > SIZE / 2 + 1].mean() if (rr > SIZE / 2 + 1).any() else 0
    ys, xs = np.nonzero(full > 0.5)
    off = np.hypot(xs.mean() + 0.5 - SIZE / 2, ys.mean() + 0.5 - SIZE / 2) if len(xs) else 99
    ring = [full[(rr > SIZE / 2 - 3) & (rr < SIZE / 2 - 1) & (np.arctan2(yy - 24, xx - 24) // (np.pi / 4) == k)].mean()
            for k in range(-4, 4)]
    # (holes inside, between the spokes, are the dark inside of the wheel: filled below)
    ok = cover_in > 0.6 and cover_out < 0.15 and off < 1.0 and min(ring) > 0.9
    log('  image de roue : disque rempli à %.0f %%, débord %.0f %%, décentré de %.1f texel, bord %.0f..%.0f %% -> %s'
        % (100 * cover_in, 100 * cover_out, off, 100 * min(ring), 100 * max(ring), 'gardée' if ok else 'REFUSÉE (roue du modèle d\'origine)'))
    if not ok:
        return None
    # greys of CLUT 0 (indices 0..14)
    lum = col.mean(2)
    lum = lum / max(lum[disc].max(), 1e-6)
    greys = clut[:15].mean(1)
    idx = np.abs(lum[..., None] ** 0.8 - greys[None, None]).argmin(2).astype(np.uint8)
    idx[full < 0.5] = 13                 # seen through the rim: the dark inside
    # the template's wheel is an octagon (its corners just outside the circle): between the
    # circle and the octagon, the colour of the tyre's edge; the tread strip of the top left
    # corner (outside the octagon) and the transparent corners stay as they are
    strip = (yy < 5) & (xx < 14)
    octagon = (old != 15) & ~strip
    edge = idx[disc & (rr > SIZE / 2 - 2)]
    tyre = np.uint8(np.bincount(edge).argmax()) if len(edge) else np.uint8(13)
    out = old.copy()
    out[octagon] = tyre
    out[disc & ~strip] = idx[disc & ~strip]
    out[~octagon & ~disc & ~strip] = 15
    return out
