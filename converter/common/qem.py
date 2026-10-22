"""Mesh simplification by quadric error metrics (Garland & Heckbert) with edge collapses.

simplify(pos, tri, target, weld_keys=None) -> new triangles (indices into the original vertices)

- Vertices are first welded: by position, or by `weld_keys` (for instance position + UV, so that
  texture seams stay seams).
- Open edges get an extra quadric (a plane through the edge, perpendicular to its face), so the
  outline of panels does not shrink.
- A collapse that flips a neighbouring face is refused.
The kept vertices are original ones (no new positions): their UVs, colours and normals stay valid.
"""
import heapq
from collections import defaultdict

import numpy as np

BOUNDARY_WEIGHT = 1000.0


def _plane_quadrics(p, tri):
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1)
    n = n / np.maximum(area, 1e-20)[:, None]
    d = -(n * a).sum(1)
    pl = np.c_[n, d]  # (F, 4)
    q = pl[:, :, None] * pl[:, None, :] * area[:, None, None]
    return q, n


def simplify(pos, tri, target, weld_keys=None):
    pos = np.asarray(pos, np.float64)
    tri = np.asarray(tri, np.int64)
    if len(tri) <= target:
        return tri
    # weld
    key = np.round(pos, 5) if weld_keys is None else weld_keys
    _, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    orig = rep  # welded vertex -> original vertex index
    t = inv[tri]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    p = pos[orig]
    nv = len(p)
    fq, fn = _plane_quadrics(p, t)
    Q = np.zeros((nv, 4, 4))
    for k in range(3):
        np.add.at(Q, t[:, k], fq)
    # boundary edges
    edges = np.sort(np.r_[t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=1)
    eface = np.tile(np.arange(len(t)), 3)
    ue, cnt = np.unique(edges, axis=0, return_counts=True)
    bset = {tuple(e) for e in ue[cnt == 1]}
    for (e, f) in zip(map(tuple, edges), eface):
        if e in bset:
            a, b = p[e[0]], p[e[1]]
            n = np.cross(b - a, fn[f])
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue
            n /= ln
            pl = np.r_[n, -n @ a]
            qq = np.outer(pl, pl) * BOUNDARY_WEIGHT * max(np.linalg.norm(b - a) ** 2, 1e-8)
            Q[e[0]] += qq
            Q[e[1]] += qq

    faces = [list(f) for f in t]
    alive = np.ones(len(faces), bool)
    vfaces = defaultdict(set)
    for i, f in enumerate(faces):
        for v in f:
            vfaces[v].add(i)
    parent = np.arange(nv)
    version = np.zeros(nv, np.int64)

    def cost(a, b):
        q = Q[a] + Q[b]
        best = None
        for v in (a, b):
            x = np.r_[p[v], 1.0]
            c = x @ q @ x
            if best is None or c < best[0]:
                best = (c, v)
        return best

    heap = []
    seen = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            e = (min(a, b), max(a, b))
            if e in seen:
                continue
            seen.add(e)
            c, keep = cost(*e)
            heapq.heappush(heap, (c, e[0], e[1], keep, version[e[0]], version[e[1]]))

    nface = len(faces)

    def flips(remove, keep):
        """Would moving `remove` onto `keep` flip or collapse a face?"""
        for fi in vfaces[remove]:
            if not alive[fi]:
                continue
            f = faces[fi]
            if keep in f:
                continue
            a, b, c = (p[v] for v in f)
            n0 = np.cross(b - a, c - a)
            g = [keep if v == remove else v for v in f]
            a, b, c = (p[v] for v in g)
            n1 = np.cross(b - a, c - a)
            if n0 @ n1 <= 1e-12 * (np.linalg.norm(n0) * np.linalg.norm(n1) + 1e-30):
                return True
        return False

    while nface > target and heap:
        c, a, b, keep, va, vb = heapq.heappop(heap)
        if version[a] != va or version[b] != vb or parent[a] != a or parent[b] != b:
            continue
        remove = b if keep == a else a
        if flips(remove, keep):
            continue
        # collapse remove -> keep
        parent[remove] = keep
        Q[keep] += Q[remove]
        for fi in list(vfaces[remove]):
            if not alive[fi]:
                continue
            f = faces[fi]
            if keep in f:
                alive[fi] = False
                nface -= 1
                for v in f:
                    vfaces[v].discard(fi)
                continue
            faces[fi] = [keep if v == remove else v for v in f]
            vfaces[keep].add(fi)
        vfaces[remove].clear()
        version[keep] += 1
        version[remove] += 1
        nbrs = set()
        for fi in vfaces[keep]:
            nbrs.update(faces[fi])
        nbrs.discard(keep)
        for n_ in nbrs:
            e = (min(keep, n_), max(keep, n_))
            c2, k2 = cost(*e)
            heapq.heappush(heap, (c2, e[0], e[1], k2, version[e[0]], version[e[1]]))
    out = np.array([faces[i] for i in range(len(faces)) if alive[i]], np.int64).reshape(-1, 3)
    return orig[out]
