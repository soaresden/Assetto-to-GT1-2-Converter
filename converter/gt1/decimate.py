"""Simplification of a closed, consistently oriented triangle mesh by edge collapses, keeping it
closed and oriented (Garland & Heckbert quadrics, with the checks that keep a manifold):

- link condition: the two ends of an edge may only share the neighbours of the faces on that
  edge (otherwise the collapse pinches the surface);
- no face may turn over or become a sliver;
- vertices on the mirror plane (x = 0) stay on it, and the open border along that plane (a half
  car, mirrored afterwards) stays a border.

simplify(verts, tris, target_verts, on_plane) -> (verts, tris), the triangles in the same
orientation as the input.
"""
import heapq

import numpy as np

PLANE_WEIGHT = 1e4


def _quadrics(p, t):
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1)
    n = n / np.maximum(area, 1e-20)[:, None]
    pl = np.c_[n, -(n * a).sum(1)]
    return pl[:, :, None] * pl[:, None, :] * area[:, None, None]


def simplify(verts, tris, target, on_plane=None, min_quality=0.12):
    p = np.array(verts, np.float64)
    nv = len(p)
    plane = np.zeros(nv, bool) if on_plane is None else np.array(on_plane, bool)
    faces = [list(map(int, f)) for f in tris]
    alive = [True] * len(faces)
    vf = [set() for _ in range(nv)]
    for i, f in enumerate(faces):
        for v in f:
            vf[v].add(i)
    Q = np.zeros((nv, 4, 4))
    fq = _quadrics(p, np.array(faces))
    for k in range(3):
        np.add.at(Q, np.array(faces)[:, k], fq)
    Q[plane] += PLANE_WEIGHT * np.outer([1, 0, 0, 0], [1, 0, 0, 0])
    gone = np.zeros(nv, bool)
    version = np.zeros(nv, np.int64)

    def edge_faces(a, b):
        return [fi for fi in vf[a] if alive[fi] and b in faces[fi]]

    def neighbours(v):
        s = set()
        for fi in vf[v]:
            if alive[fi]:
                s.update(faces[fi])
        s.discard(v)
        return s

    def border(a, b):
        return len(edge_faces(a, b)) == 1

    def place(a, b):
        q = Q[a] + Q[b]
        cands = []
        if plane[a] and plane[b]:
            # stay on the plane: solve in y, z only, x = 0
            A = q[1:3, 1:3]
            rhs = -q[1:3, 3]
            try:
                yz = np.linalg.solve(A, rhs)
                cands.append(np.r_[0.0, yz])
            except np.linalg.LinAlgError:
                pass
        elif plane[a] or plane[b]:
            cands.append(p[a] if plane[a] else p[b])
        else:
            try:
                x = np.linalg.solve(q[:3, :3], -q[:3, 3])
                cands.append(x)
            except np.linalg.LinAlgError:
                pass
        if not (plane[a] ^ plane[b]):
            cands += [p[a], p[b], (p[a] + p[b]) / 2]
        best = None
        for x in cands:
            # keep the new point near the edge (the solution of a flat region can fly away)
            seg = p[b] - p[a]
            tpar = np.clip((x - p[a]) @ seg / max(seg @ seg, 1e-12), -0.25, 1.25)
            if np.linalg.norm(x - (p[a] + tpar * seg)) > 2 * np.linalg.norm(seg) + 1e-9:
                continue
            h = np.r_[x, 1.0]
            c = h @ q @ h
            if best is None or c < best[0]:
                best = (c, x)
        return best

    def quality(a, b, c):
        e = np.linalg.norm([b - a, c - b, a - c], axis=1)
        area2 = np.linalg.norm(np.cross(b - a, c - a))
        return 2 * np.sqrt(3) * area2 / max((e ** 2).sum(), 1e-20) * 1.0  # 1 for equilateral

    def ok(a, b, x):
        ef = edge_faces(a, b)
        if len(ef) not in (1, 2):
            return False
        # link condition
        third = set()
        for fi in ef:
            third.update(v for v in faces[fi] if v not in (a, b))
        if neighbours(a) & neighbours(b) != third:
            return False
        # the border of the half mesh: a border vertex may only slide along the border
        ba = any(border(a, n) for n in neighbours(a))
        bb = any(border(b, n) for n in neighbours(b))
        if ba and bb and len(ef) != 1:
            return False
        for v in (a, b):
            for fi in vf[v]:
                if not alive[fi] or fi in ef:
                    continue
                f = faces[fi]
                q0 = [p[u] for u in f]
                q1 = [x if u in (a, b) else p[u] for u in f]
                n0 = np.cross(q0[1] - q0[0], q0[2] - q0[0])
                n1 = np.cross(q1[1] - q1[0], q1[2] - q1[0])
                if n0 @ n1 <= 0.2 * np.linalg.norm(n0) * np.linalg.norm(n1):
                    return False
                if quality(*q1) < min_quality and quality(*q1) < quality(*q0):
                    return False
        return True

    heap = []

    def push(a, b):
        r = place(a, b)
        if r is None:
            return
        heapq.heappush(heap, (r[0], a, b, int(version[a]), int(version[b])))

    seen = set()
    for f in faces:
        for i in range(3):
            a, b = sorted((f[i], f[(i + 1) % 3]))
            if (a, b) not in seen:
                seen.add((a, b))
                push(a, b)
    count = nv
    while count > target and heap:
        c, a, b, va, vb = heapq.heappop(heap)
        if gone[a] or gone[b] or version[a] != va or version[b] != vb:
            continue
        r = place(a, b)
        if r is None:
            continue
        x = r[1]
        if not ok(a, b, x):
            continue
        # collapse b into a, at x
        for fi in edge_faces(a, b):
            alive[fi] = False
            for v in faces[fi]:
                vf[v].discard(fi)
        for fi in list(vf[b]):
            if alive[fi]:
                faces[fi] = [a if v == b else v for v in faces[fi]]
                vf[a].add(fi)
        vf[b].clear()
        gone[b] = True
        p[a] = x
        plane[a] = plane[a] or plane[b]
        if plane[a]:
            p[a][0] = 0.0
        Q[a] += Q[b]
        version[a] += 1
        count -= 1
        for n in neighbours(a):
            push(*sorted((a, n)))
    keep = np.nonzero(~gone)[0]
    remap = -np.ones(nv, np.int64)
    remap[keep] = np.arange(len(keep))
    out = np.array([[remap[v] for v in f] for f, al in zip(faces, alive) if al])
    return p[keep], out, plane[keep]


def check(verts, tris):
    """Closed and consistently oriented: every directed edge once, its reverse once.
    Returns (open edges, edges used twice the same way, signed volume)."""
    e = {}
    for f in tris:
        for i in range(3):
            k = (f[i], f[(i + 1) % 3])
            e[k] = e.get(k, 0) + 1
    open_ = sum(1 for (a, b) in e if (b, a) not in e)
    dup = sum(1 for v in e.values() if v > 1)
    p = np.asarray(verts)[np.asarray(tris)]
    vol = np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6
    return open_, dup, vol


def pair_quads(verts, tris, max_angle=18.0):
    """Join neighbouring triangles into quads where they are nearly flat and the quad is convex:
    list of faces (3 or 4 vertex indices, going round, the triangles' orientation)."""
    p = np.asarray(verts)
    tris = [tuple(t) for t in tris]
    n = []
    for t in tris:
        c = np.cross(p[t[1]] - p[t[0]], p[t[2]] - p[t[0]])
        n.append(c / max(np.linalg.norm(c), 1e-20))
    edge = {}
    for i, t in enumerate(tris):
        for k in range(3):
            edge[(t[k], t[(k + 1) % 3])] = (i, k)
    cand = []
    cos_max = np.cos(np.radians(max_angle))
    for (a, b), (i, k) in edge.items():
        if (b, a) not in edge or a > b:
            continue
        j, l = edge[(b, a)]
        d = n[i] @ n[j]
        if d < cos_max:
            continue
        ti, tj = tris[i], tris[j]
        c = ti[(k + 2) % 3]         # ti = (a, b, c) rotated
        dd = tj[(l + 2) % 3]        # tj = (b, a, dd) rotated
        quad = (a, dd, b, c)        # going round: a -> dd -> b -> c (orientation of ti)
        q = p[list(quad)]
        nn = n[i] + n[j]
        conv = all(np.cross(q[(m + 1) % 4] - q[m], q[(m + 2) % 4] - q[(m + 1) % 4]) @ nn > 0 for m in range(4))
        if not conv:
            continue
        cand.append((-d, i, j, quad))
    cand.sort()
    used = set()
    out = []
    for _, i, j, quad in cand:
        if i in used or j in used:
            continue
        used.update((i, j))
        out.append(list(quad))
    out += [list(t) for i, t in enumerate(tris) if i not in used]
    return out
