"""Load the Assetto Corsa KN5 as world-space meshes."""
import numpy as np

import kn5


def load(path):
    """Returns (textures, materials, meshes, nodes).

    meshes: list of dict(name, path, mat (material dict), pos Nx3, nrm Nx3, uv Nx2, tri Mx3) in AC world space.
    nodes: {name: world 4x4 matrix (row-vector convention)}.
    """
    textures, materials, root = kn5.read(path)
    meshes, nodes = [], {}

    def walk(n, parent, path):
        p = path + '/' + n['name']
        if n['cls'] == 1:
            M = np.array(n['matrix'], np.float64).reshape(4, 4)
            W = M @ parent
            nodes[n['name']] = W
        else:
            W = parent
        if n['cls'] == 2 and n['active']:
            v = np.frombuffer(n['verts'], dtype=np.float32).reshape(-1, 11)
            pos = np.c_[v[:, 0:3], np.ones(len(v))] @ W
            nrm = v[:, 3:6] @ W[:3, :3]
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
            tri = np.frombuffer(n['idx'], dtype=np.uint16).astype(np.int64).reshape(-1, 3)
            meshes.append(dict(name=n['name'], path=p, mat=materials[n['mat']], pos=pos[:, :3],
                               nrm=nrm, uv=v[:, 6:8].copy(), tri=tri, visible=n.get('visible', 1),
                               transparent=n.get('transparent', 0), lod=n.get('lod')))
        for c in n['children']:
            walk(c, W, p)

    walk(root, np.eye(4), '')
    return textures, materials, meshes, nodes
