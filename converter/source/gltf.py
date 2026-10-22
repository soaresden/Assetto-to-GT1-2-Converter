"""glTF 2.0 models (.glb, or .gltf with its .bin and pictures): the triangles of every mesh of the
default scene in world space, with their material (base colour, base colour texture, alpha mode).

Only what the converter needs: TRIANGLES primitives (others are skipped), POSITION, NORMAL,
TEXCOORD_0 and indices, node matrices or translation / rotation / scale. Sparse accessors and
Draco / meshopt compression are refused with a message (export the model again without them).
"""
import base64
import io
import json
import os
import struct

import numpy as np
from PIL import Image

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
NORM = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}


def _read_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] == b'glTF':
        _, _, total = struct.unpack_from('<III', data, 0)
        p, j, binary = 12, None, b''
        while p < total:
            n, kind = struct.unpack_from('<I4s', data, p)
            chunk = data[p + 8:p + 8 + n]
            if kind == b'JSON':
                j = json.loads(chunk.decode('utf-8'))
            elif kind == b'BIN\0':
                binary = chunk
            p += 8 + n
        return j, binary
    return json.loads(data.decode('utf-8')), None


def _quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _local(node):
    if 'matrix' in node:
        return np.array(node['matrix'], np.float64).reshape(4, 4).T     # column-major
    M = np.eye(4)
    s = node.get('scale', [1, 1, 1])
    M[:3, :3] = _quat(node.get('rotation', [0, 0, 0, 1])) * np.asarray(s)[None, :]
    M[:3, 3] = node.get('translation', [0, 0, 0])
    return M


class Gltf:
    def __init__(self, path):
        self.path = path
        self.dir = os.path.dirname(os.path.abspath(path))
        self.j, self.glb = _read_file(path)
        used = set(self.j.get('extensionsRequired', []))
        bad = used & {'KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'KHR_mesh_quantization'}
        if bad:
            raise ValueError('%s : compression %s non prise en charge (réexporter le modèle sans)'
                             % (os.path.basename(path), ', '.join(sorted(bad))))
        self.buffers = {}
        self.images = {}

    def buffer(self, i):
        if i not in self.buffers:
            b = self.j['buffers'][i]
            uri = b.get('uri')
            if uri is None:
                data = self.glb
            elif uri.startswith('data:'):
                data = base64.b64decode(uri.split(',', 1)[1])
            else:
                with open(os.path.join(self.dir, uri), 'rb') as f:
                    data = f.read()
            self.buffers[i] = data
        return self.buffers[i]

    def view(self, i):
        v = self.j['bufferViews'][i]
        b = self.buffer(v['buffer'])
        o = v.get('byteOffset', 0)
        return b[o:o + v['byteLength']], v.get('byteStride')

    def accessor(self, i):
        a = self.j['accessors'][i]
        if 'sparse' in a:
            raise ValueError('accesseur « sparse » non pris en charge')
        dt, nc, n = COMP[a['componentType']], NCOMP[a['type']], a['count']
        if 'bufferView' not in a:
            return np.zeros((n, nc), dt)
        raw, stride = self.view(a['bufferView'])
        off = a.get('byteOffset', 0)
        size = np.dtype(dt).itemsize * nc
        if stride and stride != size:
            rows = np.frombuffer(raw, np.uint8, count=stride * (n - 1) + size, offset=off)
            idx = (np.arange(n)[:, None] * stride + np.arange(size)[None]).ravel()
            out = rows[idx].view(dt).reshape(n, nc)
        else:
            out = np.frombuffer(raw, dt, count=n * nc, offset=off).reshape(n, nc)
        if a.get('normalized') and a['componentType'] in NORM:
            out = out.astype(np.float32) / NORM[a['componentType']]
        return out

    def image(self, i):
        """Texture i as HxWx4 float 0..1 (None if it cannot be read)."""
        if i in self.images:
            return self.images[i]
        try:
            tex = self.j['textures'][i]
            im = self.j['images'][tex['source']]
            if 'bufferView' in im:
                data = self.view(im['bufferView'])[0]
            elif im['uri'].startswith('data:'):
                data = base64.b64decode(im['uri'].split(',', 1)[1])
            else:
                with open(os.path.join(self.dir, im['uri']), 'rb') as f:
                    data = f.read()
            pic = Image.open(io.BytesIO(data)).convert('RGBA')
            if max(pic.size) > 1024:                       # enough for 4x the game's texels
                pic.thumbnail((1024, 1024))
            out = np.asarray(pic, np.float32) / 255.0
        except Exception:
            out = None
        self.images[i] = out
        return out

    def material(self, i):
        if i is None:
            return dict(name='(default)', rgb=(0.8, 0.8, 0.8), alpha=1.0, blend=False, tex=None)
        m = self.j['materials'][i]
        pbr = m.get('pbrMetallicRoughness', {})
        f = pbr.get('baseColorFactor', [1, 1, 1, 1])
        tex = None
        if 'baseColorTexture' in pbr:
            tex = self.image(pbr['baseColorTexture']['index'])
        ext = m.get('extensions', {})
        transmission = ext.get('KHR_materials_transmission', {}).get('transmissionFactor', 0.0)
        return dict(name=m.get('name', 'material %d' % i), rgb=tuple(f[:3]), alpha=float(f[3]),
                    blend=m.get('alphaMode') == 'BLEND' or transmission > 0.5,
                    metal=float(pbr.get('metallicFactor', 1.0)), tex=tex,
                    emissive=tuple(m.get('emissiveFactor', [0, 0, 0])))


def load(path):
    """[dict(name, material, pos Nx3, nrm Nx3, uv Nx2 or None, tri Mx3)] in the file's world
    space (metres if the file says so), and the asset's extras (author, licence...)."""
    g = Gltf(path)
    j = g.j
    scene = j.get('scene', 0)
    roots = j['scenes'][scene]['nodes'] if j.get('scenes') else list(range(len(j['nodes'])))
    mats = {}
    out = []

    def walk(ni, parent, path):
        node = j['nodes'][ni]
        W = parent @ _local(node)
        name = node.get('name', 'node %d' % ni)
        p = path + '/' + name
        if 'mesh' in node:
            mesh = j['meshes'][node['mesh']]
            for k, prim in enumerate(mesh['primitives']):
                if prim.get('mode', 4) != 4:
                    continue
                at = prim['attributes']
                pos = g.accessor(at['POSITION']).astype(np.float64)
                n = len(pos)
                tri = (g.accessor(prim['indices']).astype(np.int64).reshape(-1, 3) if 'indices' in prim
                       else np.arange(n - n % 3).reshape(-1, 3))
                nrm = g.accessor(at['NORMAL']).astype(np.float64) if 'NORMAL' in at else None
                uv = g.accessor(at['TEXCOORD_0']).astype(np.float64) if 'TEXCOORD_0' in at else None
                wp = pos @ W[:3, :3].T + W[:3, 3]
                if np.linalg.det(W[:3, :3]) < 0:
                    tri = tri[:, ::-1]
                if nrm is None:
                    nrm = _vertex_normals(wp, tri)
                else:
                    N = np.linalg.inv(W[:3, :3]).T
                    nrm = nrm @ N.T
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                mi = prim.get('material')
                if mi not in mats:
                    mats[mi] = g.material(mi)
                out.append(dict(name='%s#%d' % (name, k), path=p, material=mats[mi], pos=wp, nrm=nrm,
                                uv=uv, tri=tri))
        for c in node.get('children', []):
            walk(c, W, p)

    for r in roots:
        walk(r, np.eye(4), '')
    return out, j.get('asset', {}).get('extras', {})


def _vertex_normals(pos, tri):
    n = np.zeros_like(pos)
    fn = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
