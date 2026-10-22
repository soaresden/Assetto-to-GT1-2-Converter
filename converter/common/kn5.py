"""Minimal KN5 (Assetto Corsa) reader. Encrypted files (Custom Shaders Patch protection) are
refused with kn5.Encrypted: their geometry and textures cannot be read, and must not be."""
import struct


class R:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def take(self, n):
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def i32(self):
        return struct.unpack_from('<i', self.take(4))[0]

    def u32(self):
        return struct.unpack_from('<I', self.take(4))[0]

    def f32(self, n=1):
        v = struct.unpack_from('<%df' % n, self.take(4 * n))
        return v[0] if n == 1 else v

    def byte(self):
        return self.take(1)[0]

    def s(self):
        n = self.i32()
        return self.take(n).decode('utf-8', 'replace')


class Encrypted(ValueError):
    pass


def read(path):
    with open(path, 'rb') as fh:
        r = R(fh.read())
    if r.take(6) != b'sc6969':
        raise ValueError('%s : pas un fichier .kn5' % path)
    if b'__AC_SHADERS_PATCH_KN5ENC' in r.d:
        raise Encrypted('%s : modèle chiffré (protection du Custom Shaders Patch) ; il ne peut pas '
                        'être converti. Demander à son auteur une version non chiffrée.' % path)
    version = r.i32()
    if version > 5:
        r.i32()
    textures = {}
    for _ in range(r.i32()):
        r.i32()
        name = r.s()
        size = r.u32()
        textures[name] = r.take(size)
    materials = []
    for _ in range(r.i32()):
        m = {'name': r.s(), 'shader': r.s()}
        m['blend'] = r.byte()
        m['alphaTested'] = r.byte()
        m['depth'] = r.i32()
        props = {}
        for _ in range(r.i32()):
            pn = r.s()
            a = r.f32()
            r.f32(9)
            props[pn] = a
        m['props'] = props
        tex = {}
        for _ in range(r.i32()):
            sn = r.s()
            r.i32()
            tex[sn] = r.s()
        m['tex'] = tex
        materials.append(m)

    def node():
        cls = r.i32()
        n = {'cls': cls, 'name': r.s()}
        nchild = r.i32()
        n['active'] = r.byte()
        if cls == 1:
            n['matrix'] = r.f32(16)
        elif cls == 2:
            n['shadows'], n['visible'], n['transparent'] = r.byte(), r.byte(), r.byte()
            vc = r.u32()
            n['verts'] = r.take(vc * 44)
            n['vc'] = vc
            ic = r.u32()
            n['idx'] = r.take(ic * 2)
            n['ic'] = ic
            n['mat'] = r.u32()
            n['layer'] = r.u32()
            n['lod'] = r.f32(2)
            r.f32(4)
            n['renderable'] = r.byte()
        elif cls == 3:
            n['shadows'], n['visible'], n['transparent'] = r.byte(), r.byte(), r.byte()
            bones = []
            for _ in range(r.u32()):
                bones.append((r.s(), r.f32(16)))
            n['bones'] = bones
            vc = r.u32()
            raw = r.take(vc * 76)
            # keep pos/normal/uv/tangent (44 bytes) of each 76-byte vertex
            n['verts'] = b''.join(raw[i * 76:i * 76 + 44] for i in range(vc))
            n['vc'] = vc
            ic = r.u32()
            n['idx'] = r.take(ic * 2)
            n['ic'] = ic
            n['mat'] = r.u32()
            n['layer'] = r.u32()
            r.take(8)
            n['renderable'] = 1
        else:
            raise ValueError('unknown node class %d at %d' % (cls, r.p))
        n['children'] = [node() for _ in range(nchild)]
        return n

    root = node()
    # after the tree, some files have extra blocks (Custom Shaders Patch data): not needed here
    return textures, materials, root
