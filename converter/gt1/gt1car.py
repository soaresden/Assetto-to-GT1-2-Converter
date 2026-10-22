"""Gran Turismo (PS1) car models, "@(#)GT-CAR" (the .car entries of CAR.DAT).

Layout (little-endian; field meanings as documented by pez2k's GT2ModelTool, checked here by
re-writing every model of the disc byte for byte):

  0x00  "@(#)GT-CAR" + 6 bytes
  0x10  4 wheel positions: s16 x, y, z, w (front left, front right, rear left, rear right)
  0x30  u16 menu front wheel radius, width, rear radius, width
  0x38  4 bytes, u16 LOD count, 0x42 bytes                                 -> 0x80
  LODs (40 bytes between two LODs), each:
        u16 vertices, normals, tris, quads, 2 x u16, textured tris, textured quads, 20 bytes,
        u16 scale, 2 bytes                                                 (0x28 bytes)
        vertices: 4 x s16 (x, y, z, w)    normals: 4 x s16 (x, y, z, 0), unit = 4000
        tris, quads: 16 bytes; textured tris, quads: 28 bytes
  shadow: u16, u16 quad count, u16 scale, u16, 8 x s16 bounds, 8 bytes, 4 x quads vertices of 8 bytes

A face (16 bytes) is a 96-bit little-endian field: vertex indices at bits 0, 9, 18 and 32,
normal indices at 41, 51, 64 and 74 (9 bits each); bits 27-31 = 16 + a 2-bit value (bit 28 set:
lit, the normals are used; clear: normal 0), bit 63 = drawing order flag. Then 3 bytes (RGB of an
untextured face, FF on textured faces) and the face type (0x21 tri, 0x29 quad, 0x25 textured tri,
0x2D textured quad). Quads go round their corners 0-1-2-3. A textured face adds: uv0 (u8 u, u8 v),
u16 palette word (CLUT (w >> 6) * 4 + (w & 3); 14 = brake lights), uv1, 2 bytes (zero but on a few
cars), uv2, uv3.
Texture u, v are texel coordinates of the 256x256 4-bit texture (v from 32 on the disc's cars).
LOD header: counts, then at 0x14 the bounding box (s16 min x y z w, max x y z w), 0x24 scale,
0x26 bounding sphere radius.

Coordinates: 1/4096 m scaled by 2^(scale - 16) (scale 16 = 1/4096 m); y is up, z points to the
front (the tools negate it for their GT2 conventions; here they are kept as stored).
"""
import struct

MAGIC = b'@(#)GT-CAR'
V_BITS = (0, 9, 18, 32)
N_BITS = (41, 51, 64, 74)


def _mask(bits):
    m = 0
    for b in bits:
        m |= 0x1FF << b
    return m


IDX_MASK = _mask(V_BITS + N_BITS)


class Face:
    __slots__ = ('v', 'n', 'rest', 'tail', 'uv', 'pal', 'pad', 'quad', 'textured')

    @classmethod
    def read(cls, b, quad, textured):
        f = cls()
        x = int.from_bytes(b[:12], 'little')
        f.v = [x >> s & 0x1FF for s in V_BITS]
        f.n = [x >> s & 0x1FF for s in N_BITS]
        f.rest = x & ~IDX_MASK
        f.tail = bytes(b[12:16])
        f.quad, f.textured = quad, textured
        f.uv = f.pal = None
        f.pad = b'\0\0'
        if textured:
            t = b[16:28]
            f.uv = [(t[0], t[1]), (t[4], t[5]), (t[8], t[9]), (t[10], t[11])]
            f.pal, = struct.unpack_from('<H', t, 2)
            f.pad = bytes(t[6:8])  # zero but on a few cars
        return f

    def write(self):
        x = self.rest
        for s, i in zip(V_BITS, self.v):
            x |= (i & 0x1FF) << s
        for s, i in zip(N_BITS, self.n):
            x |= (i & 0x1FF) << s
        b = x.to_bytes(12, 'little') + self.tail
        if self.textured:
            uv = self.uv
            b += bytes(uv[0]) + struct.pack('<H', self.pal) + bytes(uv[1]) + self.pad + bytes(uv[2]) + bytes(uv[3])
        return b

    def tris(self):
        a, b, c, d = self.v
        return [(a, b, c), (a, c, d)] if self.quad else [(a, b, c)]  # quads go round 0-1-2-3


class Lod:
    def __init__(self):
        self.head = b''
        self.scale = 16
        self.verts = []    # (x, y, z, w)
        self.normals = []  # (x, y, z, w)
        self.faces = []    # tris, quads, textured tris, textured quads, in that order

    @classmethod
    def read(cls, d, o):
        L = cls()
        nv, nn, nt, nq, u0, u1, ntt, ntq = struct.unpack_from('<8H', d, o)
        if u0 or u1:  # two more kinds of faces, only on 10 models of the disc (not decoded)
            raise NotImplementedError('LOD with faces of an unknown kind')
        L.head = bytes(d[o:o + 0x28])
        L.scale, = struct.unpack_from('<H', d, o + 0x24)
        o += 0x28
        L.verts = [struct.unpack_from('<4h', d, o + 8 * i) for i in range(nv)]
        o += 8 * nv
        L.normals = [struct.unpack_from('<4h', d, o + 8 * i) for i in range(nn)]
        o += 8 * nn
        for count, quad, tex in ((nt, False, False), (nq, True, False), (ntt, False, True), (ntq, True, True)):
            size = 28 if tex else 16
            for i in range(count):
                L.faces.append(Face.read(d[o:o + size], quad, tex))
                o += size
        return L, o

    def write(self):
        kinds = [(False, False), (True, False), (False, True), (True, True)]
        counts = [sum(1 for f in self.faces if (f.quad, f.textured) == k) for k in kinds]
        h = bytearray(self.head)
        struct.pack_into('<4H', h, 0, len(self.verts), len(self.normals), counts[0], counts[1])
        struct.pack_into('<2H', h, 12, counts[2], counts[3])
        struct.pack_into('<H', h, 0x24, self.scale)
        out = bytearray(h)
        for v in self.verts:
            out += struct.pack('<4h', *v)
        for n in self.normals:
            out += struct.pack('<4h', *n)
        for k in kinds:
            for f in self.faces:
                if (f.quad, f.textured) == k:
                    out += f.write()
        return bytes(out)

    def unit(self):
        """metres per stored unit"""
        return 2.0 ** (self.scale - 16) / 4096


class Car:
    @classmethod
    def read(cls, d):
        if d[:10] != MAGIC:
            raise ValueError('not a GT-CAR')
        c = cls()
        c.head = bytes(d[:0x80])
        c.wheels = [struct.unpack_from('<4h', d, 0x10 + 8 * i) for i in range(4)]
        c.menu_wheels = struct.unpack_from('<4H', d, 0x30)
        n, = struct.unpack_from('<H', d, 0x3C)
        o = 0x80
        c.lods = []
        c.gaps = []
        for i in range(n):
            L, o = Lod.read(d, o)
            c.lods.append(L)
            if i != n - 1:
                c.gaps.append(bytes(d[o:o + 40]))
                o += 40
        c.shadow_head = bytes(d[o:o + 0x20])
        nq, = struct.unpack_from('<H', d, o + 2)
        if o + 0x20 + 32 * nq > len(d):  # a few cars (unused ones) end short: kept raw
            c.shadow_head, c.shadow, c.trailing = None, None, bytes(d[o:])
            return c
        o += 0x20
        c.shadow = [struct.unpack_from('<4h', d, o + 8 * i) for i in range(4 * nq)]
        o += 32 * nq
        c.trailing = bytes(d[o:])
        return c

    def write(self):
        h = bytearray(self.head)
        for i, w in enumerate(self.wheels):
            struct.pack_into('<4h', h, 0x10 + 8 * i, *w)
        struct.pack_into('<4H', h, 0x30, *self.menu_wheels)
        struct.pack_into('<H', h, 0x3C, len(self.lods))
        out = bytearray(h)
        for i, L in enumerate(self.lods):
            out += L.write()
            if i != len(self.lods) - 1:
                out += self.gaps[i] if i < len(self.gaps) else bytes(40)
        if self.shadow is None:
            return bytes(out + self.trailing)
        sh = bytearray(self.shadow_head)
        struct.pack_into('<H', sh, 2, len(self.shadow) // 4)
        out += sh
        for v in self.shadow:
            out += struct.pack('<4h', *v)
        out += self.trailing
        return bytes(out)
