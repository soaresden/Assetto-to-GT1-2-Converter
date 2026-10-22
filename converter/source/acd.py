"""Assetto Corsa data.acd: the car's physics files (car.ini, engine.ini, power.lut, drivetrain.ini,
tyres.ini, suspensions.ini...), packed and lightly scrambled with a key made from the name of the
car's folder (the same scheme Content Manager and AcTools read).

Layout: entries one after the other, each
  i32 name length, name (ASCII), i32 data length n, n x i32: data byte + key[i % len(key)]
where key is the text "k0-k1-...-k7" made from the folder name (create_key).

read(path, folder=None) -> {file name: bytes}; folder defaults to the name of the folder holding
data.acd (the car's folder in content/cars).
"""
import os
import struct


def _i32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def _div(a, b):
    """C# integer division (truncates towards zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _mod(a, b):
    """C# remainder (sign of the dividend)."""
    return a - _div(a, b) * b


def create_key(folder):
    c = [ord(ch) for ch in folder.lower()]
    n = len(c)
    k = [0] * 8
    v = 0
    for x in c:
        v = _i32(v + x)
    k[0] = v
    v = 0
    for i in range(0, n - 1, 2):
        v = _i32(v * c[i])
        v = _i32(v - c[i + 1])
    k[1] = v
    v = 0
    for i in range(1, n - 3, 3):
        v = _i32(v * c[i])
        v = _div(v, c[i + 1] + 27)
        v = _i32(v - 27 - c[i - 1])
    k[2] = v
    v = 5763
    for i in range(1, n):
        v = _i32(v - c[i])
    k[3] = v
    v = 66
    for i in range(1, n - 4, 4):
        v = _i32(_i32(_i32((c[i] + 15) * v) * (c[i - 1] + 15)) + 22)
    k[4] = v
    v = 101
    for i in range(0, n - 2, 2):
        v = _i32(v - c[i])
    k[5] = v
    v = 171
    for i in range(0, n - 2, 2):
        v = _mod(v, c[i])
    k[6] = v
    v = 171
    for i in range(0, n - 1):
        v = _div(v, c[i])
        v = _i32(v + c[i + 1])
    k[7] = v
    return '-'.join(str(x & 0xFF) for x in k)


def unpack(data, key):
    key = key.encode('ascii')
    out = {}
    p = 0
    if data[:4] == b'\x15\xfb\xff\xff':          # newer files: a small header first
        p = 8
    while p < len(data):
        n, = struct.unpack_from('<i', data, p)
        name = data[p + 4:p + 4 + n].decode('latin-1')
        p += 4 + n
        size, = struct.unpack_from('<i', data, p)
        p += 4
        vals = struct.unpack_from('<%di' % size, data, p)
        p += 4 * size
        out[name] = bytes((v - key[i % len(key)]) & 0xFF for i, v in enumerate(vals))
    return out


def read(path, folder=None):
    if folder is None:
        folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    with open(path, 'rb') as f:
        data = f.read()
    files = unpack(data, create_key(folder))
    if not any(v.lstrip().startswith(b'[') for k, v in files.items() if k.endswith('.ini')):
        raise ValueError('data.acd : la clé du dossier « %s » ne donne pas de .ini lisible '
                         '(il faut le nom d\'origine du dossier de la voiture)' % folder)
    return files
