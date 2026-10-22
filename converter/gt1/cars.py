"""The cars of Gran Turismo (PS1, PAL SCES-00984) that a model can replace: each code of CARINF.DAT
(SPEC table) and the CAR.DAT entries of its texture and model.

CAR.DAT has no names: its 688 day entries (texture, model; the night ones follow, + 688) are in
about the alphabetical order of the car codes, race versions included, with a few codes of its own.
So each code is matched with the model whose length and wheelbase are those of its SPEC record
(within 30 mm and 20 mm: the disc's models are made to their cars' sizes), the nearest to where
the alphabetical order puts it. Checked on the disc: aminn 70, a-ian 8, altsn 64, slgan 532.
"""
import struct

import numpy as np

import gt1car
import gt1data
import gtarc

SPEC_ENTRY = 13
NIGHT = 688


def _model_sizes(car_dat):
    out = {}
    for k in range(0, NIGHT, 2):
        try:
            c = gt1car.Car.read(car_dat[k + 1])
        except Exception:
            continue
        L = c.lods[0]
        v = np.array([x[:3] for x in L.verts], float) * 2.0 ** (L.scale - 16) / 4096
        out[k] = (v[:, 2].max() - v[:, 2].min(), (c.wheels[0][2] - c.wheels[2][2]) / 4096)
    return out


def catalogue(carinf, car_dat):
    """{code: dict(tex=index, model=index + 1, name, trim, length, width, height, wheelbase, price,
    power)}: the street cars (codes ending in n) of CARINF.DAT found in CAR.DAT."""
    arc = car_dat if isinstance(car_dat, gtarc.Arc) else gtarc.Arc(car_dat)
    spec = gt1data.Table(gtarc.Arc(carinf)[SPEC_ENTRY])
    recs = []
    for r in spec.records:
        code = bytes(r[:6]).rstrip(b'\0').decode()
        a, ta, b, tb = struct.unpack_from('<4H', r, 0x188)
        w, h, wb = struct.unpack_from('<3H', r, 0x0C)
        L, cc, ps = struct.unpack_from('<3H', r, 0x194)
        recs.append(dict(code=code, name=spec.string(a, ta), trim=spec.string(b, tb), length=L, width=w,
                         height=h, wheelbase=wb, price=struct.unpack_from('<I', r, 0x184)[0] // 100, power=ps,
                         cc=cc))
    sizes = _model_sizes(arc)
    ks = sorted(sizes)
    recs.sort(key=lambda r: r['code'])

    def fits(r, k):
        L, wb = sizes[k]
        return abs(L * 1000 - r['length']) < 30 and abs(wb * 1000 - r['wheelbase']) < 20

    # ordered alignment (the entries follow the codes' order; models of CAR.DAT's own, such as
    # race versions, sit between them): the most codes matched, each with a model of its size,
    # at increasing entries; between equal counts, the earliest entries (a street car comes
    # before its race version)
    n, m = len(recs), len(ks)
    best = np.zeros((n + 1, m + 1), np.int32)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            b = max(best[i + 1, j], best[i, j + 1])
            if fits(recs[i], ks[j]):
                b = max(b, 1 + best[i + 1, j + 1])
            best[i, j] = b
    out = {}
    i = j = 0
    while i < n and j < m:
        if fits(recs[i], ks[j]) and best[i, j] == 1 + best[i + 1, j + 1]:
            r = recs[i]
            if r['code'].endswith('n'):
                out[r['code']] = dict(r, tex=ks[j], model=ks[j] + 1)
            i, j = i + 1, j + 1
        elif best[i, j + 1] >= best[i + 1, j]:
            j += 1
        else:
            i += 1
    return out
