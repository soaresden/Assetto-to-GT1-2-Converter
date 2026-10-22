"""A raw CD image (.bin of 2352-byte sectors, or the .cue that names it) read like a CHD: same
methods as chd.Chd (frames, sector, user_data, read), so the tools take either.

open_image(path or file object) picks the right reader: a .cue, a CHD (by its magic) or a .bin.
Only single data track images are handled (the PlayStation games used here have one track).
"""
import os
import re

import chd

SECTOR = 2352
SYNC = b'\x00' + b'\xff' * 10 + b'\x00'


class Bin:
    def __init__(self, f):
        self.f = open(f, 'rb') if isinstance(f, str) else f
        self.f.seek(0, os.SEEK_END)
        size = self.f.tell()
        if size % SECTOR:
            raise ValueError('not a raw CD image: %d bytes is not a whole number of 2352-byte sectors' % size)
        self.f.seek(0)
        if self.f.read(12) != SYNC:
            raise ValueError('not a raw CD image (mode 1/2, 2352 bytes per sector)')
        self._frames = size // SECTOR

    @property
    def frames(self):
        return self._frames

    def sector(self, lba):
        self.f.seek(lba * SECTOR)
        return self.f.read(SECTOR)

    def user_data(self, lba):
        s = self.sector(lba)
        if s[15] == 2:
            if s[18] & 0x20:  # mode 2 form 2: 2324 bytes
                return s[24:24 + 2324]
            return s[24:24 + 2048]
        return s[16:16 + 2048]

    def read(self, lba, count):
        self.f.seek(lba * SECTOR)
        raw = self.f.read(count * SECTOR)
        return b''.join(self._user(raw[i * SECTOR:(i + 1) * SECTOR]) for i in range(count))

    @staticmethod
    def _user(s):
        if s[15] == 2:
            return s[24:24 + (2324 if s[18] & 0x20 else 2048)]
        return s[16:16 + 2048]


def cue_bin(cue_path):
    """The .bin a .cue names (its first FILE); ValueError if the image has several tracks."""
    text = open(cue_path, 'rb').read().decode('utf-8', 'replace')
    files = re.findall(r'^\s*FILE\s+"?([^"\r\n]+?)"?\s+BINARY\s*$', text, re.M | re.I)
    tracks = re.findall(r'^\s*TRACK\s+\d+\s+(\S+)', text, re.M | re.I)
    if not files:
        raise ValueError('%s: no FILE line' % cue_path)
    if len(tracks) > 1:
        raise ValueError('%s: %d tracks, only single-track images are supported' % (cue_path, len(tracks)))
    path = os.path.join(os.path.dirname(os.path.abspath(cue_path)), files[0])
    if not os.path.exists(path):
        raise ValueError('%s: the file it names (%s) is not next to it' % (cue_path, files[0]))
    return path


def open_image(f):
    """chd.Chd or Bin for a path (.chd, .cue, .bin) or a binary file object."""
    if isinstance(f, str) and f.lower().endswith('.cue'):
        f = cue_bin(f)
    fo = open(f, 'rb') if isinstance(f, str) else f
    fo.seek(0)
    magic = fo.read(12)
    fo.seek(0)
    if magic[:8] == b'MComprHD':
        return chd.Chd(fo)
    if magic == SYNC:
        return Bin(fo)
    raise ValueError('neither a CHD nor a raw .bin CD image')
