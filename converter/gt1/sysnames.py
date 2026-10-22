"""The short car names of SYSTEM.DAT ("@(#)GTENV"), shown by the race part of the game (settings,
race screens, replays): u8 length (with the final 0), the text, 0, one after the other. From item
126 on, one per CAR.DAT entry pair in the same order (126: "GRAN TURISMO LOGO" = entries 0/1),
race versions starting with 0x7F: the name of CAR.DAT entry `tex` (day texture) is item
126 + tex / 2 (checked: Lantis 64 -> "LANTIS TypeR", Roadster NORMAL 70 -> "ROADSTER", Legacy
Touring Wagon 532 -> "LEGACY WG GT-B ").

The names are packed: a new name keeps the old one's length (cut, or padded with spaces), so that
nothing else moves in the file.
"""
import re

FIRST = 126
ITEM = re.compile(rb'([\x03-\x20])(\x7f?[ -~]{2,30})\x00')


def items(system_dat):
    """[(offset of the text, text bytes)] of the length-prefixed strings, in order."""
    out = []
    p = 0
    b = system_dat
    for m in ITEM.finditer(b):
        if m.group(1)[0] == len(m.group(2)) + 1:
            out.append((m.start() + 1, m.group(2)))
    return out


def name_of(system_dat, tex):
    return items(system_dat)[FIRST + tex // 2][1].decode('latin-1')


def rename(system_dat, tex, text, race=True):
    """SYSTEM.DAT with the short name of CAR.DAT entry `tex` (and of the race version that follows
    it, 0x7F + name, if race) replaced by `text` at the same length."""
    b = bytearray(system_dat)
    it = items(system_dat)
    k = FIRST + tex // 2
    targets = [k]
    if race and k + 1 < len(it) and it[k + 1][1][:1] == b'\x7f':
        targets.append(k + 1)
    for t in targets:
        off, old = it[t]
        prefix = b'\x7f' if old[:1] == b'\x7f' else b''
        new = prefix + text.encode('latin-1', 'replace')
        new = new[:len(old)].ljust(len(old), b' ')
        b[off:off + len(old)] = new
    return bytes(b)
