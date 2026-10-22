"""Gran Turismo (PS1) data tables of CARINF.DAT ("@(#)SPEC", "@(#)COLOR"... one per entry of the
GT-ARC): fixed-size records, then string tables.

  0x00  "@(#)" + name, padded to 12 bytes
  0x0C  u16 0x10, u16 record count, u32 0, u32 record size
  0x18  records
        then, when there is text: u32 table count, one u32 per table, the tables: u16 string count,
        strings (u8 length, the characters, a zero), padded to 2 bytes
A record points to a string with (u16 index, u16 table).
"""
import struct


class Table:
    def __init__(self, data):
        self.head = bytes(data[:0x18])
        _, count, _, self.size = struct.unpack_from('<HHII', data, 0x0C)
        o = 0x18
        self.records = [bytearray(data[o + i * self.size:o + (i + 1) * self.size]) for i in range(count)]
        o += count * self.size
        self.strings = []
        self.extra = b''
        self.has_tables = o + 4 <= len(data)
        if self.has_tables:
            n, = struct.unpack_from('<I', data, o)
            self.table_words = list(struct.unpack_from('<%dI' % n, data, o + 4))
            o += 4 + 4 * n
            for _ in range(n):
                k, = struct.unpack_from('<H', data, o)
                o += 2
                t = []
                for _ in range(k):
                    ln = data[o]
                    t.append(bytes(data[o + 1:o + 1 + ln]))
                    o += ln + 2
                o += o & 1
                self.strings.append(t)
            self.extra = bytes(data[o:])
        else:
            self.table_words = []

    def string(self, index, table):
        return self.strings[table][index].decode('latin-1')

    def add_string(self, table, text):
        """Index of `text` in a string table (added at the end if missing)."""
        b = text.encode('latin-1')
        t = self.strings[table]
        if b not in t:
            t.append(b)
        return t.index(b)

    def write(self):
        h = bytearray(self.head)
        struct.pack_into('<H', h, 0x0E, len(self.records))
        out = h + b''.join(self.records)
        if self.has_tables:
            body = bytearray()
            sizes = []
            for t in self.strings:
                start = len(body)
                body += struct.pack('<H', len(t))
                for s in t:
                    body += bytes([len(s)]) + s + b'\0'
                if len(body) & 1:
                    body += b'\0'
                sizes.append(len(body) - start)
            words = self.table_words  # zeros on the disc
            out += struct.pack('<I', len(self.strings)) + struct.pack('<%dI' % len(words), *words) + body
        return bytes(out + self.extra)
