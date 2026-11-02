"""Recover complete ZIP entries from a downloaded prefix, validating each CRC.

The Edinburgh server is slow. No complete-dataset download is claimed here.
"""
import binascii
import json
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def entries(raw):
    offset = 0
    while raw[offset:offset + 4] == b'PK\x03\x04':
        fields = struct.unpack_from('<IHHHHHIIIHH', raw, offset)
        _, version, flags, method, mtime, mdate, crc, compressed, uncompressed, nlen, elen = fields
        name = raw[offset + 30:offset + 30 + nlen].decode()
        start = offset + 30 + nlen + elen
        if method == 0:
            data = raw[start:start + compressed]
            offset = start + compressed
        elif method == 8:
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(raw[start:])
            if not decoder.eof:
                break  # the last member is incomplete
            offset = len(raw) - len(decoder.unused_data)
        else:
            raise ValueError(method)
        if flags & 8:
            if raw[offset:offset + 4] == b'PK\x07\x08':
                offset += 4
            if len(raw) < offset + 12:
                break
            crc, compressed, uncompressed = struct.unpack_from('<III', raw, offset)
            offset += 12
        if binascii.crc32(data) & 0xffffffff != crc:
            raise ValueError('CRC mismatch: ' + name)
        if name.endswith('.wav'):
            yield name, data


if __name__ == '__main__':
    records = {}
    for label in ('clean', 'noisy'):
        path = ROOT / 'runs/sb_paper_reproductions/downloads' / f'{label}_testset_wav.partial'
        target = ROOT / 'data/sb_speech_test' / label
        target.mkdir(parents=True, exist_ok=True)
        found = list(entries(path.read_bytes()))[:12]
        for name, data in found:
            (target / Path(name).name).write_bytes(data)
        records[label] = [Path(name).name for name, _ in found]
    print(json.dumps(records, indent=2))
    (ROOT / 'runs/sb_paper_reproductions/test_subset.json').write_text(json.dumps(records, indent=2))
