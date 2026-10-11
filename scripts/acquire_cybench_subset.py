"""Retrieve exact publication members from the pinned public CY-Bench ZIP."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

URL = 'https://zenodo.org/api/records/17279151/files/cybench-data.zip/content'
SIZE = 6228749411


def request_range(start: int, end: int):
    response = urllib.request.urlopen(urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'}), timeout=120)
    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
        response.close()
        raise ValueError('The source did not honour the exact pinned range')
    return response


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        if count < 0:
            count = SIZE - self.position
        if count == 0:
            return b''
        with request_range(self.position, self.position + count - 1) as response:
            data = response.read()
        if len(data) != count:
            raise ValueError('Incomplete ZIP metadata range')
        self.position += count
        return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--list-only', action='store_true')
    p.add_argument('--pair', choices=['maize_US', 'wheat_AU'], action='append')
    p.add_argument('--apsim', action='store_true', help='Also retrieve the original AU soil-moisture input.')
    a = p.parse_args()
    pairs = [value.split('_') for value in a.pair] if a.pair else [('maize', 'US'), ('wheat', 'AU')]
    wanted = {f'{prefix}_{crop}_{country}.csv' for crop, country in pairs for prefix in ('yield', 'meteo', 'crop_calendar', 'soil', 'location', 'crop_mask')}
    if a.apsim:
        wanted.add('soil_moisture_wheat_AU.csv')
    with zipfile.ZipFile(RemoteZip()) as archive:
        selected = [i for i in archive.infolist() if Path(i.filename).name in wanted and not i.filename.startswith('__MACOSX/')]
    if len(selected) != len(wanted) or {Path(i.filename).name for i in selected} != wanted:
        raise ValueError('Publication source members are not unique')
    selected.sort(key=lambda item: item.compress_size)
    inventory = [{'member': i.filename, 'compressed_bytes': i.compress_size, 'bytes': i.file_size, 'crc32': f'{i.CRC:08x}'} for i in selected]
    print(json.dumps(inventory, indent=2), flush=True)
    if a.list_only:
        return
    a.output_root.mkdir(parents=True, exist_ok=False)
    receipts = []
    for item in selected:
        if item.compress_type not in (0, 8):
            raise ValueError('Unsupported source compression')
        with request_range(item.header_offset, item.header_offset + 29) as response:
            header = response.read()
        fields = struct.unpack('<4s5H3I2H', header)
        if fields[0] != b'PK\x03\x04':
            raise ValueError('Invalid local ZIP header')
        start = item.header_offset + 30 + fields[-2] + fields[-1]
        path = a.output_root / Path(item.filename).name
        decoder = zlib.decompressobj(-15) if item.compress_type == 8 else None
        crc, count = 0, 0
        digest = hashlib.sha256()
        with request_range(start, start + item.compress_size - 1) as response, path.open('xb') as output:
            while chunk := response.read(1024 * 1024):
                data = decoder.decompress(chunk) if decoder else chunk
                output.write(data)
                crc = zlib.crc32(data, crc)
                count += len(data)
                digest.update(data)
            if decoder:
                data = decoder.flush()
                output.write(data)
                crc = zlib.crc32(data, crc)
                count += len(data)
                digest.update(data)
        if count != item.file_size or crc != item.CRC:
            raise ValueError(f'Source member integrity failed: {item.filename}')
        receipts.append({'member': item.filename, 'local': path.name, 'bytes': count, 'crc32': f'{crc:08x}', 'sha256': digest.hexdigest()})
        (a.output_root / 'acquisition_manifest.json').write_text(json.dumps({'doi': '10.5281/zenodo.17279151', 'source': URL, 'archive_bytes': SIZE, 'verification': 'ZIP member CRC and SHA256; full archive MD5 not checked in range mode', 'members': receipts}, indent=2) + '\n')
        print('Acquired and CRC verified:', path.name, count, flush=True)


if __name__ == '__main__':
    main()
