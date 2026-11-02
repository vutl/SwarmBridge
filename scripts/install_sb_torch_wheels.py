"""Install the three official CUDA wheels without uv's failing Windows cache rename."""
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / '.venv-sb-papers/Lib/site-packages'
OUT = ROOT / 'runs/sb_paper_reproductions/downloads'

if __name__ == '__main__':
    records = []
    for package, version in [('torch', '2.5.1'), ('torchaudio', '2.5.1'), ('torchvision', '0.20.1')]:
        index = f'https://download.pytorch.org/whl/cu124/{package}/'
        listing = urllib.request.urlopen(index, timeout=90).read().decode()
        candidates = [html.unescape(url) for url in re.findall(r'href="([^"]+)"', listing)]
        needle = f'{package}-{version}+cu124-cp312-cp312-win_amd64.whl'
        url = next(u for u in candidates if needle in urllib.parse.unquote(u))
        url = urllib.parse.urljoin(index, url)
        expected = urllib.parse.parse_qs(urllib.parse.urlparse(url).fragment)['sha256'][0]
        path = OUT / needle
        if not path.exists():
            print('Downloading', needle, flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as stream:
                while data := response.read(4 * 2**20):
                    stream.write(data)
        actual = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Wheel checksum mismatch: {path}')
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                target = TARGET / name
                if not target.resolve().is_relative_to(TARGET.resolve()):
                    raise ValueError(f'Unsafe wheel member: {name}')
                if name.endswith('/'):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as source, target.open('wb') as stream:
                        while data := source.read(4 * 2**20):
                            stream.write(data)
        records.append(dict(package=package, version=version, url=url, sha256=actual))
        print('Installed', package, version, flush=True)
    (OUT.parent / 'torch_wheels.json').write_text(json.dumps(records, indent=2))
