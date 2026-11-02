"""Download official speech test assets, preserving download provenance."""
import concurrent.futures
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'runs/sb_paper_reproductions'


def download(item):
    name, url = item
    target = OUT / 'downloads' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not target.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 research reproduction'})
            temporary = target.with_suffix('.partial')
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
                final_url = response.geturl()
                length = response.headers.get('Content-Length')
                print(name, 'downloading', length, flush=True)
                while data := response.read(2**20):
                    stream.write(data)
            temporary.replace(target)
        record = dict(name=name, url=url, path=str(target.relative_to(ROOT)), bytes=target.stat().st_size)
        record['sha256'] = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
        print(json.dumps(record), flush=True)
        return record
    except Exception as error:
        print(name, type(error).__name__, str(error), flush=True)
        return dict(name=name, url=url, error=str(error))


if __name__ == '__main__':
    tasks = [
        ('sony_sbctm.ckpt', 'https://osf.io/download/f293c/?view_only=e406b105dd274657b7b33cea9dc764af'),
        ('clean_testset_wav.zip', 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip'),
        ('noisy_testset_wav.zip', 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip'),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, tasks))
    for label in ('clean', 'noisy'):
        path = OUT / 'downloads' / f'{label}_testset_wav.zip'
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as archive:
            selected = sorted(n for n in archive.namelist() if n.endswith('.wav'))[:12]
            target = ROOT / 'data/sb_speech_test' / label
            target.mkdir(parents=True, exist_ok=True)
            for name in selected:
                (target / Path(name).name).write_bytes(archive.read(name))
            print('extracted', label, len(selected), flush=True)
    (OUT / 'test_assets.json').write_text(json.dumps(records, indent=2))
