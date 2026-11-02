"""Fetch official SB research sources/checkpoints with local provenance."""
import concurrent.futures
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'runs/sb_paper_reproductions'
REPOS = ROOT / 'pdf/GithubModels'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'SwarmBridge research reproduction'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read(), response.geturl()


def repository(owner_name):
    metadata = json.loads(fetch('https://api.github.com/repos/' + owner_name)[0])
    branch = metadata['default_branch']
    tree = json.loads(fetch(f'https://api.github.com/repos/{owner_name}/git/trees/{branch}?recursive=1')[0])
    sha = tree['sha']
    name = owner_name.replace('/', '__')
    target = REPOS / name
    target.mkdir(parents=True, exist_ok=True)
    # Download only the source, not large demo/audio/image collections.
    entries = [item for item in tree['tree'] if item['type'] == 'blob' and
               (item['path'].endswith(('.py', '.yaml', '.yml', '.json', '.txt', '.md', '.toml', '.cfg', '.sh'))
                or Path(item['path']).name.startswith('LICENSE'))]
    def save(item):
        path = target / item['path']
        if not path.resolve().is_relative_to(target.resolve()):
            raise ValueError(f'Unsafe repository path: {item["path"]!r}; resolved={path.resolve()}; root={target.resolve()}')
        path.parent.mkdir(parents=True, exist_ok=True)
        content = fetch(f'https://raw.githubusercontent.com/{owner_name}/{sha}/{item["path"]}')[0]
        path.write_bytes(content)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(save, entries))
    info = dict(repository=owner_name, branch=branch, commit=sha, source_files=len(entries),
                python_files=sum(p['path'].endswith('.py') for p in entries), path=str(target.relative_to(ROOT)))
    (target / 'SWARMBRIDGE_SOURCE.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    return info


def checkpoint(repo):
    metadata = json.loads(fetch('https://huggingface.co/api/models/' + repo)[0])
    files = [p['rfilename'] for p in metadata['siblings'] if p['rfilename'].endswith('.nemo')]
    records = []
    for filename in files:
        target = OUT / 'checkpoints' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{repo}/resolve/{metadata["sha"]}/{filename}'
        if not target.exists():
            content, final = fetch(url)
            target.write_bytes(content)
        records.append(dict(url=url, path=str(target.relative_to(ROOT)), size=target.stat().st_size,
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print(json.dumps(records[-1]), flush=True)
    return records


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for repo in ['sp-uhh/sgmse', 'sony/sbctm', 'NVIDIA/diffusion-audio-restoration',
                 'Taltt/FNSE-SBGAN', 'thuhcsi/interspeech2026-BridgeCodec', 'sbmse/sbmse.github.io']:
        try:
            records.append(repository(repo))
        except Exception as error:
            records.append(dict(repository=repo, error=str(error)))
            print(repo, type(error).__name__, str(error), flush=True)
    try:
        records.extend(checkpoint('nvidia/se_den_sb_16k_small'))
    except Exception as error:
        records.append(dict(checkpoint='nvidia/se_den_sb_16k_small', error=str(error)))
        print(type(error).__name__, str(error), flush=True)
    (OUT / 'assets.json').write_text(json.dumps(records, indent=2))
