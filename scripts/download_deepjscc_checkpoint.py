"""Download and safely extract the DeepJSCC author's pretrained release."""
import io
import tarfile
import urllib.request
from pathlib import Path

root = Path("pdf/GithubModels/DeepJSCC").resolve()
url = "https://github.com/vale817/DeepJSCC/releases/download/v0.1-cifar10-awgn/runs.tar.gz"
request = urllib.request.Request(url, headers={"User-Agent": "SwarmBridge research"})
with urllib.request.urlopen(request, timeout=60) as response:
    data = response.read(100_000_001)
if len(data) > 100_000_000:
    raise ValueError("Checkpoint archive exceeds 100 MB")
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
    members = archive.getmembers()
    if sum(x.size for x in members) > 500_000_000:
        raise ValueError("Extracted checkpoint archive exceeds 500 MB")
    for member in members:
        dest = root / member.name
        if not dest.resolve().is_relative_to(root) or member.issym() or member.islnk():
            raise ValueError("Unsafe archive member")
        if dest.exists() and member.isfile():
            raise FileExistsError(dest)
    archive.extractall(root, members=members, filter="data")
print(f"Extracted {len(members)} entries from {url}")
