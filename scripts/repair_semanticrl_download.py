"""Install SemanticRL's default branch without overwriting user changes."""
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

from scripts.download_reading_list import OUT, fetch

root = Path("pdf/GithubModels/SemanticRL").resolve()
api = "https://api.github.com/repos/lukun199/SemanticRL"
metadata = json.loads(fetch(api, 1_000_000)[0])
branch = metadata["default_branch"]
archive_url = f"https://codeload.github.com/lukun199/SemanticRL/zip/refs/heads/{branch}"
data, _ = fetch(archive_url)
old_main, _ = fetch("https://raw.githubusercontent.com/lukun199/SemanticRL/main/README.md")
previous = root / "README.md"
if previous.exists() and previous.read_bytes() != old_main:
    raise RuntimeError("Existing README differs from main branch; refusing to overwrite.")
with zipfile.ZipFile(io.BytesIO(data)) as z:
    pending = []
    for item in z.infolist():
        parts = PurePosixPath(item.filename).parts
        if len(parts) < 2 or item.is_dir():
            continue
        if any(p in ("..", ".git") for p in parts):
            raise ValueError("Unsafe archive path")
        dest = root.joinpath(*parts[1:])
        if not dest.resolve().is_relative_to(root):
            raise ValueError("Unsafe archive path")
        if dest.exists() and dest != previous:
            raise FileExistsError(f"Refusing to overwrite {dest}")
        pending.append((item, dest))
    for item, dest in pending:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(item))
manifest = json.loads(OUT.read_text(encoding="utf-8"))
record = next(x for x in manifest["repositories"] if x["repository"] == "https://github.com/lukun199/SemanticRL")
record.update(branch=branch, source=archive_url, zip_bytes=len(data),
              note="Default branch source archive; no Git history. Corrects earlier main-branch download.")
OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Installed {len(pending)} files from default branch {branch}; archive {len(data)} bytes")
