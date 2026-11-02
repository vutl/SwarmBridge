"""Split and restore the project's large data files without changing their contents.

Run ``python scripts/reassemble_large_files.py restore`` after cloning.
Splitting is needed only when preparing the original upload.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "large_file_manifest.json"
SOURCES = (
    "data/coperception_uav/raw/coperception_uav_town3.zip",
    "data/cifar10/cifar-10-python.tar.gz",
    "runs/teacher/latents_train.pt",
    "runs/teacher_v1_flatground/latents_train.pt",
)
BLOCK = 1024 * 1024
PART_SIZE = 90 * BLOCK  # GitHub rejects files >= 100 MiB in normal Git.


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            sha.update(block)
    return sha.hexdigest()


def split():
    if MANIFEST.exists():
        raise FileExistsError(f"Manifest already exists: {MANIFEST}")
    for rel in SOURCES:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        if any(path.parent.glob(path.name + ".part[0-9][0-9][0-9]")):
            raise FileExistsError(f"Parts already exist for {path}")
    manifest = {"version": 1, "part_size": PART_SIZE, "files": []}
    for rel in SOURCES:
        path = ROOT / rel
        full_hash = hashlib.sha256()
        parts = []
        print(f"Splitting {rel}", flush=True)
        with path.open("rb") as source:
            for index in range((path.stat().st_size + PART_SIZE - 1) // PART_SIZE):
                part = path.with_name(path.name + f".part{index:03d}")
                remaining = PART_SIZE
                part_hash = hashlib.sha256()
                with part.open("xb") as dest:
                    while remaining:
                        data = source.read(min(BLOCK, remaining))
                        if not data:
                            break
                        dest.write(data)
                        full_hash.update(data)
                        part_hash.update(data)
                        remaining -= len(data)
                parts.append({"path": part.relative_to(ROOT).as_posix(),
                              "size": part.stat().st_size, "sha256": part_hash.hexdigest()})
        manifest["files"].append({"path": rel, "size": path.stat().st_size,
                                  "sha256": full_hash.hexdigest(), "parts": parts})
        print(f"Created {len(parts)} parts; SHA-256 {full_hash.hexdigest()}", flush=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST)
    print(f"Wrote {MANIFEST.relative_to(ROOT)}", flush=True)


def restore():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("Unsupported manifest version")
    for entry in manifest["files"]:
        target = ROOT / entry["path"]
        if target.exists():
            if target.stat().st_size != entry["size"] or digest(target) != entry["sha256"]:
                raise FileExistsError(f"Existing file differs; refusing to overwrite: {target}")
            print(f"Verified existing {entry['path']}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".reassembling")
        full_hash = hashlib.sha256()
        print(f"Restoring {entry['path']}", flush=True)
        try:
            with temporary.open("xb") as output:
                for part in entry["parts"]:
                    source_path = ROOT / part["path"]
                    if source_path.stat().st_size != part["size"]:
                        raise ValueError(f"Wrong part size: {source_path}")
                    part_hash = hashlib.sha256()
                    with source_path.open("rb") as source:
                        while data := source.read(BLOCK):
                            output.write(data)
                            part_hash.update(data)
                            full_hash.update(data)
                    if part_hash.hexdigest() != part["sha256"]:
                        raise ValueError(f"Wrong part hash: {source_path}")
            if temporary.stat().st_size != entry["size"] or full_hash.hexdigest() != entry["sha256"]:
                raise ValueError(f"Restored file does not match manifest: {target}")
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f"Restored and verified {entry['path']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("split", "restore"))
    args = parser.parse_args()
    split() if args.command == "split" else restore()
