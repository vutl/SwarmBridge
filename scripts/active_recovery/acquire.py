"""Download only VisDrone validation and two official frozen detector weights."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ASSETS = [
    ("data/visdrone/VisDrone2019-DET-val.zip", "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"),
    ("external/active_recovery/yolov8n.pt", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"),
    ("external/active_recovery/yolov8l.pt", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt"),
]


def main():
    rows = []
    for name, url in ASSETS:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            for attempt in range(3):
                try:
                    print("DOWNLOAD", name, flush=True)
                    req = urllib.request.Request(url, headers={"User-Agent": "SwarmBridge-feasibility/1.0"})
                    with urllib.request.urlopen(req, timeout=120) as src, temporary.open("wb") as dst:
                        while block := src.read(1024 * 1024): dst.write(block)
                    assert zipfile.is_zipfile(temporary), "not a ZIP/PyTorch archive"
                    temporary.replace(target)
                    break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(2)
        digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        rows.append(dict(file=name, url=url, bytes=target.stat().st_size, sha256=digest))
        print("READY", name, target.stat().st_size, digest[:12], flush=True)
    archive = ROOT / ASSETS[0][0]
    destination = ROOT / "data/visdrone"
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            resolved = (destination / member.filename).resolve()
            assert resolved.is_relative_to(destination.resolve()), "archive traversal"
        if not (destination / "VisDrone2019-DET-val/images").is_dir(): z.extractall(destination)
    out = ROOT / "runs/active_recovery/assets.json"
    out.write_text(json.dumps(dict(assets=rows, data_source="VisDrone author dataset, Ultralytics release mirror", weights="official COCO pretrained; no detector training"), indent=2) + "\n")


if __name__ == "__main__": main()
