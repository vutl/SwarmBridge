"""Collect runs/uav3d_comm/tracking/<name>/metrics_summary.json into one table (optionally filtered by a regex).
Usage: python -I track_table.py TRACK_DIR [REGEX]"""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
rows = []
for f in sorted(root.glob("*/metrics_summary.json")):
    name = f.parent.name
    if pat and not pat.search(name):
        continue
    s = json.loads(f.read_text())
    rows.append((name, s.get("amota"), s.get("amotp"), s.get("mota"), s.get("motp"), s.get("recall"), s.get("ids")))
print(f"{'name':34s} {'AMOTA':>6s} {'AMOTP':>6s} {'MOTA':>6s} {'MOTP':>6s} {'recall':>6s} {'IDS':>5s}")
for r in rows:
    print(f"{r[0]:34s} " + " ".join(f"{v:6.3f}" if isinstance(v, float) else f"{str(v):>6s}" for v in r[1:6])
          + f" {int(r[6]) if r[6] is not None else -1:5d}")
(root / "tracking_table.json").write_text(json.dumps([dict(zip(
    ["name", "amota", "amotp", "mota", "motp", "recall", "ids"], r)) for r in rows], indent=1))
