"""Where does each message type help? Recall (centre distance < 2 m) of GT cars on the held-out scenes, split by
seg-based visibility from the data audit (runs/uav3d_audit/rows_0.3.npz):
  ego_sees       : UAV0 sees it
  helper_only    : UAV0 does not, at least one helper does  (the collaboration target)
  nobody         : no UAV sees it (heavily occluded / far)
Detections with score >= 0.3. Usage: python -I recall_by_vis.py DATA_ROOT NAME=RESULTS_JSON ...
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaldet import MiniDB, in_square, load_nusc_results  # noqa: E402

HELD = ["scene_11", "scene_13", "scene_15"]


def main():
    root = Path(sys.argv[1])
    db = MiniDB(root)
    samples = json.loads((root / "v1.0-mini" / "sample.json").read_text())
    rows = np.load(Path(__file__).resolve().parents[3] / "runs" / "uav3d_audit" / "rows_0.3.npz")["rows"]
    held = set(db.tokens(HELD))
    groups = {"ego_sees": [], "helper_only": [], "nobody": []}
    for r in rows:
        tok = samples[int(r[0])]["token"]
        if tok not in held:
            continue
        xy = r[13:15]
        if not in_square(xy[None], db.uav0[tok])[0]:
            continue
        vis = r[8:13].astype(bool)
        g = "ego_sees" if vis[0] else ("helper_only" if vis[1:].any() else "nobody")
        groups[g].append((tok, xy))
    print("GT cars in the eval square:", {k: len(v) for k, v in groups.items()})
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        P = load_nusc_results(path, held)
        out = []
        for g, items in groups.items():
            hit = 0
            for tok, xy in items:
                pxy, ps = P.get(tok, (np.zeros((0, 2)), np.zeros(0)))
                pxy = pxy[ps >= 0.3]
                hit += bool(len(pxy)) and np.linalg.norm(pxy - xy, axis=1).min() < 2.0
            out.append(f"{g} {hit / max(len(items), 1):.3f}")
        print(f"{name:20s} recall@2m  " + "  ".join(out))


if __name__ == "__main__":
    main()
