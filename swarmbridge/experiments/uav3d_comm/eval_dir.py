"""Evaluate every <dir>/<variant>/results_nusc.json on the held-out mini scenes; optional bits json (kbit/frame).
Usage: python -I eval_dir.py DATA_ROOT DIR [BITS_JSON]  -> prints a table and writes DIR/det_heldout.json
Variants named *_s<seed> are additionally averaged over seeds (mean +- std)."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaldet import MiniDB, evaluate, load_nusc_results  # noqa: E402

HELD = ["scene_11", "scene_13", "scene_15"]


def main():
    db = MiniDB(sys.argv[1])
    root = Path(sys.argv[2])
    bits = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else {}
    held = db.tokens(HELD)
    rows = {}
    for f in sorted(root.rglob("results_nusc.json")):
        name = str(f.parent.relative_to(root)).replace("\\", "/")
        r = evaluate(db, load_nusc_results(f, set(held)), held)
        rows[name] = r
        b = bits.get(name)
        print(f"{name:34s} mAP {r['mAP']:.3f}  ({r['ap_0.5']:.3f}/{r['ap_1.0']:.3f}/{r['ap_2.0']:.3f}/{r['ap_4.0']:.3f})"
              + (f"  {b:10.1f} kbit/frame" if b is not None else ""), flush=True)
    groups = defaultdict(list)
    for n, r in rows.items():
        m = re.match(r"(.*)_s\d+(.*)", n)
        if m:
            groups[m.group(1) + m.group(2)].append(r["mAP"])
    for g, v in sorted(groups.items()):
        print(f"  mean over seeds {g:28s} mAP {np.mean(v):.3f} +- {np.std(v):.3f} (n={len(v)})")
    (root / "det_heldout.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
