"""Detection-level comparison on the held-out mini scenes (scene_11/13/15 are in the official val split;
the other 17 mini scenes are likely in the checkpoints' training data and are NOT used for numbers here).

Usage: python -I analyze_det.py DATA_ROOT INFER_DIR [RELEASED_DET_JSON ...]
INFER_DIR = output of infer_agents.py (one sub-dir per variant with results_nusc.json).
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaldet import MiniDB, evaluate, load_nusc_results  # noqa: E402
from fusion import BOX_BITS, HEADER_BITS, load_variant, late_fuse, to_preds, select  # noqa: E402

HELD = ["scene_11", "scene_13", "scene_15"]


def main():
    db = MiniDB(sys.argv[1])
    inf = Path(sys.argv[2])
    held = db.tokens(HELD)
    hs = set(held)
    rows = {}

    def report(name, preds, bits=None):
        r = evaluate(db, preds, held)
        rows[name] = dict(r, kbit_per_frame=bits)
        b = "" if bits is None else f"  {bits:8.2f} kbit/frame"
        print(f"{name:32s} mAP {r['mAP']:.3f}  AP@0.5/1/2/4 = "
              f"{r['ap_0.5']:.3f}/{r['ap_1.0']:.3f}/{r['ap_2.0']:.3f}/{r['ap_4.0']:.3f}{b}", flush=True)

    for f in sys.argv[3:]:
        report("released:" + Path(f).name, load_nusc_results(f, hs))
    V = {d.name: load_variant(d / "results_nusc.json") for d in sorted(inf.iterdir())
         if (d / "results_nusc.json").exists()}
    for n, v in V.items():
        report(n, to_preds({t: v[t] for t in held}))

    # sanity: our official output vs released file (same checkpoint) -> box-level agreement
    if sys.argv[3:] and "official" in V:
        rel = json.loads(Path(sys.argv[3]).read_text())["results"]
        d_all = []
        for t in held:
            a = V["official"][t]["xy"]
            b = np.array([x["translation"][:2] for x in rel[t]]).reshape(-1, 2)
            if len(a) and len(b):
                d_all += list(np.linalg.norm(a[:, None] - b[None], axis=-1).min(1))
        d_all = np.array(d_all)
        print(f"official vs released: {len(d_all)} boxes, nearest released box median {np.median(d_all):.3f} m, "
              f"<0.1 m: {(d_all < 0.1).mean():.1%}")

    agents = [V[f"agent{j}"] for j in range(5)] if all(f"agent{j}" in V for j in range(5)) else None
    if agents:
        for thr in (0.0, 0.3):
            fused = {}
            nbox = []
            for t in held:
                ego = agents[0][t]
                others = [select(a[t], a[t]["score"] >= thr) for a in agents[1:]]
                nbox.append(sum(len(o["score"]) for o in others))
                fused[t] = late_fuse(ego, others)
            kbit = (np.mean(nbox) * BOX_BITS + 4 * HEADER_BITS) / 1000
            report(f"late fusion all boxes (score>={thr})", to_preds(fused), kbit)
    out = inf / "det_heldout.json"
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
