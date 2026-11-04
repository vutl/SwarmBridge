"""Box-message policies at the receiver UAV0, same frozen detector (lowerbound) for every UAV.
Held-out mini scenes only. Writes nuScenes-format JSONs (for the tracking eval) + a rate/accuracy table.

Policies (helpers j=1..4 -> receiver 0):
  none           : UAV0 alone
  all@tau        : every helper box with score >= tau
  topk@K         : each helper sends its K most confident boxes
  novelty@tau    : UAV0 broadcasts its own boxes (score >= 0.3); helpers send only boxes (score >= tau) with no
                   receiver box within 2 m  (deployable "send what the receiver does not have")
  oracle_missed  : helpers send only boxes that hit a GT car the receiver missed (uses GT; upper reference)
Fusion at the receiver: union + centre-NMS (2 m), keeping the most confident copy.
Bits: fusion.BOX_BITS per box + HEADER_BITS per message (+ receiver broadcast for novelty).

Usage: python -I analyze_comm.py DATA_ROOT INFER_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaldet import MiniDB, evaluate  # noqa: E402
from fusion import BOX_BITS, HEADER_BITS, load_variant, late_fuse, select, to_preds  # noqa: E402

HELD = ["scene_11", "scene_13", "scene_15"]


def nearest(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.inf)
    return np.linalg.norm(a[:, None] - b[None], axis=-1).min(1)


def write_nusc(dets, raw_path, out_path):
    """Re-emit fused detections in nuScenes JSON (copy full records from the per-agent raw results)."""
    out = {"meta": {"use_camera": True, "use_lidar": False, "use_radar": False, "use_map": False,
                    "use_external": False}, "results": {}}
    for t, d in dets.items():
        out["results"][t] = d["records"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out))


def main():
    db = MiniDB(sys.argv[1])
    inf, out = Path(sys.argv[2]), Path(sys.argv[3])
    held = db.tokens(HELD)
    raw = {j: json.loads((inf / f"agent{j}" / "results_nusc.json").read_text())["results"] for j in range(5)}
    A = {j: load_variant(inf / f"agent{j}" / "results_nusc.json") for j in range(5)}
    for j in range(5):  # attach raw records so fused sets can be written back as nuScenes JSON
        for t in A[j]:
            A[j][t]["records"] = np.array(raw[j][t], dtype=object)

    def policy(name, chooser, extra_bits=lambda t: 0):
        fused, bits = {}, []
        for t in held:
            sent = [chooser(j, t) for j in range(1, 5)]
            bits.append(sum(len(s["score"]) * BOX_BITS + HEADER_BITS for s in sent) + extra_bits(t))
            f = late_fuse(A[0][t], sent)
            f["records"] = f["records"].tolist()
            fused[t] = f
        r = evaluate(db, to_preds(fused), held)
        kbit = float(np.mean(bits)) / 1000
        nbox = float(np.mean([sum(len(chooser(j, t)["score"]) for j in range(1, 5)) for t in held]))
        rows.append(dict(policy=name, mAP=r["mAP"], ap=[r[f"ap_{k}"] for k in (0.5, 1.0, 2.0, 4.0)],
                         kbit_per_frame=kbit, boxes_per_frame=nbox))
        write_nusc(fused, None, out / "dets" / name / "results_nusc.json")
        print(f"{name:22s} mAP {r['mAP']:.3f}  ({'/'.join(f'{x:.3f}' for x in rows[-1]['ap'])})  "
              f"{kbit:7.2f} kbit/frame  {nbox:6.1f} boxes/frame", flush=True)

    rows = []
    policy("none", lambda j, t: select(A[j][t], np.zeros(len(A[j][t]["score"]), bool)))
    for tau in (0.0, 0.2, 0.3, 0.5):
        policy(f"all@{tau}", lambda j, t, tau=tau: select(A[j][t], A[j][t]["score"] >= tau))
    for K in (5, 10, 20, 40):
        def topk(j, t, K=K):
            o = np.argsort(-A[j][t]["score"])[:K]
            m = np.zeros(len(A[j][t]["score"]), bool); m[o] = True
            return select(A[j][t], m)
        policy(f"topk@{K}", topk)
    BROAD = 0.3
    for tau in (0.2, 0.3, 0.5):
        def novelty(j, t, tau=tau):
            rx = A[0][t]["xy"][A[0][t]["score"] >= BROAD]
            d = A[j][t]
            return select(d, (d["score"] >= tau) & (nearest(d["xy"], rx) > 2.0))
        policy(f"novelty@{tau}", novelty,
               extra_bits=lambda t: int((A[0][t]["score"] >= BROAD).sum()) * BOX_BITS + HEADER_BITS)

    def oracle(j, t):
        g = db.gt_xy(t)
        rx = A[0][t]["xy"][A[0][t]["score"] >= BROAD]
        g_missed = g[nearest(g, rx) > 2.0]
        d = A[j][t]
        return select(d, (d["score"] >= 0.2) & (nearest(d["xy"], g_missed) < 2.0))
    policy("oracle_missed", oracle)

    # redundancy diagnostics
    dup = []
    for t in held:
        rx = A[0][t]["xy"][A[0][t]["score"] >= BROAD]
        for j in range(1, 5):
            d = A[j][t]; m = d["score"] >= 0.3
            if m.sum():
                dup.append((nearest(d["xy"][m], rx) <= 2.0).mean())
    print(f"helper boxes (score>=0.3) already detected by UAV0: {np.mean(dup):.1%}")
    (out / "box_policies.json").write_text(json.dumps(rows, indent=1))
    print("wrote", out / "box_policies.json")


if __name__ == "__main__":
    main()
