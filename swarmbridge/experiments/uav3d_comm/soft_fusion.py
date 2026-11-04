"""Can BOX messages recover the weak-evidence cars that feature fusion gets? Send low-score boxes too and fuse
scores across UAVs instead of keeping the max.

Fusion at the receiver: cluster all boxes (ego + helpers) by centre distance (greedy around the highest score,
radius 2 m); fused centre = score-weighted mean; fused score:
  max      : max score (= late_fuse)
  noisyor  : 1 - prod(1 - s_i)  over the UAVs in the cluster (one box per UAV, its best)
  mean5    : sum_i s_i / 5      (absent UAVs count as 0; 'votes')
Helpers send boxes with score >= tau_send (bits counted per box).
Usage: python -I soft_fusion.py DATA_ROOT AGENT_DET_DIR OUT
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaldet import MiniDB, evaluate  # noqa: E402
from fusion import BOX_BITS, HEADER_BITS, load_variant  # noqa: E402

HELD = ["scene_11", "scene_13", "scene_15"]


def fuse(sets, rule, radius=2.0, return_records=False):
    """sets: list of (agent, xy [N,2], score [N]). Returns fused xy, score and member records index."""
    xy = np.concatenate([s[1] for s in sets]) if sets else np.zeros((0, 2))
    sc = np.concatenate([s[2] for s in sets]) if sets else np.zeros(0)
    ag = np.concatenate([np.full(len(s[2]), s[0]) for s in sets]) if sets else np.zeros(0, int)
    recs = [r for s in sets for r in (s[3] if len(s) > 3 else [None] * len(s[2]))]
    order = np.argsort(-sc)
    used = np.zeros(len(sc), bool)
    out_xy, out_s, out_r = [], [], []
    for i in order:
        if used[i]:
            continue
        near = (~used) & (np.linalg.norm(xy - xy[i], axis=1) < radius)
        idx = np.where(near)[0]
        used[idx] = True
        best = {}
        for k in idx:                       # one (best) box per UAV in the cluster
            if ag[k] not in best or sc[k] > sc[best[ag[k]]]:
                best[ag[k]] = k
        ks = np.array(list(best.values()))
        s = sc[ks]
        w = s / s.sum()
        out_xy.append((xy[ks] * w[:, None]).sum(0))
        out_r.append(recs[i])
        if rule == "max":
            out_s.append(s.max())
        elif rule == "noisyor":
            out_s.append(1 - np.prod(1 - s))
        else:
            out_s.append(s.sum() / 5)
    if return_records:
        res = []
        for p, s, r in zip(out_xy, out_s, out_r):
            r = dict(r); r["translation"] = [float(p[0]), float(p[1]), r["translation"][2]]
            r["detection_score"] = float(s); res.append(r)
        return res
    return np.array(out_xy).reshape(-1, 2), np.array(out_s)


def main():
    db = MiniDB(sys.argv[1])
    adir, out = Path(sys.argv[2]), Path(sys.argv[3])
    held = db.tokens(HELD)
    A = {j: load_variant(adir / f"agent{j}" / "results_nusc.json") for j in range(5)}
    rows = []
    for tau in (0.3, 0.1, 0.05):
        for rule in ("max", "noisyor", "mean5"):
            preds, bits = {}, []
            for t in held:
                sets = []
                for j in range(5):
                    d = A[j][t]
                    m = d["score"] >= (tau if j else 0.05)      # receiver always uses its own boxes >= 0.05
                    sets.append((j, d["xy"][m], d["score"][m]))
                    if j:
                        bits.append(int(m.sum()) * BOX_BITS + HEADER_BITS)
                preds[t] = fuse(sets, rule)
            r = evaluate(db, preds, held)
            kbit = float(np.sum(bits)) / len(held) / 1000
            rows.append(dict(tau=tau, rule=rule, mAP=r["mAP"], kbit=kbit,
                             ap=[r[f"ap_{k}"] for k in (0.5, 1.0, 2.0, 4.0)]))
            print(f"send>={tau:<5} {rule:8s} mAP {r['mAP']:.3f}  ({'/'.join(f'{x:.3f}' for x in rows[-1]['ap'])})"
                  f"  {kbit:6.1f} kbit/frame", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "soft_fusion.json").write_text(json.dumps(rows, indent=1))
    # nuScenes JSON of the score-weighted-centre fusion (helpers send >= 0.1 / >= 0.3, max score) for tracking
    raw = {j: json.loads((adir / f"agent{j}" / "results_nusc.json").read_text())["results"] for j in range(5)}
    for tau in (0.1, 0.3):
        res = {}
        for t in held:
            sets = []
            for j in range(5):
                d = A[j][t]
                m = d["score"] >= (tau if j else 0.05)
                sets.append((j, d["xy"][m], d["score"][m], [raw[j][t][k] for k in np.where(m)[0]]))
            res[t] = [dict(r, sample_token=t) for r in fuse(sets, "max", return_records=True)]
        p = out / "dets" / f"wmean@{tau}" / "results_nusc.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                                          "use_map": False, "use_external": False}, "results": res}))
        print("wrote", p)


if __name__ == "__main__":
    main()
