"""Box messages under delay and packet loss (receiver UAV0, helpers UAV1..4, same lowerbound detector, all@0.3).

  delay{d}_raw : helpers' boxes from frame t-d used as they are (global frame)
  delay{d}_cv  : CERF-style compensation: each delayed box gets a constant-velocity forecast; velocity from
                 matching it to the same helper's boxes at t-d-1 (nearest centre within 3 m), else 0
  loss{p}_s{k} : helper messages dropped with the SAME Bernoulli draws as the feature experiments (drops.json)
Writes nuScenes JSONs to OUT/dets/<cond>/results_nusc.json and prints mAP.

Usage: python -I box_stress.py DATA_ROOT AGENT_DET_DIR DROPS_JSON OUT
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaldet import MiniDB, evaluate  # noqa: E402
from fusion import load_variant, late_fuse, select, to_preds  # noqa: E402

from soft_fusion import fuse as soft_fuse  # noqa: E402

HELD = ["scene_11", "scene_13", "scene_15"]
TAU = 0.3
WMEAN = "--wmean" in sys.argv          # score-weighted-centre fusion instead of keep-the-max-box
if WMEAN:
    sys.argv.remove("--wmean")
RULE = "noisyor" if "--noisyor" in sys.argv else "max"
if "--noisyor" in sys.argv:
    sys.argv.remove("--noisyor")


def main():
    db = MiniDB(sys.argv[1])
    adir, drops, out = Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4])
    held = db.tokens(HELD)
    raw = {j: json.loads((adir / f"agent{j}" / "results_nusc.json").read_text())["results"] for j in range(5)}
    A = {j: load_variant(adir / f"agent{j}" / "results_nusc.json") for j in range(5)}
    for j in range(5):
        for t in A[j]:
            A[j][t]["records"] = np.array(raw[j][t], dtype=object)
    prev = {t: db.samples[t]["prev"] for t in db.samples}

    def back(t, d):
        for _ in range(d):
            t = prev[t] or t
        return t

    def msg(j, t_now, d, cv):
        src = back(t_now, d)
        m = A[j][src]
        m = select(m, m["score"] >= TAU)
        if d == 0 or src == t_now:
            return m
        m = {k: (v.copy() if k != "records" else np.array([dict(r) for r in v], dtype=object)) for k, v in m.items()}
        if cv:
            older = prev[src]
            if older:
                o = A[j][older]["xy"][A[j][older]["score"] >= TAU]
                for i in range(len(m["xy"])):
                    if len(o):
                        dd = np.linalg.norm(o - m["xy"][i], axis=1)
                        k = int(np.argmin(dd))
                        if dd[k] < 3.0:
                            m["xy"][i] = m["xy"][i] + d * (m["xy"][i] - o[k])
        for i, r in enumerate(m["records"]):       # write positions back into the records for the tracker
            r["translation"] = [float(m["xy"][i][0]), float(m["xy"][i][1]), r["translation"][2]]
            r["sample_token"] = t_now
        return m

    conds = {"clean": (0, False, None)}
    for d in (1, 2):
        conds[f"delay{d}_raw"] = (d, False, None)
        conds[f"delay{d}_cv"] = (d, True, None)
    for key in drops:
        conds[f"loss{key}"] = (0, False, key)
    rows = {}
    for c, (d, cv, dk) in conds.items():
        fused = {}
        for t in held:
            sent = []
            for j in range(1, 5):
                if dk and drops[dk][t][j - 1]:
                    continue
                sent.append(msg(j, t, d, cv))
            if WMEAN:
                sets = [(0, A[0][t]["xy"], A[0][t]["score"], list(A[0][t]["records"]))]
                sets += [(j + 1, s["xy"], s["score"], list(s["records"])) for j, s in enumerate(sent)]
                recs = [dict(r, sample_token=t) for r in soft_fuse(sets, RULE, return_records=True)]
                f = {"xy": np.array([r["translation"][:2] for r in recs]).reshape(-1, 2),
                     "score": np.array([r["detection_score"] for r in recs])}
            else:
                f = late_fuse(A[0][t], sent)
                recs = []
                for r in f["records"]:
                    r = dict(r); r["sample_token"] = t; recs.append(r)
            f["records"] = recs
            fused[t] = f
        r = evaluate(db, to_preds(fused), held)
        rows[c] = r
        p = out / "dets" / c / "results_nusc.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                                          "use_map": False, "use_external": False},
                                 "results": {t: fused[t]["records"] for t in held}}))
        print(f"{c:22s} mAP {r['mAP']:.3f}  ({r['ap_0.5']:.3f}/{r['ap_1.0']:.3f}/{r['ap_2.0']:.3f}/{r['ap_4.0']:.3f})",
              flush=True)
    (out / "box_stress.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
