"""Arrival-age value curve of helper messages (diagnostic; uses GT; dev scenes only; CPU only).

For every receiver frame t >= 3 and helper j, take the SAME kind of message that j would have sent at s = t - a
(a = 0..3 frames old on arrival), pass it through the codec, place it at t either by Kalman forecast (KF-PV, age * 0.2 s)
or stale (no forecast), and count against ground truth at t inside UAV0's evaluation square:
  recov    GT cars that UAV0's own confident detections (score >= 0.3) miss but the message matches (<= 2 m)
  uniq     recov cars that none of the OTHER helpers' messages (same age a) also matches  -> marginal value given others
  confirm  GT cars that UAV0 already detects and the message also matches (confirmation / redundancy)
  fp       message boxes in the square that match no GT car
  err      mean centre error of matched message boxes (m)
Matching is one-to-one, greedy by message score, 2 m. GT speed from the annotation vx, vy: static < 0.5 m/s,
slow 0.5-3, fast > 3. Messages: "kf" = KF-PV content (box_proto kf family), "box" = raw detections (box family).
Usage: python -I value_curve.py DATA_ROOT AGENT_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import boxcodec as BC  # noqa: E402
import box_proto as BP  # noqa: E402
import kalman as K  # noqa: E402
import ledger as L  # noqa: E402
from evaldet import MiniDB, in_square  # noqa: E402
from fusion import load_variant, select  # noqa: E402

R_MATCH = 2.0
EGO_CONF = 0.3
AGES = [0, 1, 2, 3]
HELPERS = (1, 2, 3, 4)
BINS = [("static", 0.0, 0.5), ("slow", 0.5, 3.0), ("fast", 3.0, 1e9)]


def match(pred_xy, pred_score, gt_xy):
    """greedy one-to-one by score -> (array gt index or -1 per pred, distances)"""
    out = -np.ones(len(pred_xy), int)
    dist = np.full(len(pred_xy), np.nan)
    if len(gt_xy) == 0:
        return out, dist
    free = np.ones(len(gt_xy), bool)
    for i in np.argsort(-pred_score):
        d = np.linalg.norm(gt_xy - pred_xy[i], axis=1)
        d[~free] = np.inf
        g = int(np.argmin(d))
        if d[g] <= R_MATCH:
            out[i], dist[i] = g, d[g]
            free[g] = False
    return out, dist


def main():
    root, adir, outp = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    db = MiniDB(root)
    frames = L.scene_frames_from_db(db, BP.DEV)
    pos = BP.uav_positions(root)
    A = {j: load_variant(adir / f"agent{j}" / "results_nusc.json") for j in range(5)}
    msg = {}
    for j in HELPERS:
        for sc, toks in frames.items():
            trk = K.run_sequence([A[j][t] for t in toks])
            for k, t in enumerate(toks):
                for fam, m in (("box", BP.base(A[j][t])), ("kf", trk[k])):
                    m = select(BP.base(m), m["score"] >= BP.TAU_SEND)
                    data, _ = BC.encode(j, k, pos[(t, j)], m)
                    msg[(fam, j, t)] = BC.decode(data)[1]

    stats = {}
    for sc, toks in frames.items():
        for k in range(3, len(toks)):
            t = toks[k]
            ann = [a for a in db.gt[t]]
            gxy = np.array([a["translation"][:2] for a in ann]).reshape(-1, 2)
            gsp = np.array([np.hypot(a.get("vx", 0.0), a.get("vy", 0.0)) for a in ann])
            keep = in_square(gxy, db.uav0[t])
            gxy, gsp = gxy[keep], gsp[keep]
            ego = BP.base(A[0][t])
            ec = ego["score"] >= EGO_CONF
            em, _ = match(ego["xy"][ec], ego["score"][ec], gxy)
            ego_hit = np.zeros(len(gxy), bool)
            ego_hit[em[em >= 0]] = True
            for fam in ("kf", "box"):
                for mode in (("forecast", "stale") if fam == "kf" else ("stale",)):
                    for a in AGES:
                        src = toks[k - a]
                        hits = {}
                        res = {}
                        for j in HELPERS:
                            m = msg[(fam, j, src)]
                            xy = m["xy"] + (m["vel"] * a * K.DT if mode == "forecast" else 0.0)
                            ins = in_square(xy, db.uav0[t])
                            mm, dd = match(xy[ins], m["score"][ins], gxy)
                            h = np.zeros(len(gxy), bool)
                            h[mm[mm >= 0]] = True
                            hits[j] = h
                            res[j] = (mm, dd, int(ins.sum()))
                        for j in HELPERS:
                            mm, dd, nin = res[j]
                            others = np.any([hits[o] for o in HELPERS if o != j], axis=0)
                            rec = hits[j] & ~ego_hit
                            key = f"{fam}|{mode}|{a}"
                            s = stats.setdefault(key, {"n_msg": 0, "recov": 0, "uniq": 0, "confirm": 0, "fp": 0,
                                                       "n_box": 0, "err_sum": 0.0, "n_match": 0,
                                                       **{f"recov_{b}": 0 for b, _, _ in BINS},
                                                       **{f"missed_{b}": 0 for b, _, _ in BINS}})
                            s["n_msg"] += 1
                            s["recov"] += int(rec.sum())
                            s["uniq"] += int((rec & ~others).sum())
                            s["confirm"] += int((hits[j] & ego_hit).sum())
                            s["fp"] += int(nin - (mm >= 0).sum())
                            s["n_box"] += nin
                            s["err_sum"] += float(np.nansum(dd))
                            s["n_match"] += int((mm >= 0).sum())
                            for b, lo, hi in BINS:
                                inb = (gsp >= lo) & (gsp < hi)
                                s[f"recov_{b}"] += int((rec & inb).sum())
                                s[f"missed_{b}"] += int((~ego_hit & inb).sum())
    rows = []
    print("family mode     age | recov/msg uniq/msg confirm/msg fp/msg  err(m) | recov rate static slow fast")
    for key, s in stats.items():
        fam, mode, a = key.split("|")
        r = {"family": fam, "mode": mode, "age": int(a), **s,
             "recov_per_msg": s["recov"] / s["n_msg"], "uniq_per_msg": s["uniq"] / s["n_msg"],
             "confirm_per_msg": s["confirm"] / s["n_msg"], "fp_per_msg": s["fp"] / s["n_msg"],
             "err_m": s["err_sum"] / max(1, s["n_match"])}
        for b, _, _ in BINS:
            r[f"recov_rate_{b}"] = s[f"recov_{b}"] / max(1, s[f"missed_{b}"])
        rows.append(r)
        print(f"{fam:6s} {mode:8s} {a:>3s} | {r['recov_per_msg']:9.2f} {r['uniq_per_msg']:8.2f} "
              f"{r['confirm_per_msg']:11.2f} {r['fp_per_msg']:6.2f} {r['err_m']:7.2f} | "
              f"{r['recov_rate_static']:.3f} {r['recov_rate_slow']:.3f} {r['recov_rate_fast']:.3f}")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"r_match": R_MATCH, "ego_conf": EGO_CONF, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
