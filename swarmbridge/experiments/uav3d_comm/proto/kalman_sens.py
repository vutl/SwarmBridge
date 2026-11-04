"""Bounded sensitivity of the KF-PV baseline to the sender Kalman parameters, plus a receiver forecast gate
(protocol v1.1 section 10.2 follow-up; Codex review: "độ nhạy Kalman có giới hạn"). Dev scenes only; CPU only.

Grid: sigma_a in {1, 3, 6} m/s^2 x gate in {2, 3, 5} m (sigma_r 0.5 fixed), receiver = kf_nor (forecast, noisy-OR,
cap 500). Extra arms on the default tracker (3, 3): forecast only boxes whose decoded speed >= vmin (vmin 0.5, 1.0
m/s), others are used stale (value_curve.py showed forecasting hurts static cars and helps fast ones).
Ledgers: sync_s0, fix1_s0, fix2_s0, rand_s0 (same files as box_proto).
Usage: python -I kalman_sens.py DATA_ROOT AGENT_DIR OUT
"""
import hashlib
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
from evaldet import MiniDB, evaluate  # noqa: E402
from fusion import load_variant, select  # noqa: E402
from soft_fusion import fuse as soft_fuse  # noqa: E402

GRID = [(sa, g) for sa in (1.0, 3.0, 6.0) for g in (2.0, 3.0, 5.0)]
VMIN = [0.5, 1.0]
LEDGERS = [("sync", 0), ("fix1", 0), ("fix2", 0), ("rand", 0)]
HELPERS = (1, 2, 3, 4)


def main():
    root, adir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    db = MiniDB(root)
    frames = L.scene_frames_from_db(db, BP.DEV)
    fidx = {t: k for toks in frames.values() for k, t in enumerate(toks)}
    toks_all = [t for toks in frames.values() for t in toks]
    late = [t for t in toks_all if fidx[t] >= 3]
    pos = BP.uav_positions(root)
    A = {j: load_variant(adir / f"agent{j}" / "results_nusc.json") for j in range(5)}
    leds = {f"{s}_s{e}": L.build(frames, s, e) for s, e in LEDGERS}
    arms = [(f"sa{sa:g}_g{g:g}", sa, g, None) for sa, g in GRID] + [(f"sa3_g3_vmin{v:g}", 3.0, 3.0, v) for v in VMIN]
    rows, cache = [], {}
    for tag, sa, g, vmin in arms:
        if (sa, g) not in cache:
            msgs = {}
            for j in HELPERS:
                for sc, toks in frames.items():
                    tr = K.Tracker(sigma_a=sa, gate=g)
                    for k, t in enumerate(toks):
                        o = tr.step(A[j][t])
                        m = select(BP.base(o), o["score"] >= BP.TAU_SEND)
                        data, st = BC.encode(j, k, pos[(t, j)], m)
                        msgs[(j, t)] = (BC.decode(data)[1], st)
            cache[(sa, g)] = msgs
        msgs = cache[(sa, g)]
        for name, led in leds.items():
            use = L.lookup(led)
            preds = {}
            for t in toks_all:
                own = BP.base(A[0][t])
                sets = [(0, own["xy"], own["score"], BP.records(own, t))]
                for j in HELPERS:
                    src, age = use[t][j]
                    if src is None:
                        continue
                    dec = msgs[(j, src)][0]
                    m = K.forecast(dec, age)
                    if vmin is not None:
                        slow = np.linalg.norm(dec["vel"], axis=1) < vmin
                        m["xy"] = np.where(slow[:, None], dec["xy"], m["xy"])
                    sets.append((j, m["xy"], m["score"], BP.records(m, t)))
                recs = soft_fuse(sets, "noisyor", return_records=True)
                recs.sort(key=lambda r: -r["detection_score"])
                preds[t] = [dict(r, sample_token=t) for r in recs[:BP.CAP]]
            path = out / "dets" / tag / name / "results_nusc.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = json.dumps({"meta": BP.META, "results": preds}).encode()
            path.write_bytes(blob)
            pr = {t: (np.array([r["translation"][:2] for r in v]).reshape(-1, 2),
                      np.array([r["detection_score"] for r in v])) for t, v in preds.items()}
            ra, rl = evaluate(db, pr, toks_all), evaluate(db, pr, late)
            kb = np.mean([sum(8 * msgs[(j, t)][1]["bytes"] for j in HELPERS) for t in toks_all]) / 1000
            row = {"arm": tag, "sigma_a": sa, "gate": g, "vmin": vmin, "ledger": name, "mAP": round(ra["mAP"], 4),
                   "mAP_f3": round(rl["mAP"], 4), "kbit_tx_mean": round(float(kb), 3),
                   "pred": str(path.relative_to(out)), "sha256": hashlib.sha256(blob).hexdigest()}
            rows.append(row)
            print(f"{tag:16s} {name:8s} mAP {row['mAP']:.3f} (f>=3 {row['mAP_f3']:.3f}) tx {kb:5.1f} kbit", flush=True)
    (out / "kalman_sens.json").write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
