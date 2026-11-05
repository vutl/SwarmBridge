"""Wireless pilot C v2 (audit fixes 05/11), UAV3D side.

  payload : write the REAL per-helper, per-frame KF-PV packet size (bits, boxcodec) for the trace frames that are
            mapped to UAV3D dev frames (seed m, scene i -> trace episode TEST0 + 3 m + i, frames OFFSET ..), for c_feas2.
  track   : build ledgers from c_feas2 completion times and run the KF-PV receiver (kf_nor). A message of helper frame s
            that completes tau seconds after the frame start arrives at s*0.2 + tau and is usable at receiver frame
            k = s + ceil((tau - LAT) / 0.2) (k >= s): LAT = 0 means the receiver fuses at its frame instant, so any
            transmitted message is at least one frame old; LAT = 0.05 allows a 50 ms fusion deadline. Messages not
            complete within their 200 ms window are dropped. 'all_age0' = every message at age 0 (no wireless, as v1).
Usage: python -I c_track2.py payload DATA_ROOT AGENT_DIR OUT_JSON
       python -I c_track2.py track DATA_ROOT AGENT_DIR COMPL_NPZ OUT [--lat 0.0]
"""
import hashlib
import json
import math
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

TEST0, OFFSET, SEEDS, FRAME = 96, 10, [0, 1, 2], 0.2
HELPERS = (1, 2, 3, 4)


def setup(root, adir):
    db = MiniDB(root)
    frames = L.scene_frames_from_db(db, BP.DEV)
    pos = BP.uav_positions(root)
    A = {j: load_variant(Path(adir) / f"agent{j}" / "results_nusc.json") for j in range(5)}
    msgs, bits = {}, {}
    for j in HELPERS:
        for sc, toks in frames.items():
            trk = K.run_sequence([A[j][t] for t in toks])
            for k, t in enumerate(toks):
                m = select(BP.base(trk[k]), trk[k]["score"] >= BP.TAU_SEND)
                data, st = BC.encode(j, k, pos[(t, j)], m)
                msgs[(j, t)] = BC.decode(data)[1]
                bits[(j, t)] = 8 * st["bytes"]
    return db, frames, A, msgs, bits


def payload(root, adir, out_json):
    db, frames, A, msgs, bits = setup(root, adir)
    tab = {}
    for m in SEEDS:
        for i, (sc, toks) in enumerate(sorted(frames.items())):
            ep = f"ep{TEST0 + 3 * m + i:04d}"
            tab[ep] = {"offset": OFFSET, "bits": [[int(bits[(j, t)]) for t in toks] for j in HELPERS]}
    allb = [b for b in bits.values()]
    Path(out_json).write_text(json.dumps(dict(default_bits=int(np.mean(allb)), mean_bits=float(np.mean(allb)),
                                              max_bits=int(np.max(allb)), episodes=tab), indent=1))
    print("payload bits mean", np.mean(allb), "max", np.max(allb))


def track(root, adir, cpath, out, lat):
    db, frames, A, msgs, bits = setup(root, adir)
    fidx = {t: k for toks in frames.values() for k, t in enumerate(toks)}
    toks_all = [t for toks in frames.values() for t in toks]
    late = [t for t in toks_all if fidx[t] >= 3]
    cl = np.load(cpath)
    policies = sorted({k.split("__")[0] for k in cl.files}) + ["all_age0", "all_age1"]
    rows = []
    for pol in policies:
        for m in SEEDS:
            use, n_deliv, n_all, ages = {}, 0, 0, []
            for i, (sc, toks) in enumerate(sorted(frames.items())):
                ep = f"ep{TEST0 + 3 * m + i:04d}"
                T = len(toks)
                packets = []
                for j in HELPERS:
                    for s in range(T):
                        if pol == "all_age0":
                            arr = s
                        elif pol == "all_age1":
                            arr = s + 1
                        else:
                            tau = cl[f"{pol}__{ep}"][j - 1, OFFSET + s]
                            arr = None if not np.isfinite(tau) else s + max(0, math.ceil((tau - lat) / FRAME - 1e-9))
                        n_all += 1
                        n_deliv += arr is not None
                        packets.append({"scene": sc, "sender": j, "send_frame": s, "seq": s, "drop": arr is None,
                                        "delay": None if arr is None else arr - s, "arrival_frame": arr})
                u = L.usage(packets, T, "latest", 3)
                for t, tok in enumerate(toks):
                    use[tok] = {j: (None if u[j][t][0] is None else toks[u[j][t][0]], u[j][t][1]) for j in HELPERS}
                    ages += [u[j][t][1] for j in HELPERS if u[j][t][1] is not None]
            preds = {}
            for t in toks_all:
                own = BP.base(A[0][t])
                sets = [(0, own["xy"], own["score"], BP.records(own, t))]
                for j in HELPERS:
                    src, age = use[t][j]
                    if src is None:
                        continue
                    mm = K.forecast(msgs[(j, src)], age)
                    sets.append((j, mm["xy"], mm["score"], BP.records(mm, t)))
                recs = soft_fuse(sets, "noisyor", return_records=True)
                recs.sort(key=lambda r: -r["detection_score"])
                preds[t] = [dict(r, sample_token=t) for r in recs[:BP.CAP]]
            path = Path(out) / "dets" / pol / f"m{m}" / "results_nusc.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = json.dumps({"meta": BP.META, "results": preds}).encode()
            path.write_bytes(blob)
            pr = {t: (np.array([r["translation"][:2] for r in v]).reshape(-1, 2),
                      np.array([r["detection_score"] for r in v])) for t, v in preds.items()}
            ra, rl = evaluate(db, pr, toks_all), evaluate(db, pr, late)
            row = dict(policy=pol, seed=m, lat=lat, mAP=round(ra["mAP"], 4), mAP_f3=round(rl["mAP"], 4),
                       deliv_frac=round(n_deliv / n_all, 4), mean_age=round(float(np.mean(ages)), 3) if ages else None,
                       pred=str(path.relative_to(out)), sha256=hashlib.sha256(blob).hexdigest())
            rows.append(row)
            print(f"{pol:22s} m{m} deliv {row['deliv_frac']:.3f} age {row['mean_age']} mAP {row['mAP']:.3f}", flush=True)
    (Path(out) / "c_track2.json").write_text(json.dumps(dict(test0=TEST0, offset=OFFSET, lat=lat, rows=rows), indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "payload":
        payload(*sys.argv[2:5])
    else:
        lat = float(sys.argv[sys.argv.index("--lat") + 1]) if "--lat" in sys.argv else 0.0
        track(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], lat)
