"""Wireless pilot C: turn per-frame message delivery from link-management policies (c_feas.py on the Sionna RT traces)
into UAV3D packet ledgers and run the KF-PV receiver (kf_nor, same as box_proto), so tracking (AMOTA) measures what link
management is worth. Trace-coupled simulation: UAV3D helpers j = 1..4 are mapped to trace UAVs u = j - 1; there is no
geometric co-registration between the UAV3D scenes and the radio scene.

Mapping (seed m, dev scene i): trace episode TEST0 + 3 m + i, frames OFFSET .. OFFSET + n_frames - 1 (0.2 s frames).
A frame's message is delivered at age 0 or dropped (no queue across frames); the receiver keeps the latest delivered
message up to max_age 3 (ledger.usage, mode 'latest'). Policy 'all' = every message delivered (sync reference).
Usage: python -I c_track.py DATA_ROOT AGENT_DIR DELIV_NPZ OUT
Writes OUT/dets/<policy>/m<seed>/results_nusc.json and OUT/c_track.json (mAP, delivery stats, sha256).
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

TEST0, OFFSET, SEEDS = 96, 10, [0, 1, 2]
HELPERS = (1, 2, 3, 4)


def main():
    root, adir, dpath, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    dl = np.load(dpath)
    policies = sorted({k.split("__")[0] for k in dl.files}) + ["all"]
    db = MiniDB(root)
    frames = L.scene_frames_from_db(db, BP.DEV)
    fidx = {t: k for toks in frames.values() for k, t in enumerate(toks)}
    toks_all = [t for toks in frames.values() for t in toks]
    late = [t for t in toks_all if fidx[t] >= 3]
    pos = BP.uav_positions(root)
    A = {j: load_variant(adir / f"agent{j}" / "results_nusc.json") for j in range(5)}
    msgs = {}
    for j in HELPERS:
        for sc, toks in frames.items():
            trk = K.run_sequence([A[j][t] for t in toks])
            for k, t in enumerate(toks):
                m = select(BP.base(trk[k]), trk[k]["score"] >= BP.TAU_SEND)
                data, st = BC.encode(j, k, pos[(t, j)], m)
                msgs[(j, t)] = BC.decode(data)[1]
    rows = []
    for pol in policies:
        for m in SEEDS:
            use, n_deliv, n_all = {}, 0, 0
            for i, (sc, toks) in enumerate(sorted(frames.items())):
                ep = f"ep{TEST0 + 3 * m + i:04d}"
                T = len(toks)
                packets = []
                for j in HELPERS:
                    if pol == "all":
                        ok = np.ones(T, bool)
                    else:
                        d = dl[f"{pol}__{ep}"][j - 1]
                        assert len(d) >= OFFSET + T, (ep, len(d), T)
                        ok = d[OFFSET:OFFSET + T]
                    n_deliv += int(ok.sum()); n_all += T
                    for s in range(T):
                        packets.append({"scene": sc, "sender": j, "send_frame": s, "seq": s, "drop": not bool(ok[s]),
                                        "delay": 0, "arrival_frame": int(s) if ok[s] else None})
                u = L.usage(packets, T, "latest", 3)
                for t, tok in enumerate(toks):
                    use[tok] = {j: (None if u[j][t][0] is None else toks[u[j][t][0]], u[j][t][1]) for j in HELPERS}
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
            path = out / "dets" / pol / f"m{m}" / "results_nusc.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = json.dumps({"meta": BP.META, "results": preds}).encode()
            path.write_bytes(blob)
            pr = {t: (np.array([r["translation"][:2] for r in v]).reshape(-1, 2),
                      np.array([r["detection_score"] for r in v])) for t, v in preds.items()}
            ra, rl = evaluate(db, pr, toks_all), evaluate(db, pr, late)
            row = dict(policy=pol, seed=m, mAP=round(ra["mAP"], 4), mAP_f3=round(rl["mAP"], 4),
                       deliv_frac=round(n_deliv / n_all, 4), pred=str(path.relative_to(out)),
                       sha256=hashlib.sha256(blob).hexdigest())
            rows.append(row)
            print(f"{pol:10s} m{m} deliv {row['deliv_frac']:.3f} mAP {row['mAP']:.3f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "c_track.json").write_text(json.dumps(dict(test0=TEST0, offset=OFFSET, rows=rows), indent=1))


if __name__ == "__main__":
    main()
