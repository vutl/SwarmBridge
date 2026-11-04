"""Box-level communication under protocol v1 (dev scenes only; CPU only).

Pipeline per (method, ledger):
  sender j at send frame s : message = its own detections (score >= TAU_SEND)            [box_*]
                             or its own CV-Kalman confirmed tracks (score >= TAU_SEND)     [kf_*]
                             -> boxcodec.encode -> bytes (counted for EVERY transmitted packet, lost or not)
  channel                  : causal ledger (ledger.py): receiver at t uses the newest packet that arrived by t
  receiver UAV0 at t       : own detections (all, cap-500 head) + boxcodec.decode(packet); kf_* forecast each
                             received track by age * 0.2 s with the sender's velocity estimate
  fusion                   : max   = centre-NMS keeping the most confident copy (radius 2 m)
                             nor   = score-weighted centre, noisy-OR score over UAVs (soft_fusion.fuse)
  output cap               : keep the 500 highest-scoring fused boxes per frame
Writes OUT/dets/<method>/<ledger>/results_nusc.json and OUT/box_proto.json with, per row: mAP (all frames and
frames >= 3 = enough history), per-scene mAP, sha256 of the prediction file, transmitted/received kbit per frame
(mean, p95), and codec saturation counts.

Usage: python -I box_proto.py DATA_ROOT AGENT_DIR OUT
  AGENT_DIR/agent{0..4}/results_nusc.json = each UAV's own detections from the SAME fine-tuned decoder (cap 500).
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
import kalman as K  # noqa: E402
import ledger as L  # noqa: E402
from evaldet import MiniDB, evaluate  # noqa: E402
from fusion import load_variant, late_fuse, select  # noqa: E402
from soft_fusion import fuse as soft_fuse  # noqa: E402

DEV = ["scene_11", "scene_13", "scene_15"]
TAU_SEND = 0.3
CAP = 500
LEDGERS = [("sync", 0), ("fix1", 0), ("fix2", 0)] + [(sc, s) for sc in ("rand", "loss0.2", "loss0.5", "rand_loss0.2")
                                                     for s in (0, 1, 2)]
METHODS = ["box_max", "box_nor", "kf_max", "kf_nor"]
META = {"use_camera": True, "use_lidar": False, "use_radar": False, "use_map": False, "use_external": False}


def uav_positions(root):
    meta = Path(root) / "v1.0-mini"
    cals = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    pos = {}
    for d in json.loads((meta / "sample_data.json").read_text()):
        f = d["filename"]
        if "CAMERA_BOTTOM_id_" in f:
            j = int(f.split("CAMERA_BOTTOM_id_")[1][0])
            pos[(d["sample_token"], j)] = np.array(cals[d["calibrated_sensor_token"]]["translation"])
    return pos


def records(d, tok):
    out = []
    for i in range(len(d["score"])):
        y = float(d["yaw"][i])
        out.append({"sample_token": tok, "translation": [float(d["xy"][i, 0]), float(d["xy"][i, 1]), float(d["z"][i])],
                    "size": [float(x) for x in d["size"][i]], "rotation": [float(np.cos(y / 2)), 0.0, 0.0,
                                                                          float(np.sin(y / 2))],
                    "velocity": [float(d["vel"][i, 0]), float(d["vel"][i, 1])], "detection_name": "car",
                    "detection_score": float(d["score"][i]), "attribute_name": ""})
    return out


def base(d):
    return {k: d[k] for k in ("xy", "z", "size", "yaw", "score", "vel")}


def main():
    root, adir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    db = MiniDB(root)
    frames = L.scene_frames_from_db(db, DEV)
    fidx = {t: k for toks in frames.values() for k, t in enumerate(toks)}
    scene_of = {t: sc for sc, toks in frames.items() for t in toks}
    toks_all = [t for toks in frames.values() for t in toks]
    late = [t for t in toks_all if fidx[t] >= 3]
    pos = uav_positions(root)
    A = {j: load_variant(adir / f"agent{j}" / "results_nusc.json") for j in range(5)}

    # ---- sender side: per (method family, sender, send token) encode once; keep bytes and decoded message ----
    msgs = {"box": {}, "kf": {}}
    for j in range(1, 5):
        for sc, toks in frames.items():
            trk = K.run_sequence([A[j][t] for t in toks])
            for k, t in enumerate(toks):
                for fam, m in (("box", base(A[j][t])), ("kf", trk[k])):
                    m = select(base(m), m["score"] >= TAU_SEND)
                    data, st = BC.encode(j, k, pos[(t, j)], m)
                    _, dec = BC.decode(data)
                    msgs[fam][(j, t)] = (dec, st)
    sat = {fam: int(sum(sum(st["saturated"].values()) for _, st in v.values())) for fam, v in msgs.items()}

    ledgers = {}
    (out / "ledgers").mkdir(parents=True, exist_ok=True)
    for sc, seed in LEDGERS:
        name = f"{sc}_s{seed}"
        p = out / "ledgers" / f"{name}.json"
        led = L.build(frames, sc, seed)
        if p.exists():
            assert L.load(p)["sha256"] == led["sha256"], f"ledger changed: {p}"
        else:
            p.write_text(json.dumps(led))
        ledgers[name] = led

    rows = []

    def score(name, method, preds_full, kb_tx, kb_rx):
        path = out / "dets" / method / name / "results_nusc.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps({"meta": META, "results": preds_full}).encode()
        path.write_bytes(blob)
        pr = {t: (np.array([r["translation"][:2] for r in v]).reshape(-1, 2),
                  np.array([r["detection_score"] for r in v])) for t, v in preds_full.items()}
        r_all, r_late = evaluate(db, pr, toks_all), evaluate(db, pr, late)
        per_scene = {sc: round(evaluate(db, pr, toks)["mAP"], 4) for sc, toks in frames.items()}
        row = {"method": method, "ledger": name, "mAP": round(r_all["mAP"], 4), "mAP_f3": round(r_late["mAP"], 4),
               "ap": [round(r_all[f"ap_{k}"], 4) for k in (0.5, 1.0, 2.0, 4.0)], "per_scene": per_scene,
               "kbit_tx_mean": round(float(np.mean(kb_tx)), 2) if kb_tx else 0.0,
               "kbit_tx_p95": round(float(np.percentile(kb_tx, 95)), 2) if kb_tx else 0.0,
               "kbit_rx_mean": round(float(np.mean(kb_rx)), 2) if kb_rx else 0.0,
               "pred": str(path.relative_to(out)), "sha256": hashlib.sha256(blob).hexdigest()}
        rows.append(row)
        print(f"{method:8s} {name:16s} mAP {row['mAP']:.3f} (f>=3 {row['mAP_f3']:.3f}) scenes "
              f"{'/'.join(f'{v:.3f}' for v in per_scene.values())}  tx {row['kbit_tx_mean']:6.1f} kbit/frame",
              flush=True)

    # ego only (no communication, 0 bit)
    ego = {}
    for t in toks_all:
        d = base(A[0][t])
        keep = np.argsort(-d["score"])[:CAP]
        ego[t] = records(select(d, keep), t)
    score("none", "ego", ego, [], [])

    for name, led in ledgers.items():
        use = L.lookup(led)
        tx = L.tx_packets(led)
        for method in METHODS:
            fam, rule = method.split("_")
            # transmitted bits per receiver frame: all packets sent in that frame by the 4 helpers, lost or not
            kb_tx = []
            for sc, toks in frames.items():
                for k, t in enumerate(toks):
                    kb_tx.append(sum(msgs[fam][(j, t)][1]["bits"] for j in range(1, 5) if (sc, j, k) in tx) / 1000)
            preds, kb_rx = {}, []
            for t in toks_all:
                own = base(A[0][t])
                recv, bits = [], 0
                for j in range(1, 5):
                    src, age = use[t][j]
                    if src is None:
                        continue
                    dec, st = msgs[fam][(j, src)]
                    bits += st["bits"]
                    recv.append((j, K.forecast(dec, age) if fam == "kf" else dec))
                kb_rx.append(bits / 1000)
                if rule == "max":
                    f = late_fuse(own, [m for _, m in recv]) if recv else own
                else:
                    allm = [(0, own)] + recv
                    sets = [(j, m["xy"], m["score"], records(m, t)) for j, m in allm]
                    recs = soft_fuse(sets, "noisyor", return_records=True)
                    f = None
                if f is not None:
                    keep = np.argsort(-f["score"])[:CAP]
                    preds[t] = records(select(f, keep), t)
                else:
                    recs.sort(key=lambda r: -r["detection_score"])
                    preds[t] = [dict(r, sample_token=t) for r in recs[:CAP]]
            score(name, method, preds, kb_tx, kb_rx)

    (out / "box_proto.json").write_text(json.dumps({"tau_send": TAU_SEND, "cap": CAP, "codec_saturated": sat,
                                                    "rows": rows}, indent=1))
    print("codec saturated fields:", sat)


if __name__ == "__main__":
    main()
