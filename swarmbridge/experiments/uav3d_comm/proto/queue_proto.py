"""Budgeted selection, step 2: payload-dependent delay through a per-link queue (protocol v1.1 section 10.6).
Dev scenes only; CPU only.

Unlike ledger.py (exogenous delay), here the message SIZE decides when it arrives: a large message takes longer to
serve and also delays the messages behind it. This is the minimal model in which "what to send" and "when it
arrives" interact.

Link model (each helper j -> UAV0, independent):
  rate      R_j(f) bit served during frame f.  "const": R_j(f) = R.  "fade": R_j(f) = R * exp(N(0, 0.5^2) - 0.125)
            (log-normal, mean R), i.i.d. per link and frame, seeded per (seed, scene, sender) - the same realisation
            for every policy.  "ar": the log-rate follows an AR(1) process with correlation RHO per frame (stationary
            N(0, 0.5^2), mean rate R) - a channel whose next-frame rate is partly predictable from the last one.
  service   FIFO, work-conserving: a packet enqueued at the start of frame s arrives at the first frame f >= s at which
            the cumulative service from frame s covers the backlog in front of it plus its own size (delay 0 if it
            fits in frame s). Bits on air = bits actually served.
  drop      "fifo": never drops.  "drop": when a new packet is enqueued, every packet still waiting (not completely
            served) is discarded; partially served bits are still counted as transmitted (preemptive, AoI-style).
Receiver: newest packet (by send frame) that has arrived, max age 3, cold start (ledger.usage), KF-PV forecast by age,
noisy-OR fusion, cap 500 (identical to box_proto kf_nor).

Message content: each helper's KF-PV candidates (box_proto kf family), highest score first; a packet always has the
132-bit header; size = 8 * bytes. Size policies (n = number of boxes):
  full_fifo / full_drop   all candidates (as in box_proto), FIFO / preemptive queue
  cap{c}_fifo             at most c * R bits per packet, c in CAPS
  adapt{a}_fifo           sender knows its own backlog Q (bits still waiting at the start of the frame) and the MEAN
                          rate R, not the realised rate: size = a * R - Q (no packet if not even one box fits), a in
                          ADAPT; a = 1 targets finishing in the current frame on average (causal, local information)
  est{a}_fifo             as adapt{a}, but the sender estimates the rate by the realised rate of the PREVIOUS frame
                          (known from link-layer ACKs; R for the first frame): size = a * R_j(s-1) - Q (causal)
  oracle_fifo             diagnostic: knows the realised rate of the current frame R_j(s): size = R_j(s) - Q
Usage: python -I queue_proto.py DATA_ROOT AGENT_DIR OUT [--smoke]
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

RATES = [4, 8]                              # kbit per frame per link (= 20, 40 kbit/s at 0.2 s/frame)
CHANNELS = [("const", 0), ("fade", 0), ("fade", 1), ("fade", 2), ("ar", 0), ("ar", 1), ("ar", 2)]
RHO = 0.8
CAPS = [0.25, 0.5, 0.75, 1.0, 2.0]
ADAPT = [0.5, 0.75, 1.0]           # adapt{a}: size = a * R - Q
POLICIES = (["full_fifo", "full_drop"] + [f"cap{c}_fifo" for c in CAPS] + [f"adapt{a}_fifo" for a in ADAPT]
            + [f"est{a}_fifo" for a in ADAPT] + ["oracle_fifo"])
HELPERS = (1, 2, 3, 4)
SIGMA = 0.5
MAX_AGE = 3
if "--smoke" in sys.argv:
    sys.argv.remove("--smoke")
    RATES, CHANNELS = [8], [("fade", 0)]


def rate_series(kind, seed, scene, j, n, R):
    if kind == "const":
        return np.full(n, float(R))
    key = hashlib.sha256(f"{seed}|{scene}|{j}|rate".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    if kind == "fade":
        return R * np.exp(rng.normal(0, SIGMA, n) - SIGMA ** 2 / 2)
    assert kind == "ar", kind
    e = rng.normal(0, SIGMA, n)
    x = np.zeros(n)
    x[0] = e[0]
    for f in range(1, n):
        x[f] = RHO * x[f - 1] + np.sqrt(1 - RHO ** 2) * e[f]
    return R * np.exp(x - SIGMA ** 2 / 2)


def n_fit(bits, n_max):
    """largest n <= n_max with 8 * packet_bytes(n) <= bits (0 if even the header does not fit)."""
    n = max(0, int((bits - BC.HEADER_BITS) // BC.BOX_BITS))
    n = min(n, n_max)
    while n > 0 and 8 * BC.packet_bytes(n) > bits:
        n -= 1
    return n


def simulate(sizes_fn, rate, n_frames, drop):
    """sizes_fn(s, backlog_bits) -> packet bits for the packet sent at frame s.
    Returns packets [{send_frame, size, arrival_frame or None, drop}], bits served per frame."""
    queue = []                     # [packet index, remaining bits]
    pk, served = [], np.zeros(n_frames)
    for s in range(n_frames):
        backlog = sum(r for _, r in queue)
        size = sizes_fn(s, backlog)
        if size == 0:              # nothing worth sending this frame: no packet (the receiver keeps its cache)
            pk.append({"send_frame": s, "size": 0, "arrival_frame": None, "drop": True})
        elif drop:
            for i, _ in queue:
                pk[i]["drop"] = True
            queue = []
        if size > 0:
            pk.append({"send_frame": s, "size": size, "arrival_frame": None, "drop": False})
            queue.append([len(pk) - 1, float(size)])
        cap = rate[s]
        while queue and cap > 0:
            take = min(cap, queue[0][1])
            queue[0][1] -= take
            cap -= take
            served[s] += take
            if queue[0][1] <= 1e-9:
                pk[queue[0][0]]["arrival_frame"] = s
                queue.pop(0)
    return pk, served              # packets still queued at the end keep arrival_frame None (episode tail)


def main():
    root, adir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    db = MiniDB(root)
    frames = L.scene_frames_from_db(db, BP.DEV)
    fidx = {t: k for toks in frames.values() for k, t in enumerate(toks)}
    toks_all = [t for toks in frames.values() for t in toks]
    late = [t for t in toks_all if fidx[t] >= 3]
    pos = BP.uav_positions(root)
    A = {j: load_variant(adir / f"agent{j}" / "results_nusc.json") for j in range(5)}
    cand = {}
    for j in HELPERS:
        for sc, toks in frames.items():
            trk = K.run_sequence([A[j][t] for t in toks])
            for k, t in enumerate(toks):
                c = select(BP.base(trk[k]), trk[k]["score"] >= BP.TAU_SEND)
                cand[(j, t)] = select(c, np.argsort(-c["score"]))   # highest score first
    enc = {}

    def message(j, t, n):
        key = (j, t, n)
        if key not in enc:
            m = select(cand[(j, t)], np.arange(n))
            data, st = BC.encode(j, fidx[t], pos[(t, j)], m)
            enc[key] = (BC.decode(data)[1], 8 * st["bytes"])
        return enc[key]

    rows = []
    for R in RATES:
        Rb = R * 1000
        for kind, seed in CHANNELS:
            chan = f"{kind}_s{seed}"
            for policy in POLICIES:
                drop = policy.endswith("_drop")
                use, tx, sizes, ages, nmsg, n_none, n_links, n_tail, n_skip = {}, [], [], [], [], 0, 0, 0, 0
                for sc, toks in frames.items():
                    T = len(toks)
                    pk_all = []
                    for j in HELPERS:
                        rate = rate_series(kind, seed, sc, j, T, Rb)
                        chosen = {}

                        def sizes_fn(s, backlog, j=j, rate=rate, chosen=chosen):
                            t = toks[s]
                            nmax = len(cand[(j, t)]["score"])
                            if policy.startswith("full"):
                                n = nmax
                            elif policy.startswith("cap"):
                                n = n_fit(float(policy[3:].split("_")[0]) * Rb, nmax)
                            elif policy.startswith("adapt"):
                                n = n_fit(float(policy[5:].split("_")[0]) * Rb - backlog, nmax)
                            elif policy.startswith("est"):
                                rhat = rate[s - 1] if s > 0 else Rb
                                n = n_fit(float(policy[3:].split("_")[0]) * rhat - backlog, nmax)
                            elif policy == "oracle_fifo":
                                n = n_fit(rate[s] - backlog, nmax)
                            else:
                                raise ValueError(policy)
                            chosen[s] = n
                            return message(j, t, n)[1] if n > 0 else 0
                        pk, served = simulate(sizes_fn, rate, T, drop)
                        n_tail += sum(p["arrival_frame"] is None and not p["drop"] for p in pk)
                        n_skip += sum(p["size"] == 0 for p in pk)
                        tx.append(served)
                        for p in pk:
                            p.update(sender=j, scene=sc, n=chosen[p["send_frame"]])
                            if p["arrival_frame"] is None:
                                p["drop"] = True        # never arrived (preempted or still queued at the end)
                            if p["size"] > 0:
                                sizes.append(p["size"])
                                nmsg.append(p["n"])
                        pk_all += pk
                    u = L.usage(pk_all, T, "latest", MAX_AGE)
                    nby = {(p["sender"], p["send_frame"]): p["n"] for p in pk_all}
                    for k, t in enumerate(toks):
                        use[t] = {}
                        for j in HELPERS:
                            s, age = u[j][k]
                            n_links += 1
                            if s is None:
                                n_none += 1
                                use[t][j] = None
                            else:
                                ages.append(age)
                                use[t][j] = (toks[s], age, nby[(j, s)])
                preds = {}
                for t in toks_all:
                    own = BP.base(A[0][t])
                    sets = [(0, own["xy"], own["score"], BP.records(own, t))]
                    for j in HELPERS:
                        if use[t][j] is None:
                            continue
                        src, age, n = use[t][j]
                        m = K.forecast(message(j, src, n)[0], age)
                        sets.append((j, m["xy"], m["score"], BP.records(m, t)))
                    recs = soft_fuse(sets, "noisyor", return_records=True)
                    recs.sort(key=lambda r: -r["detection_score"])
                    preds[t] = [dict(r, sample_token=t) for r in recs[:BP.CAP]]
                tag = f"{policy}_R{R}"
                path = out / "dets" / tag / chan / "results_nusc.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                blob = json.dumps({"meta": BP.META, "results": preds}).encode()
                path.write_bytes(blob)
                pr = {t: (np.array([r["translation"][:2] for r in v]).reshape(-1, 2),
                          np.array([r["detection_score"] for r in v])) for t, v in preds.items()}
                r_all, r_late = evaluate(db, pr, toks_all), evaluate(db, pr, late)
                txf = np.array(tx).reshape(len(frames), len(HELPERS), -1).sum(1)   # bits on air per receiver frame
                row = {"policy": policy, "rate_kbit": R, "channel": chan, "mAP": round(r_all["mAP"], 4),
                       "mAP_f3": round(r_late["mAP"], 4), "kbit_onair_mean": round(float(txf.mean()) / 1000, 3),
                       "pkt_kbit_mean": round(float(np.mean(sizes)) / 1000, 3),
                       "boxes_per_pkt": round(float(np.mean(nmsg)), 1), "frac_no_msg": round(n_none / n_links, 4),
                       "age_hist": [int(sum(a == d for a in ages)) for d in range(MAX_AGE + 1)],
                       "mean_age": round(float(np.mean(ages)), 3) if ages else None, "n_tail_packets": n_tail, "n_frames_not_sent": n_skip,
                       "pred": str(path.relative_to(out)), "sha256": hashlib.sha256(blob).hexdigest()}
                rows.append(row)
                print(f"{tag:16s} {chan:8s} mAP {row['mAP']:.3f} (f>=3 {row['mAP_f3']:.3f}) air "
                      f"{row['kbit_onair_mean']:5.2f} kbit/frame pkt {row['pkt_kbit_mean']:5.2f} kbit "
                      f"boxes {row['boxes_per_pkt']:5.1f} nomsg {row['frac_no_msg']:.2f} age {row['mean_age']} "
                      f"hist {row['age_hist']}", flush=True)
    (out / "queue_proto.json").write_text(json.dumps({"rates": RATES, "sigma": SIGMA, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
