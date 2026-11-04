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
  prog                    progressive message with a deadline (no rate knowledge): during frame s the sender streams
                          its boxes in score order as independently decodable chunks (header 132 bit once + CHUNK
                          boxes of 87 bit + 32 bit chunk CRC/sequence each); at the end of frame s whatever whole chunks
                          were served arrive (age 0), the rest is discarded. No queue carries over between frames.
  prog_novh1              prog, but boxes that a LOWER-index helper also had in its previous-frame message (forecast
                          by 0.2 s, <= 2 m) are moved to the end (one helper keeps each shared car first; causal: needs
                          the other helpers' previous message, e.g. overheard broadcast)
  prog_lv                 learned ordering: logistic-regression estimate of P(box recovers a car UAV0 misses), fitted
                          leave-one-scene-out on the dev scenes (fit on the other two, applied to the held-out one).
                          Causal features: score, distance to UAV0 / to the helper, decoded speed, inside UAV0's square,
                          distance to UAV0's nearest confident detection at the PREVIOUS frame (needs 1-frame-old
                          feedback, not charged), number of other helpers with the car in their previous message
  prog_tp                 same features and LOSO fit, but the label is "box matches ANY GT car inside UAV0's square"
                          (value includes confirmation), ordering by that probability
  prog_gt                 diagnostic ordering: boxes within 2 m of a GT car that UAV0's own confident detections (score
                          >= 0.3) miss at the send frame go first (uses GT; not a policy)
  prog_fb{tau}            prog, but boxes within 2 m of a UAV0 detection with score >= tau at the PREVIOUS frame
                          (receiver already sure; 1-frame-old feedback, not charged) are moved to the end
  oracle_fifo             diagnostic: knows the realised rate of the current frame R_j(s): size = R_j(s) - Q
Usage: python -I queue_proto.py DATA_ROOT AGENT_DIR OUT [--smoke] [--only P1,P2,...]
  --only: run just these policies and merge their rows into an existing OUT/queue_proto.json
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
FB_TAU = [0.6, 0.8]
POLICIES = (["full_fifo", "full_drop"] + [f"cap{c}_fifo" for c in CAPS] + [f"adapt{a}_fifo" for a in ADAPT]
            + [f"est{a}_fifo" for a in ADAPT] + ["prog", "prog_novh1", "prog_lv", "prog_tp", "prog_gt", "oracle_fifo"]
            + [f"prog_fb{x}" for x in FB_TAU])
CHUNK, CHUNK_BITS = 8, 32
HELPERS = (1, 2, 3, 4)
SIGMA = 0.5
MAX_AGE = 3
ONLY = None
if "--only" in sys.argv:
    _k = sys.argv.index("--only")
    ONLY = sys.argv[_k + 1].split(",")
    del sys.argv[_k:_k + 2]
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


def prog_boxes(capacity, n_max):
    """number of boxes delivered by the end of a frame with `capacity` bits when streaming whole chunks."""
    left = capacity - 8 * BC.packet_bytes(0)
    n = 0
    while n < n_max:
        k = min(CHUNK, n_max - n)
        cost = k * BC.BOX_BITS + CHUNK_BITS
        if cost > left:
            break
        left -= cost
        n += k
    return n


def simulate_prog(rate, n_frames, nmax_fn):
    """-> packets (one per frame, arriving at its own frame with n boxes), bits on air per frame, n per frame."""
    pk, served, ns = [], np.zeros(n_frames), []
    for s in range(n_frames):
        nmax = nmax_fn(s)
        n = prog_boxes(rate[s], nmax)
        bits_needed = 8 * BC.packet_bytes(0) + nmax * BC.BOX_BITS + -(-nmax // CHUNK) * CHUNK_BITS
        served[s] = min(rate[s], bits_needed)
        ns.append(n)
        pk.append({"send_frame": s, "size": int(served[s]), "arrival_frame": s if n > 0 else None, "drop": n == 0})
    return pk, served, ns


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
    order = {}
    for sc, toks in frames.items():
        for k, t in enumerate(toks):
            gt = db.gt_xy(t)
            e = A[0][t]
            exy = BP.base(e)["xy"][e["score"] >= 0.3]
            miss = gt[[not (len(exy) and np.linalg.norm(exy - g, axis=1).min() <= 2.0) for g in gt]] if len(gt) else gt
            for j in HELPERS:
                c = cand[(j, t)]
                idx = np.arange(len(c["score"]))
                order[("score", j, t)] = idx
                dup = np.zeros(len(idx), bool)
                if k > 0:
                    for jj in HELPERS:
                        if jj < j:
                            o = cand[(jj, toks[k - 1])]
                            oxy = o["xy"] + o["vel"] * K.DT
                            if len(oxy) and len(idx):
                                dup |= (np.linalg.norm(c["xy"][:, None] - oxy[None], axis=2) <= 2.0).any(1)
                order[("novh1", j, t)] = np.concatenate([idx[~dup], idx[dup]])
                first = np.zeros(len(idx), bool)
                if len(miss) and len(idx):
                    first = (np.linalg.norm(c["xy"][:, None] - miss[None], axis=2) <= 2.0).any(1)
                order[("gt", j, t)] = np.concatenate([idx[first], idx[~first]])
                for tau in FB_TAU:
                    sure = np.zeros(len(idx), bool)
                    if k > 0 and len(idx):
                        pe = A[0][toks[k - 1]]
                        pxy = BP.base(pe)["xy"][pe["score"] >= tau]
                        if len(pxy):
                            sure = (np.linalg.norm(c["xy"][:, None] - pxy[None], axis=2) <= 2.0).any(1)
                    order[(f"fb{tau}", j, t)] = np.concatenate([idx[~sure], idx[sure]])

    # ---- learned value ordering (leave-one-scene-out logistic regression) ----
    feats, labels, labels_tp, keys = [], [], [], []
    for sc, toks in frames.items():
        for k, t in enumerate(toks):
            gt = db.gt_xy(t)
            e = A[0][t]
            exy = BP.base(e)["xy"][e["score"] >= 0.3]
            hit = np.array([len(exy) > 0 and np.linalg.norm(exy - g, axis=1).min() <= 2.0 for g in gt], bool)
            miss = gt[~hit] if len(gt) else gt
            if k > 0:
                pe = A[0][toks[k - 1]]
                pexy = BP.base(pe)["xy"][pe["score"] >= 0.3]
            else:
                pexy = np.zeros((0, 2))
            u0 = db.uav0[t]
            for j in HELPERS:
                c = cand[(j, t)]
                if not len(c["score"]):
                    continue
                d_ego = (np.linalg.norm(c["xy"][:, None] - pexy[None], axis=2).min(1) if len(pexy)
                         else np.full(len(c["score"]), 50.0))
                n_oth = np.zeros(len(c["score"]))
                if k > 0:
                    for jj in HELPERS:
                        if jj != j:
                            o = cand[(jj, toks[k - 1])]
                            if len(o["score"]):
                                oxy = o["xy"] + o["vel"] * K.DT
                                n_oth += (np.linalg.norm(c["xy"][:, None] - oxy[None], axis=2) <= 2.0).any(1)
                dist0 = np.linalg.norm(c["xy"] - u0, axis=1)
                insq = (np.abs(c["xy"] - u0) < 102.4).all(1)
                distj = np.linalg.norm(c["xy"] - pos[(t, j)][:2], axis=1)
                f = np.stack([c["score"], np.minimum(dist0, 150) / 100, np.minimum(distj, 150) / 100,
                              np.linalg.norm(c["vel"], axis=1), insq.astype(float), np.minimum(d_ego, 10.0) / 10,
                              (d_ego <= 2.0).astype(float), n_oth / 3], 1)
                y = ((np.linalg.norm(c["xy"][:, None] - miss[None], axis=2) <= 2.0).any(1) if len(miss)
                     else np.zeros(len(c["score"]), bool))
                ytp = ((np.linalg.norm(c["xy"][:, None] - gt[None], axis=2) <= 2.0).any(1) & insq if len(gt)
                       else np.zeros(len(c["score"]), bool))
                for i in range(len(y)):
                    feats.append(f[i]), labels.append(float(y[i])), labels_tp.append(float(ytp[i]))
                    keys.append((sc, j, t, i))
    X, Y, Y2 = np.array(feats), np.array(labels), np.array(labels_tp)
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    sc_of = np.array([k[0] for k in keys])
    pv, pv2 = np.zeros(len(Y)), np.zeros(len(Y))

    def logreg(Xa, ya, l2=1e-2, it=50):
        w = np.zeros(Xa.shape[1])
        for _ in range(it):                       # Newton / IRLS
            p_ = 1 / (1 + np.exp(-Xa @ w))
            g = Xa.T @ (p_ - ya) + l2 * w
            Hm = (Xa * (p_ * (1 - p_))[:, None]).T @ Xa + l2 * np.eye(len(w))
            w -= np.linalg.solve(Hm, g)
        return w
    lv_info = {}
    for sc in frames:
        tr = sc_of != sc
        w = logreg(X[tr], Y[tr])
        pv[~tr] = 1 / (1 + np.exp(-X[~tr] @ w))
        te = ~tr
        order_te = np.argsort(-pv[te])
        prec = Y[te][order_te][: max(1, int(Y[te].sum()))].mean() if Y[te].sum() else float("nan")
        w2 = logreg(X[tr], Y2[tr])
        pv2[te] = 1 / (1 + np.exp(-X[te] @ w2))
        o2 = np.argsort(-pv2[te])
        o_s = np.argsort(-X[te][:, 0])
        npos2 = max(1, int(Y2[te].sum()))
        lv_info[sc] = {"w": [round(float(v), 3) for v in w], "pos_rate": round(float(Y[te].mean()), 4),
                       "precision_at_npos": round(float(prec), 4), "w_tp": [round(float(v), 3) for v in w2],
                       "tp_rate": round(float(Y2[te].mean()), 4),
                       "tp_precision_at_npos": round(float(Y2[te][o2][:npos2].mean()), 4),
                       "score_precision_at_npos": round(float(Y2[te][o_s][:npos2].mean()), 4)}
    print("learned value ordering (LOSO):", json.dumps(lv_info), flush=True)
    for name, pvx in (("lv", pv), ("tp", pv2)):
        val = {}
        for (sc, j, t, i), v in zip(keys, pvx):
            val.setdefault((j, t), {})[i] = v
        for j in HELPERS:
            for t in toks_all:
                n = len(cand[(j, t)]["score"])
                v = np.array([val.get((j, t), {}).get(i, 0.0) for i in range(n)])
                order[(name, j, t)] = np.argsort(-v, kind="stable")

    def message(j, t, n, ok="score"):
        key = (j, t, n, ok)
        if key not in enc:
            m = select(cand[(j, t)], order[(ok, j, t)][:n])
            data, st = BC.encode(j, fidx[t], pos[(t, j)], m)
            enc[key] = (BC.decode(data)[1], 8 * st["bytes"])
        return enc[key]

    rows = []
    for R in RATES:
        Rb = R * 1000
        for kind, seed in CHANNELS:
            chan = f"{kind}_s{seed}"
            for policy in (ONLY or POLICIES):
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
                        if policy.startswith("prog"):
                            pk, served, ns = simulate_prog(rate, T, lambda s, j=j: len(cand[(j, toks[s])]["score"]))
                            for s, n in enumerate(ns):
                                chosen[s] = n
                        else:
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
                        ok = {"prog_novh1": "novh1", "prog_gt": "gt", "prog_lv": "lv", "prog_tp": "tp"}.get(
                            policy, policy[5:] if policy.startswith("prog_fb") else "score")
                        m = K.forecast(message(j, src, n, ok)[0], age)
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
    if ONLY:
        old = json.loads((out / "queue_proto.json").read_text())["rows"]
        rows = [r for r in old if r["policy"] not in ONLY] + rows
    (out / "queue_proto.json").write_text(json.dumps({"rates": RATES, "sigma": SIGMA, "rho": RHO, "learned_value": lv_info,
                                                      "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
