"""Methodological audit of H1 Gate 0A (exploratory correction; EXPL = MDMT-test public labels only).

Requested by review #6. Does NOT replace runs/uav_semcom/h1/gate0a_v21.json; writes gate0a_audit_v1.json.
1. Exact oracles: S0-exact (TOK: enumerate per-UAV token counts; CROP: MILP with per-UAV packetisation)
   versus the heuristic S0 of h1_gate0a.py (pre-charged headers / greedy).
2. Deployable cost of S2: relabelled "lagged-identity oracle". Manifest variants S2m(m): each UAV uploads a
   manifest of m bits per current track (association itself still ideal), BS returns a bitmap.
3. Strong no-peer baseline in a temporal mode: each UAV remembers what it delivered (perfect ACK) and only
   resends objects whose BS box is stale (IoU < 0.5) in its own view. Coordination then only removes
   cross-view resends. Myopic per-epoch oracle S0-hist (exact per epoch, not a dynamic optimum).
4. Per-sequence variation with fixed schemes (no best-of selection on EXPL).
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import h1_gate0a as G

HIST_TOK = [100, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000]
HIST_CROP = [5e3, 10e3, 20e3, 50e3, 100e3, 200e3, 500e3]


# ---------------------------------------------------------------- exact oracles
def tok_cost(n):
    return 0 if n == 0 else n * G.TOK_BITS + G.msg_overhead(n * G.TOK_BITS)


def exact_tok(nA, nB, U, budget):
    """max distinct objects; UAV1 may send n1 <= nA, UAV2 n2 <= nB; distinct = min(n1 + n2, U)
    (UAV1 takes its exclusive objects first, UAV2 avoids UAV1's picks)."""
    best = 0
    for n1 in range(nA + 1):
        c1 = tok_cost(n1)
        if c1 > budget:
            break
        rest = budget - c1
        n2 = G.tok_capacity(rest, nB) if rest > 0 else 0
        best = max(best, min(n1 + n2, U))
        if best == U:
            break
    return best


def exact_crop(A, B, cA, cB, budget, return_sel=False):
    """MILP: x1_i (i in A), x2_j (j in B) binary; x1_k + x2_k <= 1 on shared; per-UAV packets p_v integer with
    payload_v + 48 s_v <= 9600 p_v; total payload + 48(s1+s2) + 96(p1+p2) <= budget. Maximise sum x."""
    A, B = sorted(A), sorted(B)
    if budget <= 0 or not (A or B):
        return (0, []) if return_sel else 0
    nA, nB = len(A), len(B)
    n = nA + nB + 4                   # x1.., x2.., s1, s2, p1, p2
    iS1, iS2, iP1, iP2 = nA + nB, nA + nB + 1, nA + nB + 2, nA + nB + 3
    c = np.zeros(n); c[:nA + nB] = -1
    rows, lo, hi = [], [], []
    idxB = {j: nA + k for k, j in enumerate(B)}
    for k, i in enumerate(A):
        if i in idxB:
            r = np.zeros(n); r[k] = 1; r[idxB[i]] = 1; rows.append(r); lo.append(-np.inf); hi.append(1)
    r = np.zeros(n); r[:nA] = [cA[i] for i in A]; r[iS1] = G.MSG_HDR; r[iP1] = -G.MTU
    rows.append(r); lo.append(-np.inf); hi.append(0)
    r = np.zeros(n); r[nA:nA + nB] = [cB[j] for j in B]; r[iS2] = G.MSG_HDR; r[iP2] = -G.MTU
    rows.append(r); lo.append(-np.inf); hi.append(0)
    for k in range(nA):
        r = np.zeros(n); r[k] = 1; r[iS1] = -1; rows.append(r); lo.append(-np.inf); hi.append(0)
    for k in range(nB):
        r = np.zeros(n); r[nA + k] = 1; r[iS2] = -1; rows.append(r); lo.append(-np.inf); hi.append(0)
    r = np.zeros(n); r[:nA] = [cA[i] for i in A]; r[nA:nA + nB] = [cB[j] for j in B]
    r[iS1] = r[iS2] = G.MSG_HDR; r[iP1] = r[iP2] = G.APP_HDR
    rows.append(r); lo.append(-np.inf); hi.append(budget)
    pmax = budget / G.MTU + 2
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.r_[np.ones(nA + nB + 2), pmax, pmax]),
               options=dict(time_limit=30))
    assert res.status == 0, res.message
    if return_sel:
        x = np.round(res.x).astype(int)
        sel = [(1, i) for k, i in enumerate(A) if x[k]] + [(2, j) for k, j in enumerate(B) if x[nA + k]]
        return int(round(-res.fun)), sel
    return int(round(-res.fun))


def heuristic_s0_tok(A, B, budget):
    return G.tok_coordinated(A, B, A & B, budget, 0.0, 1.0)["cov"]


def heuristic_s0_crop(A, B, cA, cB, budget):
    return G.crop_coordinated(A, B, cA, cB, set(), budget, 0.0, 1.0, oracle=True)["cov"]


# ---------------------------------------------------------------- geometry
def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


# ---------------------------------------------------------------- snapshot audit
def snapshot_audit(data, stride=10, bpp=0.65, c0=2400.0):
    out = {"TOK": defaultdict(lambda: defaultdict(float)), "CROP": defaultdict(lambda: defaultdict(float))}
    per_seq = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    diff_ep = {"TOK": defaultdict(int), "CROP": defaultdict(int)}
    n_ep = 0
    for seq, f, A, B, Abox, Bbox, prev, first in G.epochs(data, stride):
        n_ep += 1
        U = len(A | B)
        cA = {i: G.crop_bits(b, bpp, c0) for i, b in Abox.items()}
        cB = {i: G.crop_bits(b, bpp, c0) for i, b in Bbox.items()}
        C = A & B
        sch = G.schemes(seq, f, A, B, Bbox, prev, {"8x8": set(), "16x9": set()})
        for br, budgets in (("TOK", G.TOK_BUDGETS), ("CROP", G.CROP_BUDGETS)):
            for Bg in budgets:
                acc = out[br][Bg]
                acc["U"] += U
                if br == "TOK":
                    h = heuristic_s0_tok(A, B, Bg); e = exact_tok(len(A), len(B), U, Bg)
                else:
                    h = heuristic_s0_crop(A, B, cA, cB, Bg); e = exact_crop(A, B, cA, cB, Bg)
                assert e >= h - 1e-6, (seq, f, br, Bg, e, h)
                diff_ep[br][Bg] += e > h + 1e-6
                acc["S0_heur"] += h; acc["S0_exact"] += e
                # independent baselines (same as 0A)
                for a in G.ALPHAS:
                    r = G.tok_independent(A, B, a * Bg, (1 - a) * Bg) if br == "TOK" else \
                        G.crop_independent(A, B, cA, cB, a * Bg, (1 - a) * Bg)
                    acc[f"IND_a{a}"] += r["cov"]
                share = len(A) / max(len(A) + len(B), 1); qb = Bg - G.QUOTA_BITS
                r = G.tok_independent(A, B, share * qb, (1 - share) * qb) if br == "TOK" else \
                    G.crop_independent(A, B, cA, cB, share * qb, (1 - share) * qb)
                acc["Q"] += r["cov"]
                # lagged-identity oracle S2 (as in 0A)
                sup, dl = sch["S2"]
                r = G.tok_coordinated(A, B, sup, Bg, dl, 1.0) if br == "TOK" else \
                    G.crop_coordinated(A, B, cA, cB, sup, Bg, dl, 1.0)
                acc["S2_lagID"] += r["cov"]
                per_seq[seq][(br, Bg)]["S2_lagID"] += r["cov"]
                # manifest variants: association ideal but its uplink cost is charged
                for m in (24, 52):
                    if br == "TOK":       # UAV1 tokens double as its manifest; UAV2 uploads an m-bit manifest
                        ul = m * len(B) + (G.msg_overhead(m * len(B)) if B else 0)
                        r = G.tok_coordinated(A, B, C, Bg - ul, len(B) + G.DL_HDR, 1.0)
                    else:                 # both UAVs upload manifests; crops follow
                        ul = sum(m * len(S) + (G.msg_overhead(m * len(S)) if S else 0) for S in (A, B))
                        r = G.crop_coordinated(A, B, cA, cB, C, Bg - ul, len(B) + G.DL_HDR, 1.0)
                    acc[f"S2m{m}"] += r["cov"]
                    per_seq[seq][(br, Bg)][f"S2m{m}"] += r["cov"]
                ps = per_seq[seq][(br, Bg)]
                ps["U"] += U; ps["S0_exact"] += e
                ps["IND_a0.5"] += (G.tok_independent(A, B, .5 * Bg, .5 * Bg) if br == "TOK"
                                   else G.crop_independent(A, B, cA, cB, .5 * Bg, .5 * Bg))["cov"]
    return out, per_seq, diff_ep, n_ep


# ---------------------------------------------------------------- temporal (history) mode
def _hist_run(data, br, Bg, pol, alpha, rng, stride, bpp, c0):
    """one fixed policy over all sequences; returns (covered, U, payload bits)."""
    cov_sum = U_sum = bits = 0.0
    for seq in sorted(data):
        v1, v2 = data[seq][1], data[seq][2]
        last = {1: {}, 2: {}}
        for f in sorted(set(v1) | set(v2))[::stride]:
            boxes = {1: v1.get(f, {}), 2: v2.get(f, {})}
            A, B = set(boxes[1]), set(boxes[2])
            allo = A | B
            ok = lambda v, i: i in last[v] and iou(last[v][i], boxes[v][i]) >= 0.5
            covered0 = {i for i in allo if (i in A and ok(1, i)) or (i in B and ok(2, i))}
            stale = {1: {i for i in A if not ok(1, i)}, 2: {i for i in B if not ok(2, i)}}
            need = allo - covered0
            cost = {v: ({i: G.TOK_BITS for i in boxes[v]} if br == "TOK" else
                        {i: G.crop_bits(b, bpp, c0) for i, b in boxes[v].items()}) for v in (1, 2)}
            sent = {1: set(), 2: set()}
            if pol in ("IND", "SEND_STALE_ALL"):
                bud = {1: alpha * Bg, 2: (1 - alpha) * Bg} if pol == "IND" else {1: 1e15, 2: 1e15}
                for v in (1, 2):
                    # own-history priority: never delivered by me first, then stale; random (TOK) / cheapest (CROP)
                    cand = sorted(stale[v], key=lambda i: (i in last[v], rng.random() if br == "TOK" else cost[v][i]))
                    pay = 0.0
                    for i in cand:
                        if pay + cost[v][i] + G.msg_overhead(pay + cost[v][i]) <= bud[v]:
                            pay += cost[v][i]; sent[v].add(i)
            else:
                nA_, nB_ = need & A, need & B
                avail = Bg
                if pol != "S0":
                    m = int(pol[3:])
                    avail -= m * len(B) + (G.msg_overhead(m * len(B)) if B else 0) + len(B) + G.DL_HDR
                avail = max(avail, 0)
                if br == "TOK":
                    k = exact_tok(len(nA_), len(nB_), len(need), avail)
                    order1 = sorted(nA_ - nB_) + sorted(nA_ & nB_)
                    for n1 in range(len(nA_) + 1):
                        rest = avail - tok_cost(n1)
                        if rest < 0:
                            break
                        n2 = G.tok_capacity(rest, len(nB_)) if rest > 0 else 0
                        if min(n1 + n2, len(need)) == k:
                            sent[1] = set(order1[:n1])
                            rem = sorted(nB_ - sent[1])
                            sent[2] = set(rem[:G.tok_capacity(rest, len(rem)) if rest > 0 else 0])
                            break
                else:
                    _, sel = exact_crop(nA_, nB_, cost[1], cost[2], avail, return_sel=True)
                    for v, i in sel:
                        sent[v].add(i)
            for v in (1, 2):
                for i in sent[v]:
                    last[v][i] = boxes[v][i]
                bits += sum(cost[v][i] for i in sent[v])
            cov_sum += len(covered0 | sent[1] | sent[2]); U_sum += len(allo)
    return cov_sum, U_sum, bits


def history_audit(data, stride=10, bpp=0.65, c0=2400.0, seeds=(0, 1, 2, 3, 4)):
    """Temporal mode. BS keeps the last delivered box per (view, id); an object is covered if, in some view where
    it is visible, that box has IoU >= 0.5 with the current GT box. Senders know their own deliveries (ACK).
    IND_a: fixed budget split alpha for all epochs (no per-epoch selection). S0: per-epoch exact (myopic)
    coordinated choice over objects not covered. S2m{m}: same choice but UAV2 manifest (m bits/track, UL) and
    bitmap (DL, rho = 1) are charged; association ideal."""
    n_ep = sum(len(sorted(set(data[s][1]) | set(data[s][2]))[::stride]) for s in data)
    res = {}
    for br, budgets in (("TOK", HIST_TOK), ("CROP", HIST_CROP)):
        c, U, bits = _hist_run(data, br, 0, "SEND_STALE_ALL", 0.5, random.Random(0), stride, bpp, c0)
        stream = bits / n_ep
        for Bg in budgets:
            r = {"stale_stream_bits_per_epoch_no_budget": stream}
            for a in G.ALPHAS:
                sds = seeds if br == "TOK" else (0,)
                r[f"IND_a{a}"] = 100 * np.mean([(lambda t: t[0] / t[1])(_hist_run(data, br, Bg, "IND", a, random.Random(sd),
                                                                             stride, bpp, c0)) for sd in sds])
            for pol in ("S0", "S2m24", "S2m52"):
                cv, U, _ = _hist_run(data, br, Bg, pol, 0.5, random.Random(0), stride, bpp, c0)
                r[pol] = 100 * cv / U
            r["IND_best"] = max(r[f"IND_a{a}"] for a in G.ALPHAS)
            res[(br, Bg)] = r
            print("hist", br, Bg, {k: round(v, 1) for k, v in r.items()}, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("out")
    a = ap.parse_args()
    data = G.load(a.labels)
    out, per_seq, diff_ep, n_ep = snapshot_audit(data)
    snap = {}
    for br in ("TOK", "CROP"):
        snap[br] = []
        for Bg, acc in out[br].items():
            U = acc["U"]
            rec = {k: 100 * v / U for k, v in acc.items() if k != "U"}
            base = max([rec[f"IND_a{x}"] for x in G.ALPHAS] + [rec["Q"]])
            seqd = {}
            # consistency guard: per-sequence sums must reproduce the pooled sums
            for k in ("U", "S0_exact", "S2_lagID", "S2m24", "S2m52", "IND_a0.5"):
                tot = sum(per_seq[s][(br, Bg)][k] for s in per_seq)
                ref = acc["U"] if k == "U" else acc[k]
                assert abs(tot - ref) < 1e-6 * max(1.0, ref), (br, Bg, k, tot, ref)
            for s in per_seq:
                d = per_seq[s][(br, Bg)]
                b5 = 100 * d["IND_a0.5"] / d["U"]
                seqd[s] = {k: 100 * d[k] / d["U"] - b5 for k in ("S0_exact", "S2_lagID", "S2m24", "S2m52")}
            snap[br].append(dict(budget=Bg, recall=rec, base=base, epochs_exact_gt_heur=diff_ep[br][Bg], n_epochs=n_ep,
                                 d_free_heur=rec["S0_heur"] - base, d_free_exact=rec["S0_exact"] - base,
                                 d_S2_lagID=rec["S2_lagID"] - base, d_S2m24=rec["S2m24"] - base, d_S2m52=rec["S2m52"] - base,
                                 per_seq_vs_IND05=seqd))
        print("snapshot", br, "done", flush=True)
    hist = history_audit(data)
    print("history done", flush=True)
    json.dump(dict(note="exploratory correction of Gate 0A (EXPL only); not confirmatory", snapshot=snap,
                   history={f"{k[0]}|{k[1]}": v for k, v in hist.items()}), open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
