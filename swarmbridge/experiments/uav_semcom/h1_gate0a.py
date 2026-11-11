"""H1 Gate 0A (exploratory): cross-UAV coordination opportunity on MDMT public test labels.

Spec: reports/swarmbridge-h1-gate0-protocol-v2.md sec 10 (v2.1, locked before running).
Perfect reports (GT boxes); two separate hypotheses: TOK (52-bit tokens) and CROP (bit model).
Budget B = total of both UAVs per epoch incl. headers + rho * downlink coordination bits.
BS removes duplicates with ideal identity in every policy; coordination only saves uplink budget.

Usage: python h1_gate0a.py LABEL_DIR OUT_JSON
"""
import argparse
import glob
import hashlib
import json
import math
import os
from collections import defaultdict

W_IMG, H_IMG = 1920, 1080
TOK_BITS = 52            # r1b_sim tok_box
MSG_HDR = 48             # r1b_sim tok_hdr
APP_HDR = 96             # per 1200-byte packet
MTU = 1200 * 8
DL_HDR = 48
QUOTA_BITS = 22
TOK_BUDGETS = [250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000]
CROP_BUDGETS = [10e3, 20e3, 50e3, 100e3, 200e3, 500e3, 1e6, 2e6]
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7]
RATES = [1, 2, 5, 10]


# ---------------------------------------------------------------- data
def load(label_dir, drop_occluded=False):
    """-> {seq: {view: {frame: {id: (x, y, w, h)}}}} from MOT-style gt (frame,id,x,y,w,h,_,cls,vis)."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for f in sorted(glob.glob(os.path.join(label_dir, "*-*.txt"))):
        seq, view = os.path.basename(f)[:-4].split("-")
        for line in open(f):
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            if drop_occluded and len(p) > 8 and float(p[8]) < 1:
                continue
            out[seq][int(view)][int(p[0])][int(p[1])] = tuple(float(v) for v in p[2:6])
    return out


def crop_bits(box, bpp, c0):
    x, y, w, h = box
    cw, ch = min(max(1.5 * w, 64), W_IMG), min(max(1.5 * h, 64), H_IMG)
    return c0 + bpp * cw * ch


# ---------------------------------------------------------------- header / capacity helpers
def msg_overhead(payload):
    return 0 if payload <= 0 else MSG_HDR + APP_HDR * math.ceil((MSG_HDR + payload) / MTU)


def tok_capacity(budget):
    """max tokens one UAV message can carry within budget (incl. headers)."""
    if budget < TOK_BITS + msg_overhead(TOK_BITS):
        return 0
    k = int((budget - MSG_HDR - APP_HDR) // TOK_BITS)
    while k > 0 and k * TOK_BITS + msg_overhead(k * TOK_BITS) > budget:
        k -= 1
    return k


def p_hit_random(k, n):
    return 0.0 if n == 0 else min(1.0, k / n)


# ---------------------------------------------------------------- TOK policies (exact expectations)
def tok_independent(A, B, bA, bB):
    """each UAV sends a uniformly random subset of its objects within its own budget."""
    a_only, b_only, both = len(A - B), len(B - A), len(A & B)
    kA, kB = tok_capacity(bA), tok_capacity(bB)
    pA, pB = p_hit_random(kA, len(A)), p_hit_random(kB, len(B))
    cov = a_only * pA + b_only * pB + both * (1 - (1 - pA) * (1 - pB))
    sent = min(kA, len(A)) + min(kB, len(B))
    dup = both * pA * pB
    bits = sent * TOK_BITS + msg_overhead(min(kA, len(A)) * TOK_BITS) + msg_overhead(min(kB, len(B)) * TOK_BITS)
    return dict(cov=cov, dup_tok=dup, ul_bits=bits, dl_bits=0.0, lost=0.0)


def tok_coordinated(A, B, sup, budget, dl_bits, rho):
    """UAV1 sends A; UAV2 sends B minus suppressed set. Joint budget; if short, a uniformly random
    subset of the candidate copies is sent (all tokens cost the same). Expected distinct coverage."""
    candA, candB = set(A), set(B) - sup
    lost = len(sup - A)                                     # suppressed by UAV2 but UAV1 does not have it
    dups = len(candA & candB)
    n = len(candA) + len(candB)
    distinct = len(candA | candB)
    avail = budget - rho * dl_bits
    # one message per UAV with a non-empty list; header cost counted with full-list packetisation
    # (upper bound on overhead; <= 183 tokens fit one 1200-byte packet, so one packet per UAV in practice)
    nh = (1 if candA else 0) + (1 if candB else 0)
    hdr = msg_overhead(len(candA) * TOK_BITS) + msg_overhead(len(candB) * TOK_BITS)
    K = max(0, min(n, int((avail - hdr) // TOK_BITS))) if avail > hdr else 0
    if K >= n:
        cov, dup_sent = distinct, dups
    elif n == 0:
        cov, dup_sent = 0.0, 0.0
    else:
        singles, doubles = distinct - dups, dups
        p1 = K / n
        p2 = 1 - ((n - K) * (n - K - 1)) / (n * (n - 1)) if n > 1 else p1
        both_sel = (K * (K - 1)) / (n * (n - 1)) if n > 1 else 0.0
        cov, dup_sent = singles * p1 + doubles * p2, doubles * both_sel
    sent = min(K, n)
    return dict(cov=cov, dup_tok=dup_sent, ul_bits=sent * TOK_BITS + (hdr if sent else 0),
                dl_bits=dl_bits, lost=lost)


# ---------------------------------------------------------------- CROP policies (cheapest first)
def greedy_items(items, budget, per_uav=True):
    """items: list of (cost, view, id). Cheapest first; per-UAV message overhead with packetisation.
    Returns selected list."""
    pay = {1: 0.0, 2: 0.0}
    sel = []
    for c, v, i in sorted(items):
        trial = dict(pay); trial[v] += c
        tot = sum(trial.values()) + sum(msg_overhead(p) for p in trial.values())
        if tot <= budget:
            pay = trial; sel.append((c, v, i))
    return sel


def crop_independent(A, B, cA, cB, bA, bB):
    sA = greedy_items([(cA[i], 1, i) for i in A], bA)
    sB = greedy_items([(cB[i], 2, i) for i in B], bB)
    idsA, idsB = {i for _, _, i in sA}, {i for _, _, i in sB}
    bits = sum(c for c, _, _ in sA + sB) + msg_overhead(sum(c for c, _, _ in sA)) + msg_overhead(sum(c for c, _, _ in sB))
    return dict(cov=len(idsA | idsB), dup_bits=sum(cB[i] for i in idsA & idsB), ul_bits=bits, dl_bits=0.0, lost=0)


def crop_coordinated(A, B, cA, cB, sup, budget, dl_bits, rho, oracle=False):
    if oracle:      # S0: each shared object sent once, from the cheaper view
        C = A & B
        items = [(cA[i], 1, i) for i in A - C] + [(cB[i], 2, i) for i in B - C] + \
                [(min(cA[i], cB[i]), 1 if cA[i] <= cB[i] else 2, i) for i in C]
        lost = 0
    else:
        items = [(cA[i], 1, i) for i in A] + [(cB[i], 2, i) for i in set(B) - sup]
        lost = len(sup - A)
    sel = greedy_items(items, budget - rho * dl_bits)
    seen, dup_bits = set(), 0.0
    for c, v, i in sorted(sel):
        if i in seen:
            dup_bits += c
        seen.add(i)
    pays = {1: sum(c for c, v, _ in sel if v == 1), 2: sum(c for c, v, _ in sel if v == 2)}
    return dict(cov=len(seen), dup_bits=dup_bits, ul_bits=sum(pays.values()) + sum(msg_overhead(p) for p in pays.values()),
                dl_bits=dl_bits, lost=lost)


# ---------------------------------------------------------------- coordination schemes (suppression sets)
def bloom_fp(seq, f, oid, rate):
    h = int(hashlib.blake2b(f"{seq}|{f}|{oid}".encode(), digest_size=8).hexdigest(), 16) / 2 ** 64
    return h < rate


def grid_mask(Bboxes, shared, g):
    gx, gy = g
    tot, sh = defaultdict(int), defaultdict(int)
    for i, (x, y, w, h) in Bboxes.items():
        c = (min(int((x + w / 2) / W_IMG * gx), gx - 1), min(int((y + h / 2) / H_IMG * gy), gy - 1))
        tot[c] += 1; sh[c] += i in shared
    return {c for c in tot if sh[c] / tot[c] >= 0.5}


def in_mask(box, mask, g):
    gx, gy = g
    x, y, w, h = box
    return (min(int((x + w / 2) / W_IMG * gx), gx - 1), min(int((y + h / 2) / H_IMG * gy), gy - 1)) in mask


def schemes(seq, f, A, B, Bbox, prev, static_mask):
    """-> {name: (suppressed set, dl_bits)} for one epoch. prev = (A_prev, B_prev, Bbox_prev) or None."""
    out = {"S0": (A & B, 0.0)}
    if prev is None:
        for k in ("S1", "S2", "S2b_1", "S2b_5", "S3_8x8", "S3_16x9", "S3s_8x8", "S3s_16x9"):
            out[k] = (set(), 0.0)
        return out
    Ap, Bp, Bboxp = prev
    Cp = Ap & Bp
    sup = set(B) & Cp
    out["S1"] = (sup, TOK_BITS * len(Cp) + DL_HDR)
    out["S2"] = (sup, len(Bp) + DL_HDR)
    for fr, nm in ((0.01, "S2b_1"), (0.05, "S2b_5")):
        fp = {i for i in set(B) - Cp if bloom_fp(seq, f, i, fr)}
        out[nm] = (sup | fp, math.ceil(1.44 * math.log2(1 / fr) * len(Cp)) + DL_HDR)
    for g, nm in (((8, 8), "8x8"), ((16, 9), "16x9")):
        m = grid_mask(Bboxp, Cp, g)
        out["S3_" + nm] = ({i for i in B if in_mask(Bbox[i], m, g)}, g[0] * g[1] + DL_HDR)
        sm = static_mask[nm]
        out["S3s_" + nm] = ({i for i in B if in_mask(Bbox[i], sm, g)}, 0.0)
    return out


# ---------------------------------------------------------------- main evaluation
def epochs(data, stride):
    for seq in sorted(data):
        v1, v2 = data[seq][1], data[seq][2]
        frames = sorted(set(v1) | set(v2))[::stride]
        prev = None
        first = None
        for f in frames:
            A, B = set(v1.get(f, {})), set(v2.get(f, {}))
            yield seq, f, A, B, v1.get(f, {}), v2.get(f, {}), prev, first
            Bbox = v2.get(f, {})
            prev = (A, B, Bbox)
            if first is None:
                first = (A, B, Bbox)


def evaluate(data, stride=10, rho=1.0, bpp=0.65, c0=2400.0, branches=("TOK", "CROP")):
    acc = {}                     # (branch, budget, policy) -> sums
    per_seq = defaultdict(dict)  # seq -> (branch, budget, policy) -> [cov, U]
    n_ep = 0; tot_U = 0; tot_both = 0; tot_ab = 0
    static_cache = {}

    def add(key, seq, r, U):
        s = acc.setdefault(key, defaultdict(float))
        for k, v in r.items():
            s[k] += v
        s["U"] += U
        ps = per_seq[seq].setdefault(key, [0.0, 0])
        ps[0] += r["cov"]; ps[1] += U

    for seq, f, A, B, Abox, Bbox, prev, first in epochs(data, stride):
        n_ep += 1
        U = len(A | B); tot_U += U; tot_both += len(A & B); tot_ab += len(A) + len(B)
        if seq not in static_cache and first is not None:
            fA, fB, fBbox = first
            static_cache[seq] = {nm: grid_mask(fBbox, fA & fB, g) for g, nm in (((8, 8), "8x8"), ((16, 9), "16x9"))}
        sm = static_cache.get(seq, {"8x8": set(), "16x9": set()})
        sch = schemes(seq, f, A, B, Bbox, prev, sm)
        cA = {i: crop_bits(b, bpp, c0) for i, b in Abox.items()}
        cB = {i: crop_bits(b, bpp, c0) for i, b in Bbox.items()}
        for br in branches:
            budgets = TOK_BUDGETS if br == "TOK" else CROP_BUDGETS
            for Bg in budgets + ["all"]:
                if Bg == "all":
                    Bg_eff = 1e12
                else:
                    Bg_eff = Bg
                # independent (equal split, best alpha decided later), quota
                for a in ALPHAS:
                    bA, bB = a * Bg_eff, (1 - a) * Bg_eff
                    r = tok_independent(A, B, bA, bB) if br == "TOK" else crop_independent(A, B, cA, cB, bA, bB)
                    add((br, Bg, f"IND_a{a}"), seq, r, U)
                q_b = Bg_eff - rho * QUOTA_BITS
                share = len(A) / max(len(A) + len(B), 1)
                r = tok_independent(A, B, share * q_b, (1 - share) * q_b) if br == "TOK" else \
                    crop_independent(A, B, cA, cB, share * q_b, (1 - share) * q_b)
                r["dl_bits"] = QUOTA_BITS
                add((br, Bg, "Q"), seq, r, U)
                for nm, (sup, dl) in sch.items():
                    if br == "TOK":
                        r = tok_coordinated(A, B, sup if nm != "S0" else (A & B), Bg_eff, dl, rho)
                    else:
                        r = crop_coordinated(A, B, cA, cB, sup, Bg_eff, dl, rho, oracle=(nm == "S0"))
                    add((br, Bg, nm), seq, r, U)
    return acc, per_seq, dict(epochs=n_ep, U=tot_U, both=tot_both, sum_ab=tot_ab)


def summarise(acc, per_seq, stats):
    res = {"stats": stats, "branches": {}}
    for br in ("TOK", "CROP"):
        budgets = (TOK_BUDGETS if br == "TOK" else CROP_BUDGETS) + ["all"]
        rows = []
        for Bg in budgets:
            pol = {k[2]: v for k, v in acc.items() if k[0] == br and k[1] == Bg}
            if not pol:
                continue
            rec = {p: 100 * v["cov"] / v["U"] for p, v in pol.items()}
            ind_best_a = max(ALPHAS, key=lambda a: rec[f"IND_a{a}"])
            base = max(rec["IND_a0.5"], rec[f"IND_a{ind_best_a}"], rec["Q"])
            real = {k: rec[k] for k in rec if k.startswith("S") and k != "S0"}
            best_real = max(real, key=real.get)
            d_free, d_real = rec["S0"] - base, real[best_real] - base
            ep = stats["epochs"]
            rows.append(dict(
                budget=Bg, recall=rec, base=base, ind_best_alpha=ind_best_a, d_free=d_free, best_real=best_real,
                d_real=d_real, headroom=bool(d_free >= 5 and d_real >= 0.5 * d_free),
                ul_bits_per_epoch={p: v["ul_bits"] / ep for p, v in pol.items()},
                dl_bits_per_epoch={p: v["dl_bits"] / ep for p, v in pol.items()},
                lost_per_epoch={p: v["lost"] / ep for p, v in pol.items()},
                dup_per_epoch={p: (v.get("dup_tok", 0) * TOK_BITS if br == "TOK" else v.get("dup_bits", 0)) / ep
                               for p, v in pol.items()},
                kbps={r: (Bg * r / 1e3 if Bg != "all" else None) for r in RATES},
                per_seq={s: {p: 100 * d[(br, Bg, p)][0] / max(d[(br, Bg, p)][1], 1) for p in ("S0", "IND_a0.5", "Q", best_real)}
                         for s, d in per_seq.items()}))
        res["branches"][br] = rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("out")
    a = ap.parse_args()
    data = load(a.labels)
    out = {"spec": "reports/swarmbridge-h1-gate0-protocol-v2.md sec 10 (v2.1)", "runs": {}}
    configs = [("main", dict(stride=10, rho=1.0, bpp=0.65, c0=2400.0), False),
               ("s1", dict(stride=1, rho=1.0, branches=("TOK",)), False),
               ("s30", dict(stride=30, rho=1.0), False),
               ("rho0", dict(stride=10, rho=0.0), False),
               ("rho0.5", dict(stride=10, rho=0.5), False),
               ("crop_bpp0.3", dict(stride=10, rho=1.0, bpp=0.3, branches=("CROP",)), False),
               ("crop_c0_0", dict(stride=10, rho=1.0, c0=0.0, branches=("CROP",)), False),
               ("no_occluded", dict(stride=10, rho=1.0), True)]
    for name, kw, drop in configs:
        d = load(a.labels, drop_occluded=drop) if drop else data
        acc, per_seq, stats = evaluate(d, **kw)
        out["runs"][name] = dict(config=kw, **summarise(acc, per_seq, stats))
        print("done", name, stats, flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
