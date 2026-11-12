"""Offline gate analysis on the pilot 0B EXPL records (review #12, step 3): can cheap ONBOARD signals predict, per
(epoch, view), whether sending crops (T1) beats sending tokens only (T0)? No new image is opened: the precomputed
records and the scorer GT are reused. Leave-one-SEQUENCE-out (11 folds; sequences are the independent units):
everything (tau_decl per mode, feature / direction / threshold, logistic model and its threshold) is chosen on the
10 training sequences, then applied to the held-out one.

Policies per view (UAV share = B / 2, alpha fixed at 0.5 to avoid another tuned knob):
  T0 always tokens | T1 always tokens + P1 crops (q) | GATE-F best single feature threshold | GATE-LR logistic
  regression on all features | RAND random gate with the held-out crop rate of GATE-LR (mean of 50 draws) |
  ORACLE per-(epoch, view) choice of the mode with the larger net TP gain (uses GT; reference only).
Features (causal, computed on the UAV from its own YOLO11n output; P1 bit total needs the encodes, which the
CPU model of T1 already pays): n_det, n_strong, n_weak, weak_frac, mean_strong_conf, mean_weak_conf, n_p1,
p1_kbit, med_strong_area (log), frac_small_strong.
Usage: python h1_pilot0b_gate.py PRECOMP_DIR OUT_JSON --zip MDMT_ZIP [--q 50]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import h1_pilot0b as P
import h1_pilot0b_eval as E

FEATS = ("n_det", "n_strong", "n_weak", "weak_frac", "mean_strong_conf", "mean_weak_conf", "n_p1", "p1_kbit",
         "log_med_strong_area", "frac_small_strong")
QUANT = np.linspace(0.0, 1.0, 21)


def features(rec, q):
    d = rec["det"]; s = d["s"]
    st, wk = s >= P.TAU_HI, (s >= P.TAU_LO) & (s < P.TAU_HI)
    area = (d["b"][:, 2] - d["b"][:, 0]) * (d["b"][:, 3] - d["b"][:, 1])
    sa = area[st]
    return dict(n_det=len(s), n_strong=int(st.sum()), n_weak=int(wk.sum()), weak_frac=wk.sum() / max(len(s), 1),
                mean_strong_conf=float(s[st].mean()) if st.any() else 0.0,
                mean_weak_conf=float(s[wk].mean()) if wk.any() else 0.0, n_p1=len(rec["cands"]["P1"]),
                p1_kbit=sum(c["bits"][q] for c in rec["cands"]["P1"]) / 1e3,
                log_med_strong_area=float(np.log1p(np.median(sa))) if len(sa) else 0.0,
                frac_small_strong=float((sa < 32 * 32).mean()) if len(sa) else 0.0)


def tabulate(pairs_by_seq, q, B):
    """per seq: counts[n_ev, mode(2), tau, (tp, fp, fn)], ul[n_ev, mode], X[n_ev, n_feat]."""
    out = {}
    for s, pairs in pairs_by_seq.items():
        C, U, X = [], [], []
        for r1, r2, g1, g2 in pairs:
            for rec, g in ((r1, g1), (r2, g2)):
                cm, um = [], []
                for mode in ("T0", "T1"):
                    snd = P.send(rec, mode, q, B / 2)
                    dec = P.bs_decode(rec, snd, q)
                    cm.append([[(c["tp"], c["fp"], c["fn"]) for c in [P.view_counts(dec, g, t)]][0] for t in E.TAUS])
                    um.append(snd["ul"])
                C.append(cm); U.append(um)
                f = features(rec, q); X.append([f[k] for k in FEATS])
        out[s] = dict(C=np.array(C, float), U=np.array(U, float), X=np.array(X, float))
    return out


def f1_of(c):
    tp, fp, fn = c
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 1.0


def seq_f1(T, dec, t0, t1):
    """dec: bool per (e, v) -> F1 of the sequence with modes chosen by dec."""
    c = np.where(dec[:, None], T["C"][:, 1, t1, :], T["C"][:, 0, t0, :]).sum(0)
    return f1_of(c), float(np.where(dec, T["U"][:, 1], T["U"][:, 0]).mean() * 2)


def macro(Ts, decs, t0, t1):
    return float(np.mean([seq_f1(T, d, t0, t1)[0] for T, d in zip(Ts, decs)]))


def net_gain(T, t0, t1):
    c1, c0 = T["C"][:, 1, t1, :], T["C"][:, 0, t0, :]
    return (c1[:, 0] - c1[:, 1]) - (c0[:, 0] - c0[:, 1])


def fold(tab, held):
    from sklearn.linear_model import LogisticRegression
    tr = [s for s in tab if s != held]
    Ts = [tab[s] for s in tr]
    nt = len(E.TAUS)
    t0 = max(range(nt), key=lambda t: macro(Ts, [np.zeros(len(T["U"]), bool) for T in Ts], t, 0))
    t1 = max(range(nt), key=lambda t: macro(Ts, [np.ones(len(T["U"]), bool) for T in Ts], 0, t))
    Xtr = np.concatenate([T["X"] for T in Ts])
    best = None
    for j, name in enumerate(FEATS):
        for sign in (1, -1):
            for th in np.unique(np.quantile(Xtr[:, j], QUANT)):
                f = macro(Ts, [sign * T["X"][:, j] >= sign * th for T in Ts], t0, t1)
                if best is None or f > best[0] + 1e-12:
                    best = (f, name, sign, float(th))
    _, fname, fsign, fth = best
    j = FEATS.index(fname)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    ytr = np.concatenate([net_gain(T, t0, t1) > 0 for T in Ts])
    if len(set(ytr.tolist())) < 2:                               # one class only: constant gate
        const = bool(ytr[0])
        prob = lambda X: np.full(len(X), float(const))           # noqa: E731
        coef = {}
    else:
        lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit((Xtr - mu) / sd, ytr)
        prob = lambda X: lr.predict_proba((X - mu) / sd)[:, 1]   # noqa: E731
        coef = dict(zip(FEATS, lr.coef_[0].round(3).tolist()))
    pth = max(np.linspace(0.05, 0.95, 19), key=lambda p: macro(Ts, [prob(T["X"]) >= p for T in Ts], t0, t1))
    H = tab[held]; n = len(H["U"])
    dec = dict(T0=np.zeros(n, bool), T1=np.ones(n, bool), GATE_F=fsign * H["X"][:, j] >= fsign * fth,
               GATE_LR=prob(H["X"]) >= pth, ORACLE=net_gain(H, t0, t1) > 0)
    res = {k: dict(zip(("f1", "ul"), seq_f1(H, d, t0, t1)), crop_rate=float(d.mean())) for k, d in dec.items()}
    rng = np.random.default_rng(int(held))
    rate = dec["GATE_LR"].mean()
    rnd = [seq_f1(H, rng.random(n) < rate, t0, t1) for _ in range(50)]
    res["RAND"] = dict(f1=float(np.mean([r[0] for r in rnd])), ul=float(np.mean([r[1] for r in rnd])), crop_rate=rate)
    return dict(tau0=E.TAUS[t0], tau1=E.TAUS[t1], gate_feature=fname, gate_sign=fsign, gate_th=fth,
                lr_threshold=float(pth), lr_coef=coef, res=res)


def summarise(folds):
    seqs = sorted(folds, key=int)
    pols = ("T0", "T1", "GATE_F", "GATE_LR", "RAND", "ORACLE")
    f = {p: np.array([folds[s]["res"][p]["f1"] for s in seqs]) for p in pols}
    ul = {p: float(np.mean([folds[s]["res"][p]["ul"] for s in seqs])) for p in pols}
    keep = np.array([s != "59" and s != 59 for s in seqs])
    out = dict(seqs=seqs, macro_f1={p: float(f[p].mean()) for p in pols},
               macro_f1_without_59={p: float(f[p][keep].mean()) for p in pols}, mean_ul_bits=ul,
               crop_rate={p: float(np.mean([folds[s]["res"][p]["crop_rate"] for s in seqs])) for p in pols},
               per_seq_f1={p: f[p].round(4).tolist() for p in pols})
    for g in ("GATE_F", "GATE_LR"):
        out[f"{g}_vs_T0_seqs_positive"] = int((f[g] > f["T0"] + 1e-12).sum())
        out[f"{g}_vs_T1_seqs_positive"] = int((f[g] > f["T1"] + 1e-12).sum())
        out[f"{g}_vs_RAND_seqs_positive"] = int((f[g] > f["RAND"] + 1e-12).sum())
        hd = f["ORACLE"] - f["T0"]
        out[f"{g}_captured_oracle_headroom_vs_T0"] = float((f[g] - f["T0"]).sum() / max(hd.sum(), 1e-12))
    out["gate_features_chosen"] = [folds[s]["gate_feature"] for s in seqs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("precomp"); ap.add_argument("out"); ap.add_argument("--zip", required=True)
    ap.add_argument("--q", type=int, default=50)
    a = ap.parse_args()
    pairs, prov = E.load_pairs(a.precomp, a.zip, ("test",))
    lv = P.budget_levels([p for v in pairs.values() for p in v])
    res = dict(status="EXPLORATORY offline analysis on existing EXPL records; leave-one-sequence-out",
               q=a.q, alpha=0.5, budget=lv, feats=FEATS, by_B={})
    for B in lv["levels"]:
        tab = tabulate(pairs, a.q, B)
        folds = {str(s): fold(tab, s) for s in sorted(tab)}
        res["by_B"][str(B)] = dict(summary=summarise(folds), folds=folds)
        print(B, json.dumps(res["by_B"][str(B)]["summary"]["macro_f1"]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
