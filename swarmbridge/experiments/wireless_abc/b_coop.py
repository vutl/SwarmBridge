"""B: do CURRENT measurements of other UAVs add value beyond own pilots and a map learned offline? (CPU)

Physics: at 28 GHz the complex channel decorrelates within ~1 cm, so nothing complex-valued is transferable between
positions; what transfers is the angular structure (which directions carry power), i.e. a local covariance. Every
estimator below is LMMSE with a zero-mean prior covariance C chosen by the method, then beam choice by analytic expected
gain (and outage) on the 16-beam DFT codebook. Measurement as in A1 (downlink beam training, Np pilots, Ep = 1).
Estimators (per UAV u, time t):
  global  C = train covariance (no location)
  map     C from TRAIN channels (true h, offline survey) within R m of the UAV's position (>= MIN_NB neighbours,
          else global). This is the channel-knowledge-map baseline
  coop    C = a * C_map + (1 - a) * C_coop, C_coop from the OTHER UAVs' own LMMSE estimates (from their own pilots,
          shared as decoded measurements) at positions within R m, with age <= MAX_AGE s and t' <= t - 1 step (causal),
          weighted by exp(-age / TAU_AGE); a chosen on dev from {0, .25, .5, .75}; falls back to map when no message
  shuffle same as coop but messages taken from another episode of the same kind (same cost; tests that the gain
          is from the right information)
  oracle  C_coop built from the other UAVs' TRUE channels (diagnostic only)
Covariances are trace-normalised to N before mixing (only the direction structure matters for zero-mean LMMSE; the
scale enters through the global median-norm normalisation, as in A1).
Message cost per shared measurement: Np complex values at 2 x 8 bit + position 3 x 16 bit + time 16 bit.
Evaluated every STRIDE-th RT step (0.2 s). Split by episode 60/20/20 (same as A1); results by episode kind (convoy = correlated paths, indep = control) and by
whether the map has neighbours. Output OUT/b_results.json. Usage: python b_coop.py TRACE_DIR OUT [--seeds 0 1 2]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ncx2

import linear as Lc
from a1_bench import batch_lmmse, N, GAMMA, FLOOR_DB

R, MIN_NB, MAX_AGE, TAU_AGE, DT, STRIDE = 5.0, 5, 10.0, 5.0, 0.1, 2
ALPHAS = [0.0, 0.25, 0.5, 0.75]
SNRS_DB = [-5, 5, 15]
NPS = [4, 8]
PROTOS = ["dft", "random"]


def load(tdir):
    eps = []
    for f in sorted(Path(tdir).glob("ep*.npz")):
        d = np.load(f)
        eps.append(dict(ep=int(f.stem[2:]), kind=str(d["kind"]), pos=d["pos"].astype(np.float64),
                        h=d["a"].sum(2).astype(np.complex128)))           # h: (U, T, N)
    return eps


def tnorm(C):
    return C * (N / max(np.real(np.trace(C)), 1e-12))


def cov_of(X, w=None):
    if len(X) == 0:
        return None
    w = np.ones(len(X)) if w is None else w
    return tnorm((X.T * w) @ X.conj() / w.sum() + 1e-3 * np.eye(N))


def decide(M, P, snr):
    F = Lc.dft_codebook(N)
    v = np.real(np.einsum("bi,ij,jb->b", F.conj().T, P, F))
    mb = np.abs(M @ F.conj()) ** 2
    eg = mb + v[None]
    po = ncx2.cdf(2 * (GAMMA / snr) / v[None], df=2, nc=2 * mb / v[None])
    return np.argmax(eg, 1), np.argmin(po - 1e-9 * eg, 1)


def outcome(h, b, snr):
    F = Lc.dft_codebook(N)
    g = np.abs(h @ F.conj()) ** 2
    gb = g[np.arange(len(h)), b]
    return Lc.rate(gb, snr), (snr * gb < GAMMA), Lc.rate(g.max(1), snr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("out"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eps = load(args.trace)
    n_ep = len(eps)
    a_, b_ = int(0.6 * n_ep), int(0.8 * n_ep)
    tr = [e for e in eps if e["ep"] < a_]
    allp = np.concatenate([np.sum(np.abs(e["h"]) ** 2, -1).ravel() for e in tr])
    med = np.median(allp[allp > 0])
    scale = np.sqrt(med / N)
    floor = med * 10 ** (-FLOOR_DB / 10)
    for e in eps:
        e["h"] = e["h"] / scale
        e["ok"] = np.sum(np.abs(e["h"]) ** 2, -1) * scale ** 2 > floor
    # map: train channels with their positions
    Ptr = np.concatenate([e["pos"][e["ok"]] for e in tr])
    Htr = np.concatenate([e["h"][e["ok"]] for e in tr])
    tree = cKDTree(Ptr)
    C_glob = tnorm(np.cov(Htr.T))
    info = dict(n_ep=n_ep, train_eps=a_, map_points=len(Ptr), R=R, MIN_NB=MIN_NB, MAX_AGE=MAX_AGE, TAU_AGE=TAU_AGE)
    print(info, flush=True)

    def map_cov(p):
        nb = tree.query_ball_point(p, R)
        if len(nb) < MIN_NB:
            return C_glob, False
        return cov_of(Htr[nb]), True

    t0 = time.time()
    for e in eps[a_:]:                                    # map prior per position: computed once (does not depend on noise)
        U, T, _ = e["h"].shape
        e["cm"] = np.empty((U, T, N, N), complex)
        e["has_map"] = np.zeros((U, T), bool)
        for u in range(U):
            for t in range(T):
                e["cm"][u, t], e["has_map"][u, t] = map_cov(e["pos"][u, t])
    print(f"map priors {time.time() - t0:.0f}s", flush=True)
    rows = []
    for seed in args.seeds:
        for proto in PROTOS:
            for n_p in NPS:
                rng = np.random.default_rng([seed, n_p, PROTOS.index(proto), 99])
                W = Lc.pilot_beams(N, n_p, proto, rng)
                A = W.conj().T
                for sdb in SNRS_DB:
                    snr, nv = 10 ** (sdb / 10), 10 ** (-sdb / 10)
                    t0 = time.time()
                    acc = {}
                    # pass 1: every UAV's own pilots and own LMMSE estimate with the MAP prior (what it would share)
                    for e in eps[a_:]:
                        g = np.random.default_rng([seed, e["ep"], sdb + 100, n_p, PROTOS.index(proto)])
                        U, T, _ = e["h"].shape
                        e["y"] = e["h"] @ A.T + np.sqrt(nv) * Lc.crandn(g, U, T, n_p)
                        e["est"] = np.empty((U, T, N), complex)
                        for u in range(U):
                            for t in range(T):
                                e["est"][u, t] = batch_lmmse(e["y"][u, t][None], A, nv, np.zeros(N), e["cm"][u, t])[0][0]
                    same_kind = {k: [x for x in eps[a_:] if x["kind"] == k] for k in ("convoy", "indep")}
                    for e in eps[a_:]:
                        split_ = "dev" if e["ep"] < b_ else "test"
                        U, T, _ = e["h"].shape
                        pool = same_kind[e["kind"]]
                        other = pool[(pool.index(e) + 1) % len(pool)]          # shuffled source episode
                        for u in range(U):
                            for t in range(0, T, STRIDE):
                                if not e["ok"][u, t]:
                                    continue
                                p = e["pos"][u, t]
                                srcs = {"coop": (e, "est"), "oracle": (e, "h"), "shuffle": (other, "est")}
                                C_co, nmsg = {}, 0
                                for name, (src, field) in srcs.items():
                                    X, w = [], []
                                    for v in range(U):
                                        if v == u and name != "shuffle":
                                            continue
                                        t_lo = max(0, t - int(MAX_AGE / DT))
                                        P = src["pos"][v, t_lo:t]                  # strictly before t (causal)
                                        if len(P) == 0:
                                            continue
                                        d = np.linalg.norm(P - p, axis=1)
                                        sel = np.where(d <= R)[0]
                                        if len(sel):
                                            age = (t - (t_lo + sel)) * DT
                                            X.append(src[field][v, t_lo + sel])
                                            w.append(np.exp(-age / TAU_AGE))
                                    if X:
                                        C_co[name] = cov_of(np.concatenate(X), np.concatenate(w))
                                        if name == "coop":
                                            nmsg = len(np.concatenate(w))
                                h = e["h"][u, t][None]
                                y = e["y"][u, t][None]
                                Cm = e["cm"][u, t]
                                cands = {"global": C_glob, "map": Cm}
                                for name in ("coop", "shuffle", "oracle"):
                                    for a in ALPHAS:
                                        cands[f"{name}_a{a}"] = (tnorm(a * Cm + (1 - a) * C_co[name]) if name in C_co
                                                                 else Cm)
                                for mname, Cp in cands.items():
                                    M, P_ = batch_lmmse(y, A, nv, np.zeros(N), Cp)
                                    be, bo = decide(M, P_, snr)
                                    for rule, b in (("egain", be), ("outage", bo)):
                                        r, o, rb = outcome(h, b, snr)
                                        key = (split_, e["kind"], bool(e["has_map"][u, t]), "coop" in C_co, mname, rule)
                                        s = acc.setdefault(key, [0, 0.0, 0.0, 0.0, 0])
                                        s[0] += 1; s[1] += r[0]; s[2] += o[0]; s[3] += rb[0] - r[0]; s[4] += nmsg
                    for (split_, kind, hasmap, hascoop, mname, rule), s in acc.items():
                        rows.append(dict(seed=seed, proto=proto, n_p=n_p, snr_db=sdb, split=split_, kind=kind,
                                         has_map=hasmap, has_coop=hascoop, method=mname, rule=rule, n=s[0],
                                         rate_sum=s[1], outage_sum=s[2], regret_sum=s[3], msgs_sum=s[4]))
                    print(f"seed {seed} {proto} Np={n_p} snr={sdb:+d} {time.time() - t0:.0f}s", flush=True)
                    (out / "b_results.json").write_text(json.dumps(dict(info=info, rows=rows)))
    (out / "b_results.json").write_text(json.dumps(dict(info=info, rows=rows,
                                                         msg_bits_per_measurement={str(n): 16 * n + 48 + 16 for n in NPS})))


if __name__ == "__main__":
    main()
