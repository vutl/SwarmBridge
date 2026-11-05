"""B v2 (audit fixes 05/11): cooperation with a real message codec, a local-history baseline, link loss and period.

Changes vs b_coop.py (kept for reproducibility of the first round):
  * Real codec. A message is what is counted: the sender's Np pilot measurements, quantised to 8 bit per real
    component with a float16 per-message scale, + position (3 x 16 bit) + time (16 bit) + noise-level index (8 bit):
    BITS = 16 Np + 16 + 48 + 16 + 8. The receiver decodes y_hat, knows the shared pilot codebook W and the shared
    offline map, and reconstructs the sender's LMMSE estimate with the map prior at the SENDER's position and the noise
    variance inflated by the quantisation noise (2 * Delta^2 / 12 per complex sample). Round-trip is tested.
  * Local-history baseline 'own': map prior mixed with the UAV's OWN past estimates (same radius / age / weighting as
    the cooperative messages), so cooperation must beat memory, not only the static map.
  * 'coop' = map + own history + decoded messages of the other UAVs; 'shuffle' = map + own history + messages from
    another episode of the same kind (same count, same bits); 'oracle' = map + own history + other UAVs' TRUE channels.
  * Link: a message is broadcast every PERIOD RT steps (0.1 s each), lost i.i.d. with probability LOSS (same loss
    for every receiver), usable from the next RT step on (causal), within R m and MAX_AGE s.
  * Map coverage: the map is built from a fraction MAP_FRAC of the train episodes (0 = no map, global prior only).
Mixing weight a in C = a C_map + (1 - a) C_hist is chosen on DEV per method / config from {0, .25, .5, .75}; rule
'egain' (analytic expected gain). Np = 4. Output OUT/b2_results.json. Usage: python b_coop2.py TRACE_DIR OUT
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import linear as Lc
from a1_bench import batch_lmmse, N, GAMMA, FLOOR_DB
from b_coop import load, tnorm, cov_of, decide, outcome, R, MIN_NB, MAX_AGE, TAU_AGE, DT

NP, STRIDE = 4, 2
ALPHAS = [0.0, 0.25, 0.5, 0.75]
F = Lc.dft_codebook(N)


def msg_bits(n_p):
    return 16 * n_p + 16 + 48 + 16 + 8


def quantise(y):
    """8-bit uniform quantiser per real component, per-message float16 scale. Returns (y_hat, delta)."""
    s = np.float16(max(np.abs(y.real).max(), np.abs(y.imag).max(), 1e-12)).astype(np.float64)
    delta = 2 * s / 255
    q = lambda x: np.clip(np.round((x + s) / delta), 0, 255) * delta - s
    return q(y.real) + 1j * q(y.imag), delta


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
    for e in eps:
        e["ok"] = np.sum(np.abs(e["h"]) ** 2, -1) > med * 10 ** (-FLOOR_DB / 10)
        e["h"] = e["h"] / scale
    ev = eps[a_:]
    # main setting + ablations (proto random, SNR 5 dB)
    settings = [dict(name="main", map_frac=1.0, loss=0.0, period=1, protos=["dft", "random"], snrs=[-5, 5, 15])]
    for nm, kw in (("map25", dict(map_frac=0.25)), ("nomap", dict(map_frac=0.0)), ("loss30", dict(loss=0.3)),
                   ("period5", dict(period=5))):
        st = dict(name=nm, map_frac=1.0, loss=0.0, period=1, protos=["random"], snrs=[5])
        st.update(kw)
        settings.append(st)
    rows = []
    info = dict(R=R, MAX_AGE=MAX_AGE, TAU_AGE=TAU_AGE, NP=NP, bits_per_msg=msg_bits(NP), stride=STRIDE,
                settings=settings)
    for st in settings:
        # ---- map from a fraction of the train episodes
        rngm = np.random.default_rng(7)
        tr_ep = sorted(rngm.permutation([e["ep"] for e in tr])[:int(round(st["map_frac"] * len(tr)))])
        trm = [e for e in tr if e["ep"] in set(tr_ep)]
        Htr_all = np.concatenate([e["h"][e["ok"]] for e in tr])
        C_glob = tnorm(np.cov(Htr_all.T))
        if trm:
            Ptr = np.concatenate([e["pos"][e["ok"]] for e in trm])
            Htr = np.concatenate([e["h"][e["ok"]] for e in trm])
            tree = cKDTree(Ptr)
        t0 = time.time()
        for e in ev:
            U, T, _ = e["h"].shape
            e["cm"] = np.empty((U, T, N, N), complex)
            e["has_map"] = np.zeros((U, T), bool)
            for u in range(U):
                for t in range(T):
                    nb = tree.query_ball_point(e["pos"][u, t], R) if trm else []
                    if len(nb) >= MIN_NB:
                        e["cm"][u, t], e["has_map"][u, t] = cov_of(Htr[nb]), True
                    else:
                        e["cm"][u, t] = C_glob
        print(f"[{st['name']}] map from {len(trm)} train episodes, priors {time.time() - t0:.0f}s", flush=True)
        for seed in args.seeds:
            for proto in st["protos"]:
                rng = np.random.default_rng([seed, NP, ["dft", "random"].index(proto), 99])
                W = Lc.pilot_beams(N, NP, proto, rng)
                A = W.conj().T
                for sdb in st["snrs"]:
                    snr, nv = 10 ** (sdb / 10), 10 ** (-sdb / 10)
                    t0 = time.time()
                    for e in ev:
                        g = np.random.default_rng([seed, e["ep"], sdb + 100, NP, ["dft", "random"].index(proto)])
                        U, T, _ = e["h"].shape
                        e["y"] = e["h"] @ A.T + np.sqrt(nv) * Lc.crandn(g, U, T, NP)
                        CAh = e["cm"] @ A.conj().T
                        e["est"] = np.einsum("utnk,utk->utn", CAh,
                                             np.linalg.solve(A @ CAh + nv * np.eye(NP), e["y"][..., None])[..., 0])
                        # decoded message -> receiver-side reconstruction of the sender's estimate
                        e["rx"] = np.empty_like(e["est"])
                        for u in range(U):
                            for t in range(T):
                                yq, dlt = quantise(e["y"][u, t])
                                nq = nv + 2 * dlt ** 2 / 12
                                e["rx"][u, t] = batch_lmmse(yq[None], A, nq, np.zeros(N), e["cm"][u, t])[0][0]
                        gl = np.random.default_rng([seed, e["ep"], 555, sdb + 100])
                        e["sent"] = (np.arange(T)[None] % st["period"] == 0) & (gl.random((U, T)) >= st["loss"])
                    same_kind = {k: [x for x in ev if x["kind"] == k] for k in ("convoy", "indep")}
                    acc = {}
                    for e in ev:
                        split_ = "dev" if e["ep"] < b_ else "test"
                        U, T, _ = e["h"].shape
                        pool = same_kind[e["kind"]]
                        other = pool[(pool.index(e) + 1) % len(pool)]
                        for u in range(U):
                            for t in range(0, T, STRIDE):
                                if not e["ok"][u, t]:
                                    continue
                                p = e["pos"][u, t]
                                t_lo = max(0, t - int(MAX_AGE / DT))

                                def gather(src, field, vs, need_sent):
                                    X, w = [], []
                                    for v in vs:
                                        P_ = src["pos"][v, t_lo:t]
                                        if len(P_) == 0:
                                            continue
                                        ok = np.linalg.norm(P_ - p, axis=1) <= R
                                        if need_sent:
                                            ok &= src["sent"][v, t_lo:t]
                                        sel = np.where(ok)[0]
                                        if len(sel):
                                            X.append(src[field][v, t_lo + sel])
                                            w.append(np.exp(-(t - (t_lo + sel)) * DT / TAU_AGE))
                                    return X, w
                                Xo, wo = gather(e, "est", [u], False)                       # own history (local)
                                others = [v for v in range(U) if v != u]
                                Xc, wc = gather(e, "rx", others, True)                      # decoded messages
                                Xs, ws = gather(other, "rx", others, True)                  # shuffled source
                                Xr, wr = gather(e, "h", others, True)                       # oracle content
                                n_msgs = int(sum(len(x) for x in Xc))
                                hists = {"own": (Xo, wo), "coop": (Xo + Xc, wo + wc), "shuffle": (Xo + Xs, wo + ws),
                                         "oracle": (Xo + Xr, wo + wr)}
                                Cm = e["cm"][u, t]
                                cands = {"global": C_glob, "map": Cm}
                                for nm_, (X, w) in hists.items():
                                    Ch = cov_of(np.concatenate(X), np.concatenate(w)) if X else None
                                    for a in ALPHAS:
                                        cands[f"{nm_}_a{a}"] = Cm if Ch is None else tnorm(a * Cm + (1 - a) * Ch)
                                h = e["h"][u, t][None]
                                y = e["y"][u, t][None]
                                for mname, Cp in cands.items():
                                    M, P_ = batch_lmmse(y, A, nv, np.zeros(N), Cp)
                                    be, _ = decide(M, P_, snr)
                                    r, o, _ = outcome(h, be, snr)
                                    key = (split_, e["kind"], bool(e["has_map"][u, t]), n_msgs > 0, mname)
                                    s = acc.setdefault(key, [0, 0.0, 0.0, 0])
                                    s[0] += 1; s[1] += r[0]; s[2] += o[0]; s[3] += n_msgs
                    for (split_, kind, hm, hc, mname), s in acc.items():
                        rows.append(dict(setting=st["name"], seed=seed, proto=proto, n_p=NP, snr_db=sdb, split=split_,
                                         kind=kind, has_map=hm, has_msgs=hc, method=mname, n=s[0], rate_sum=s[1],
                                         outage_sum=s[2], msgs_used_sum=s[3]))
                    print(f"[{st['name']}] seed {seed} {proto} snr {sdb:+d} {time.time() - t0:.0f}s", flush=True)
                    (out / "b2_results.json").write_text(json.dumps(dict(info=info, rows=rows)))
    (out / "b2_results.json").write_text(json.dumps(dict(info=info, rows=rows)))


if __name__ == "__main__":
    main()
