"""G-L2 steps 3-4 (reports/uav-semcom-g-l2.md 0.5 / 0.6): causal on-board predictor of per-image task value per
mode, matched-bitrate evaluation against the strongest content-agnostic baseline, PASS / INCONCLUSIVE / FAIL.

Candidates: feature sets A / A+B / A+C / A+B+C x targets 'abs' (TP_i(m)) / 'pair' (TP_i(m) - TP_i(mbt q3)) x models
ridge (alpha by grouped CV on train) / gbm (depth 3, 200 rounds, lr 0.05). Trained on train; per budget the lambda
mixture hitting B and ONE candidate (largest val Delta) are chosen on val; test and test-dev evaluated once.
Families are stored as member x image matrices, so video bootstrap / leave-one-video-out only re-index images.
Usage: python g2_predict.py [--boot 2000]
"""
import argparse
import json
import time

import numpy as np

import a5_common as C
import g2_common as G

PT = 100.0
FSETS = {"A": "A", "A+B": "AB", "A+C": "AC", "A+B+C": "ABC"}


class Fam:
    """Family of per-image choices: ch [K, n] mode indices for the images I (n)."""

    def __init__(self, ch, d, I, tp, fp, tps):
        self.ch = np.asarray(ch)
        rows = np.asarray(I)[None, :]
        self.T, self.B = tp[rows, self.ch], d.bits[rows, self.ch]
        self.F, self.S = fp[rows, self.ch], tps[rows, self.ch]

    def pts(self, pos, N):
        return [(float(b), float(t)) for b, t in zip(self.B[:, pos].mean(1), self.T[:, pos].sum(1) / N)]

    def at(self, B, pos, N):
        P = self.pts(pos, N)
        return hull_at(P, B)

    def fixed_at(self, B, pos, N):
        best, k = -np.inf, None
        for j, (b, r) in enumerate(self.pts(pos, N)):
            if b <= B + 1e-9 and r > best:
                best, k = r, j
        return (best, [(k, 1.0)]) if k is not None else (np.nan, None)


def hull_at(P, B):
    h = G.hull(P)
    if B < P[h[0]][0] - 1e-9:
        return np.nan, None
    if B >= P[h[-1]][0]:
        return P[h[-1]][1], [(h[-1], 1.0)]
    for a, b in zip(h[:-1], h[1:]):
        if P[a][0] - 1e-9 <= B <= P[b][0]:
            w = (B - P[a][0]) / max(P[b][0] - P[a][0], 1e-12)
            return (1 - w) * P[a][1] + w * P[b][1], [(a, 1 - w), (b, w)]
    return P[h[0]][1], [(h[0], 1.0)]


def mixval(fam, ws, pos, what="T"):
    M = getattr(fam, what)
    return sum(w * M[k, pos] for k, w in ws)


class Baselines:
    def __init__(self, d, I, train, menu):
        self.d, self.I = d, np.asarray(I)
        tp, fp, tps = d.tp, d.fp, d.tps
        mu = tp[train].mean(0)
        menu = list(menu)
        n = len(self.I)
        self.F = Fam([np.full(n, m) for m in menu], d, self.I, tp, fp, tps)
        ftr = G.Curve(G.fixed_choices(train, menu), train, d.bits, tp, d.ngt)
        self.MX = Fam([np.full(n, menu[k]) for k in ftr.h], d, self.I, tp, fp, tps)
        self.RV = Fam(G.lagrange_choices(np.broadcast_to(mu, d.bits.shape), d.bits, self.I, menu), d, self.I, tp, fp, tps)
        Bi = d.bits[self.I]
        caps = np.unique(np.quantile(Bi[:, menu], np.linspace(0, 1, 400)))
        order = sorted(menu, key=lambda m: -mu[m])
        cheapest = np.array(menu)[np.argmin(Bi[:, menu], 1)]
        cb = []
        for c in caps:
            ch = np.full(n, -1)
            for m in order:
                ch[(ch < 0) & (Bi[:, m] <= c)] = m
            ch[ch < 0] = cheapest[ch < 0]
            cb.append(ch)
        self.CB = Fam(cb, d, self.I, tp, fp, tps)

    def at(self, B, pos):
        N = self.d.ngt[self.I[pos]].sum()
        c = {"F": self.F.fixed_at(B, pos, N), "MX": self.MX.at(B, pos, N), "RV": self.RV.at(B, pos, N),
             "CB": self.CB.at(B, pos, N)}
        k = max((k for k in c if np.isfinite(c[k][0])), key=lambda k: c[k][0])
        return c[k][0], k, c[k][1], {kk: v[0] for kk, v in c.items()}


_AP = {}


def ap50(d, I, ch, pool):
    """COCO AP50 (a6l1_ap) of sending image I[k] in mode ch[k], every image once."""
    import a6l1_ap as AP
    key = "val" if pool in ("test", "val") else "testdev"
    if key not in _AP:
        b = AP.build(key)
        _AP[key] = (b, {n: i for i, n in enumerate(b["images"])})
    b, idx = _AP[key]
    return AP.ap50([(key, idx[d.images[i]], d.modes[m]) for i, m in zip(I, ch)], {key: b})


def features(d, rows):
    man = {e["name"]: e for e in C.manifest()["images"]}
    by = {r["image"]: r for r in rows}
    kB = sorted(rows[0]["B"])
    kC = sorted(rows[0]["C"])
    n = len(d.images)
    A = np.full((n, len(d.modes) + 1), np.nan)
    Bf = np.full((n, len(kB)), np.nan)
    Cf = np.full((n, len(kC)), np.nan)
    for i, name in enumerate(d.images):
        if name not in by or name not in man:
            continue
        A[i, :-1] = np.log(np.maximum(d.bits[i], 1.0))
        A[i, -1] = np.log(man[name]["w"] * man[name]["h"])
        Bf[i] = [by[name]["B"][k] for k in kB]
        Cf[i] = [np.log1p(by[name]["C"][k]) if k.startswith(("n", "small", "mid")) else by[name]["C"][k] for k in kC]
    return dict(A=A, B=Bf, C=Cf), dict(B=kB, C=kC)


def design(F, fs):
    return np.concatenate([F[k] for k in fs], 1)


def fit_predict(X, Y, groups, Xs, model):
    """Fit on (X, Y) [n, M], predict every array in Xs."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model == "ridge":
        best = None
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            err = 0.0
            for tr, te in GroupKFold(5).split(X, Y, groups):
                m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], Y[tr])
                err += float(((m.predict(X[te]) - Y[te]) ** 2).sum())
            if best is None or err < best[0]:
                best = (err, alpha)
        m = make_pipeline(StandardScaler(), Ridge(alpha=best[1])).fit(X, Y)
        return [m.predict(x) for x in Xs], dict(alpha=best[1])
    outs = [np.zeros((len(x), Y.shape[1])) for x in Xs]
    for j in range(Y.shape[1]):
        if np.allclose(Y[:, j], Y[0, j]):
            for o in outs:
                o[:, j] = Y[0, j]
            continue
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, early_stopping=False,
                                          random_state=0).fit(X, Y[:, j])
        for o, x in zip(outs, Xs):
            o[:, j] = m.predict(x)
    return outs, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    d = G.Data(True)                                        # one table: train / val / test / testdev
    pools = {k: np.asarray(v) for k, v in d.pools.items()}
    train, val = pools["train"], pools["val"]
    Bs = G.budgets(d)
    menu = list(range(len(d.modes)))
    rows = [json.loads(l) for l in open(G.G2 / "features.jsonl")]
    timing = json.loads((G.G2 / "features_timing.json").read_text())
    F, fnames = features(d, rows)
    q3 = d.modes.index("mbt2018-mean_q3")
    evalp = {"val": val, "test": pools["test"], "testdev": pools["testdev"]}
    for p, I in list(evalp.items()) + [("train", train)]:
        assert not np.isnan(design(F, "ABC")[I]).any(), f"features missing ({p})"
    Ytp = d.tp[train]
    preds = {}
    for fl, fs in FSETS.items():
        for target in ("abs", "pair"):
            Y = Ytp if target == "abs" else Ytp - Ytp[:, [q3]]
            for model in ("ridge", "gbm"):
                X = design(F, fs)
                outs, info = fit_predict(X[train], Y, d.seq[train], [X[I] for I in evalp.values()], model)
                preds[(fl, target, model)] = (dict(zip(evalp, outs)), info)
    print(f"fitted {len(preds)} candidates {time.time() - t0:.0f}s", flush=True)

    base = {p: Baselines(d, I, train, menu) for p, I in evalp.items()}
    res = dict(budgets_bits=Bs, timing=timing, features=fnames, candidates={}, final={})
    print("\n## Bước 3: chọn ứng viên trên val (Δ_val = recall − BL ở cùng bit, điểm)\n")
    print("| ứng viên | " + " | ".join(Bs) + " |")
    print("|---|" + "---|" * len(Bs))
    chosen = {}
    fams = {}
    for key, (pr, info) in preds.items():
        cells = []
        for p, I in evalp.items():
            P = np.zeros((len(d.images), len(d.modes)))
            P[I] = pr[p]
            ch = G.lagrange_choices(P, d.bits, I, menu)
            fams[(key, p)] = Fam(ch, d, I, d.tp, d.fp, d.tps)
        pos = np.arange(len(val))
        N = d.ngt[val].sum()
        res["candidates"]["|".join(key)] = {}
        for m, B in Bs.items():
            r, ws = fams[(key, "val")].at(B, pos, N)
            rb = base["val"].at(B, pos)[0]
            dv = (r - rb) * PT
            res["candidates"]["|".join(key)][m] = dict(delta_val=dv, ws=ws, info=info)
            cells.append(f"{dv:+.2f}")
            if m not in chosen or dv > chosen[m][1]:
                chosen[m] = (key, dv, ws)
        print(f"| {' / '.join(key)} | " + " | ".join(cells) + " |")
    print()
    for m, (key, dv, ws) in chosen.items():
        print(f"- {m}: chọn **{' / '.join(key)}** (Δ_val {dv:+.2f}); trộn λ: {[(float(G.LAMS[k]), round(w, 3)) for k, w in ws]}")
    print()

    gate = {}
    for bi, (m, B) in enumerate(Bs.items()):
        key, dv, ws = chosen[m]
        fs = FSETS[key[0]]
        cost_ms = (timing["B_cpu_ms_median"] if "B" in fs else 0.0) + \
                  (timing[f"C_cpu_threads{timing['cpu']}_ms_median"] if "C" in fs else 0.0)
        res["final"][m] = dict(candidate="|".join(key), cost_cpu_ms=cost_ms, pools={})
        for pi, p in enumerate(("test", "testdev")):
            I = evalp[p]
            fam, bl = fams[(key, p)], base[p]
            n = len(I)
            pos = np.arange(n)

            def delta(pos):
                N = d.ngt[I[pos]].sum()
                rec = mixval(fam, ws, pos).sum() / N
                bits = mixval(fam, ws, pos, "B").mean()
                rb, name, bws, allb = bl.at(bits, pos)
                return (rec - rb) * PT, rec * PT, rb * PT, bits, name, allb
            dlt, rec, rb, bits, name, allb = delta(pos)
            seqs = d.seq[I]
            us = np.unique(seqs)
            by = {s: np.where(seqs == s)[0] for s in us}
            rng = np.random.default_rng([557, bi, pi])
            boots = [delta(np.concatenate([by[s] for s in rng.choice(us, len(us), replace=True)]))[0]
                     for _ in range(a.boot)]
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            lovo = {s: delta(np.where(seqs != s)[0])[0] for s in us}
            Np, Ns = d.ngt[I].sum(), d.nsmall[I].sum()
            dom = max(ws, key=lambda kw: kw[1])[0]
            share = np.bincount(fam.ch[dom], minlength=len(d.modes)) / n
            rbv, rbname, rbws, _ = bl.at(bits, pos)
            row = dict(delta=dlt, ci=ci, recall=rec, BL=rb, BL_name=name, BL_all={k: v * PT for k, v in allb.items()},
                       mbit_per_frame=bits / 1e6, lovo_min=float(min(lovo.values())),
                       lovo_n_nonpos=int(sum(v <= 0 for v in lovo.values())),
                       recall_small=float(mixval(fam, ws, pos, "S").sum() / Ns * PT),
                       fp_per_frame=float(mixval(fam, ws, pos, "F").mean()),
                       diversity=float(1 - share.max()), dominant_modes={d.modes[j]: float(s) for j, s in
                                                                        enumerate(share) if s > 0.02})
            # oracle at the same bits (optimistic) and the share of its headroom captured
            O = G.Curve(G.lagrange_choices(d.tp, d.bits, I, menu), I, d.bits, d.tp, d.ngt).at(bits)[0] * PT
            row["oracle"] = O
            row["oracle_share"] = float(dlt / (O - rb)) if O - rb > 1e-9 else float("nan")
            # failure cases: images where the dominant member loses most TP against the RV member nearest in bits
            rvk = int(np.argmin([abs(b - bits) for b, _ in bl.RV.pts(pos, Np)]))
            loss = fam.T[dom] - bl.RV.T[rvk]
            worst = np.argsort(loss)[:8]
            row["failures"] = [dict(image=d.images[I[i]], seq=str(seqs[i]), tp_loss=float(loss[i]),
                                    n_gt=float(d.ngt[I[i]]), mode_pred=d.modes[fam.ch[dom, i]],
                                    mode_rv=d.modes[bl.RV.ch[rvk, i]]) for i in worst]
            row["rv_member_bits"] = float(bl.RV.B[rvk].mean() / 1e6)
            # policy-level COCO AP50 (deterministic members): predictor's dominant member, nearest RV member, F at B
            fk = bl.F.fixed_at(B, pos, Np)[1][0][0]
            row["ap50"] = dict(pred=ap50(d, I, fam.ch[dom], p) * PT, rv=ap50(d, I, bl.RV.ch[rvk], p) * PT,
                               fixed=ap50(d, I, bl.F.ch[fk], p) * PT,
                               bits=dict(pred=float(fam.B[dom].mean() / 1e6), rv=float(bl.RV.B[rvk].mean() / 1e6),
                                         fixed=float(bl.F.B[fk].mean() / 1e6)))
            res["final"][m]["pools"][p] = row
        t, td = res["final"][m]["pools"]["test"], res["final"][m]["pools"]["testdev"]
        c1 = all(x["delta"] >= 1.0 and x["ci"][0] > 0 for x in (t, td))
        c2 = all(x["lovo_n_nonpos"] == 0 for x in (t, td))
        c3 = cost_ms <= 100.0
        c4 = all(x["diversity"] >= 0.20 for x in (t, td))
        part = (sum(x["delta"] >= 1.0 and x["ci"][0] > 0 for x in (t, td)) == 1 or
                (all(x["delta"] >= 1.0 for x in (t, td)) and not c1) or (c1 and not (c2 and c3)))
        gate[m] = dict(c1=c1, c2=c2, c3=c3, c4=c4, verdict="PASS" if (c1 and c2 and c3 and c4) else
                       ("INCONCLUSIVE" if part else "FAIL"))
    res["gate"] = gate
    overall = "PASS" if any(g["verdict"] == "PASS" for g in gate.values()) else \
        ("INCONCLUSIVE" if any(g["verdict"] == "INCONCLUSIVE" for g in gate.values()) else "FAIL")
    res["verdict"] = overall

    print("## Bước 3–4: đánh giá cuối (test / test-dev, một lần)\n")
    print("| mức bit | pool | ứng viên | Mbit/khung | recall | BL (loại) | **Δ** [CI video] | LOVO min | recall nhỏ | FP/khung |"
          " đa dạng | oracle | phần dư địa |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in Bs:
        for p in ("test", "testdev"):
            r = res["final"][m]["pools"][p]
            print(f"| {m} | {p} | {res['final'][m]['candidate']} | {r['mbit_per_frame']:.3f} | {r['recall']:.2f} | "
                  f"{r['BL']:.2f} ({r['BL_name']}) | **{r['delta']:+.2f}** [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | "
                  f"{r['lovo_min']:+.2f} | {r['recall_small']:.2f} | {r['fp_per_frame']:.1f} | {r['diversity']:.2f} | "
                  f"{r['oracle']:.2f} | {r['oracle_share'] * 100:.0f}% |")
    print()
    print("AP50 COCO (thành viên đơn định: predictor trội / RV gần bit nhất / mode cố định tốt nhất ≤ B; Mbit/khung):\n")
    for m in Bs:
        for p in ("test", "testdev"):
            a_ = res["final"][m]["pools"][p]["ap50"]
            print(f"- {m} {p}: predictor {a_['pred']:.2f} ({a_['bits']['pred']:.3f}), RV {a_['rv']:.2f} "
                  f"({a_['bits']['rv']:.3f}), cố định {a_['fixed']:.2f} ({a_['bits']['fixed']:.3f})")
    print()
    for m, g in gate.items():
        print(f"- {m}: điều kiện 1 {g['c1']}, 2 {g['c2']}, 3 {g['c3']} (chi phí {res['final'][m]['cost_cpu_ms']:.0f} ms CPU),"
              f" 4 {g['c4']} → **{g['verdict']}**")
    print(f"\n**Quyết định G-L2: {overall}**")
    print(f"\nChi phí đo: {json.dumps(timing)}")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (G.G2 / "predict.json").write_text(json.dumps(clean(res), indent=1))
    print("PREDICT_DONE", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
