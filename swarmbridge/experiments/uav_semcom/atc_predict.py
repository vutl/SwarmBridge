"""Review #46 step C, D1: causal on-board object-count predictors (protocol reports/phases/2026-11-17_paperA-scheduling/swarmbridge-atc-protocol.md).

Target log(1 + n_gt) (labels of the TRAIN pool only). Ridge on standardised features, alpha in {0.1, 1, 10, 100, 1000}
by 5-fold GroupKFold over train sequences.
  pA  log probe payload (AVIF-5, encoded on board in every arm) + log pixels
  pB  pA + the 13 G-L2 image statistics (runs/uav_semcom/g_l2/features.jsonl, 'B')
  pC  pA + the 14 G-L2 light-detector features ('C'; A5 detector at imgsz 480 on the original frame -> optimistic)
Train-pool predictions are OUT-OF-FOLD (GroupKFold(5) by sequence; alpha chosen inside each fold's training part);
every other frame uses the model fitted on all train frames. Offline accuracy is printed for the VAL pool only
(test-dev labels are not read here).
Output: runs/uav_semcom/atc/pred.json  {n_hat: {pA: {image: n}, ...}, alpha, val: {...}, cost_ms}
Usage: python atc_predict.py
"""
import json

import numpy as np

import a5_common as C

OUT = C.REPO / "runs" / "uav_semcom" / "atc"
FEAT = C.REPO / "runs" / "uav_semcom" / "g_l2" / "features.jsonl"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
COST_MS = {"pA": 0.0, "pB": 10.0, "pC": 35.0}          # G-L2 features_timing.json medians (CPU, 1 thread), rounded up
POOLS = ("train", "val", "testdev")


def check_independence(man):
    """No video sequence and no image hash shared between the train, val and test-dev pools."""
    seqs = {p: {e["seq"] for e in man if e["split"] == p} for p in POOLS}
    shas = {p: {e["sha256"] for e in man if e["split"] == p} for p in POOLS}
    for i, a in enumerate(POOLS):
        for b in POOLS[i + 1:]:
            assert not seqs[a] & seqs[b], (a, b, seqs[a] & seqs[b])
            assert not shas[a] & shas[b], (a, b)
    return {p: dict(frames=sum(e["split"] == p for e in man), videos=len(seqs[p])) for p in POOLS}


def design(names, tab, man, feats, fset):
    pj = tab.probe_j
    row = {n: i for i, n in enumerate(tab.images)}
    kB, kC = sorted(next(iter(feats.values()))["B"]), sorted(next(iter(feats.values()))["C"])
    X = []
    for n in names:
        x = [np.log(tab.bits[row[n], pj]), np.log(man[n]["w"] * man[n]["h"])]
        if fset == "pB":
            x += [feats[n]["B"][k] for k in kB]
        if fset == "pC":
            x += [np.log1p(feats[n]["C"][k]) if k.startswith(("n", "small", "mid")) else feats[n]["C"][k] for k in kC]
        X.append(x)
    return np.asarray(X, float)


def fit(X, y, groups):
    """alpha by GroupKFold(5) over groups, then refit on all of (X, y)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    best = None
    for a in ALPHAS:
        err = 0.0
        for tr, te in GroupKFold(5).split(X, y, groups):
            m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X[tr], y[tr])
            err += float(((m.predict(X[te]) - y[te]) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, a)
    return make_pipeline(StandardScaler(), Ridge(alpha=best[1])).fit(X, y), best[1]


def main():
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold
    OUT.mkdir(parents=True, exist_ok=True)
    tab, _ = C.load_table(True)
    man_l = C.manifest()["images"]
    indep = check_independence(man_l)
    man = {e["name"]: e for e in man_l}
    feats = {r["image"]: r for r in map(json.loads, open(FEAT))}
    ngt = {}
    for line in open(C.REPO / "runs" / "uav_semcom" / "a6l1" / "fidelity" / "per_image_val.jsonl"):   # train + val only
        r = json.loads(line)
        if r["mode"] == "original":
            ngt[r["image"]] = r["n_gt"]
    names = [n for n in tab.images if n in man and n in feats]
    tr = [n for n in names if man[n]["split"] == "train"]
    va = [n for n in names if man[n]["split"] == "val"]
    ytr = np.log1p([ngt[n] for n in tr])
    gtr = np.array([man[n]["seq"] for n in tr])
    out = dict(n_hat={}, alpha={}, alpha_oof={}, val={}, cost_ms=COST_MS, independence=indep, n_train=len(tr))
    for fs in ("pA", "pB", "pC"):
        Xtr, Xall = design(tr, tab, man, feats, fs), design(names, tab, man, feats, fs)
        model, a = fit(Xtr, ytr, gtr)
        pred = dict(zip(names, model.predict(Xall)))
        oof, al = np.zeros(len(tr)), []
        for k_tr, k_te in GroupKFold(5).split(Xtr, ytr, gtr):
            m, ak = fit(Xtr[k_tr], ytr[k_tr], gtr[k_tr])
            oof[k_te] = m.predict(Xtr[k_te])
            al.append(ak)
        for n, p in zip(tr, oof):
            pred[n] = p
        out["n_hat"][fs] = {n: float(max(np.expm1(p), 0.0)) for n, p in pred.items()}
        out["alpha"][fs], out["alpha_oof"][fs] = a, al
        pv = np.array([pred[n] for n in va])
        yv = np.log1p([ngt[n] for n in va])
        out["val"][fs] = dict(spearman=float(spearmanr(pv, yv)[0]), rmse_log1p=float(np.sqrt(np.mean((pv - yv) ** 2))),
                              rmse_log1p_const=float(np.sqrt(np.mean((ytr.mean() - yv) ** 2))),
                              spearman_oof_train=float(spearmanr(oof, ytr)[0]))
        print(f"{fs}: alpha {a} (oof folds {al})  val spearman {out['val'][fs]['spearman']:.3f}  "
              f"rmse log1p {out['val'][fs]['rmse_log1p']:.3f} (train-mean constant {out['val'][fs]['rmse_log1p_const']:.3f})"
              f"  oof-train spearman {out['val'][fs]['spearman_oof_train']:.3f}")
    json.dump(out, open(OUT / "pred.json", "w"))
    print("independence:", indep)


if __name__ == "__main__":
    main()
