"""R1a addendum part 2 (EXPLORATORY, post hoc; the pre-registered R1a verdict is unchanged): a deployable,
validation-locked image baseline replacing the post-hoc RV_probe envelope.

Problem fixed: RV_probe in r1a_eval.py / r1a_addendum.py (Fam.at -> hull_at) builds the rate-recall hull from the TP
and true bits of the EVALUATION pool, so the lambda members, their mixing weight and the operating point hitting B are
chosen with test outcomes (and refitted inside every bootstrap replicate). That curve is kept as
RV_probe_posthoc_envelope (reference only). Two deployable versions:

  RV_locked(T)  : payload predictor fitted on train (tab.predict, AVIF-5 probe), mu = train mean TP per mode; on VAL
                  the hull of the lambda family gives the two members and weight that hit mean bits T; that rule
                  (lambda indices, weight, realisation seed) is applied unchanged to test / test-dev. Achieved bits on
                  test may differ from T (as for the val-locked non-image configurations).
  RV_online(T)  : same per-image rule argmax_m mu_m - lam_t * bhat_i(m); log lam_t = log lam_0 + kappa * D_t / T with
                  D_t = sum_{s<t} (b_s - T) the bits already sent minus the target (integral controller, only past
                  observed bits). lam_0 from RV_locked(T) on val; kappa chosen on VAL (max val recall s.t. val mean
                  bits <= 1.02 T). Images streamed in name order (sequence, frame).

Comparisons per pool / budget / val-selected family configuration (R1a raw scoring, conf 0.25; per-image TP rebuilt
exactly as r1a_eval.py): Delta recall vs RV_locked(B) and vs RV_online(achieved bits of the family), video cluster
bootstrap with the policies FIXED (no refitting per replicate); AP50 / FP / small recall of the executed image
selections; recall at matched FP (thresholds from addendum.json, set on val) vs RV_locked(B).
Also a G-L2 check: the G-L2 predictor (lambda mixture locked on val, realised bits) vs RV_true / RV_probe locked /
online at its realised bits, instead of the test-pool envelope used by the G-L2 gate.
Writes runs/uav_semcom/r1a/vallocked.json. Usage: python r1a_vallocked.py MEASURE_DIR [--boot 2000]
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import a5_common as C
import g2_common as G
import r1a_eval as E
from a5_fidelity import read_gt
from g2_predict import Fam, ap50

PT = 100.0
LMIN, LMAX = np.log(1e-12), np.log(1.0)
KAPPAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("measure")
    ap_.add_argument("--boot", type=int, default=2000)
    a = ap_.parse_args()
    M = pickle.load(open(Path(a.measure) / "measure.pkl", "rb"))["images"]
    ev = json.loads((C.A5.parent / "r1a" / "eval.json").read_text())
    ad = json.loads((C.A5.parent / "r1a" / "addendum.json").read_text())
    cav = json.loads((C.A5.parent / "g_l2" / "caveats.json").read_text())
    d = G.Data(True)
    tab = d.tab
    man = {e["name"]: e for e in C.manifest()["images"]}
    nI = len(d.images)
    order = lambda I: np.asarray(sorted(I, key=lambda i: d.images[i]))
    full = {p: order(d.pools[p]) for p in ("train", "val", "test", "testdev")}
    pools = {p: order([i for i in d.pools[p] if d.images[i] in M]) for p in full}
    print("images per pool (measured / all):", {p: (len(pools[p]), len(full[p])) for p in full}, flush=True)
    menu = np.arange(len(d.modes))
    mu = d.tp[full["train"]].mean(0)
    pred_bits = np.array([tab.predict(i)[0] for i in range(nI)])

    # ---------------- image policies
    def lam_family(I, bh):
        Bh = bh[I]
        ch = [menu[np.argmin(Bh, 1) if np.isinf(l) else np.argmax(mu[None, :] - l * Bh, 1)] for l in G.LAMS]
        return Fam(ch, d, I, d.tp, d.fp, d.tps)

    def lock(T, bh, Ival):
        """Rule chosen on VAL: [(lambda index, weight)] hitting mean bits T on val (val hull)."""
        fv = lam_family(Ival, bh)
        v, ws = fv.at(T, np.arange(len(Ival)), d.ngt[Ival].sum())
        if ws is None:                                  # T below the cheapest val point -> cheapest member
            ws = [(int(np.argmin(fv.B.mean(1))), 1.0)]
        return ws

    def apply_lock(ws, I, bh, seed):
        f = lam_family(I, bh)
        if len(ws) == 1:
            return f.ch[ws[0][0]].copy()
        (k0, _), (k1, w1) = ws
        return np.where(np.random.default_rng(seed).random(len(I)) < w1, f.ch[k1], f.ch[k0])

    def lam0(ws):
        k = max(ws, key=lambda kw: kw[1])[0]
        l = G.LAMS[k]
        return LMIN if l <= 0 else (LMAX if np.isinf(l) else float(np.clip(np.log(l), LMIN, LMAX)))

    def online(I, T, kappa, l0, bh):
        ch = np.empty(len(I), int)
        D = 0.0
        for t, i in enumerate(I):
            lam = np.exp(np.clip(l0 + kappa * D / T, LMIN, LMAX))
            ch[t] = menu[np.argmax(mu - lam * bh[i])]
            D += d.bits[i, ch[t]] - T
        return ch

    def met(I, ch, p, with_ap=True):
        r = dict(recall=float(d.tp[I, ch].sum() / d.ngt[I].sum() * PT),
                 small=float(d.tps[I, ch].sum() / d.nsmall[I].sum() * PT), fp=float(d.fp[I, ch].mean()),
                 mbit=float(d.bits[I, ch].mean() / 1e6))
        if with_ap:
            r["ap50"] = float(ap50(d, I, ch, p) * PT)
        return r

    def pick_kappa(T, bh, Ival, ws):
        best, bk, rows = None, None, {}
        for k in KAPPAS:
            ch = online(Ival, T, k, lam0(ws), bh)
            m = met(Ival, ch, "val", False)
            rows[k] = m
            ok = m["mbit"] * 1e6 <= 1.02 * T
            key = (ok, m["recall"] if ok else -m["mbit"])
            if best is None or key > best:
                best, bk = key, k
        return bk, rows

    def policies(T, bh, Ival, I, p, seed):
        ws = lock(T, bh, Ival)
        k, krows = pick_kappa(T, bh, Ival, ws)
        chL = apply_lock(ws, I, bh, seed)
        chO = online(I, T, k, lam0(ws), bh)
        return dict(lock_ws=[(int(kk), float(w)) for kk, w in ws], kappa=k, kappa_val=krows), chL, chO

    # ---------------- per-image TP of the val-selected non-image configurations (as r1a_eval.py)
    sel = dict(ev["selected"])
    cfg = {E.name(c): c for c in E.configs()}
    bgd = {}
    for key, path in E.DETS.items():
        D_ = np.load(path)
        names = [str(x) for x in D_["images"]]
        for m in E.BGS:
            arr = D_[m]
            for ii in np.unique(arr[:, 0]).astype(int):
                r = arr[arr[:, 0] == ii]
                bgd[(names[ii], m)] = np.column_stack([r[:, 1], r[:, 2], r[:, 1] + r[:, 3], r[:, 2] + r[:, 4],
                                                       r[:, 5], r[:, 6] - 1]).astype(np.float32)
    FT = {cn: {k: np.full(nI, np.nan) for k in ("tp", "bits")} for cn in set(sel.values())}
    for p in ("test", "testdev"):
        for i in pools[p]:
            nm = d.images[i]
            gt = read_gt(E.ANN["testdev" if man[nm]["split"] == "testdev" else "val"] / (Path(nm).stem + ".txt"))
            for cn in FT:
                fam, u, t, bg, q = cfg[cn]
                ob = M[nm]["onboard"][u]
                tok = ob[ob[:, 4] >= t]
                parts, bits = [tok], E.TOKEN_HDR + E.TOKEN_BOX * len(tok)
                if bg:
                    parts.append(bgd.get((nm, bg), np.zeros((0, 6), np.float32)))
                    bits += d.bits[i, d.modes.index(bg)]
                if fam == "ROI":
                    unc = ob[(ob[:, 4] >= 0.05) & (ob[:, 4] < t)]
                    cx, cy = (unc[:, 0] + unc[:, 2]) / 2, (unc[:, 1] + unc[:, 3]) / 2
                    for tl in M[nm]["tiles"]:
                        x0, y0, x1, y1 = tl["box"]
                        if np.any((cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)):
                            parts.append(tl[q]["dets"])
                            bits += E.TILE_HDR + 8 * tl[q]["bytes"]
                dd = E.nms(np.concatenate(parts)) if fam != "TOK" else tok
                FT[cn]["tp"][i], FT[cn]["bits"][i] = E.score(dd, gt)[0], bits
    print("family per-image TP rebuilt", flush=True)

    res = dict(note="EXPLORATORY post hoc; RV_posthoc = old envelope (test hull), RV_locked / RV_online deployable",
               kappas=KAPPAS, pools={}, gl2={})
    for pi, p in enumerate(("test", "testdev")):
        I = pools[p]
        seqs = d.seq[I]
        us = np.unique(seqs)
        by = {s: np.where(seqs == s)[0] for s in us}
        res["pools"][p] = {}
        for bi, B in enumerate(E.BUDGETS):
            bkey = f"{B / 1e6:g}"
            info, chL, chO = policies(B, pred_bits, pools["val"], I, p, [562, bi, pi])
            mL, mO = met(I, chL, p), met(I, chO, p)
            rng = np.random.default_rng([563, bi, pi])
            reps = [np.concatenate([by[s] for s in rng.choice(us, len(us), replace=True)]) for _ in range(a.boot)]
            rec = lambda tp, pp: tp[pp].sum() / d.ngt[I[pp]].sum() * PT
            tpL, tpO = d.tp[I, chL], d.tp[I, chO]
            row = dict(policy=info, RV_locked=mL, RV_online=mO,
                       RV_posthoc=ev["pools"][p][bkey]["RV_probe"],
                       RV_posthoc_executed=ad["pools"][p][bkey]["rv_executed"], fam={})
            for key, cn in sel.items():
                f, b = key.split("@")
                if abs(float(b) * 1e6 - B) > 1:
                    continue
                raw, add = ev["pools"][p][bkey]["fam"][f], ad["pools"][p][bkey]["fam"][f]
                tpF = FT[cn]["tp"][I]
                bf = float(np.mean(FT[cn]["bits"][I]))
                assert abs(rec(tpF, np.arange(len(I))) - raw["recall"]) < 1e-6, (cn, p, B)
                infoA, chLA, chOA = policies(bf, pred_bits, pools["val"], I, p, [564, bi, pi])
                mLA, mOA = met(I, chLA, p), met(I, chOA, p)
                tpOA = d.tp[I, chOA]
                bootL = np.array([rec(tpF, pp) - rec(tpL, pp) for pp in reps])
                bootO = np.array([rec(tpF, pp) - rec(tpOA, pp) for pp in reps])
                ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
                row["fam"][f] = dict(
                    config=cn, recall=raw["recall"], mbit=bf / 1e6, ap50_raw=raw["ap50"], ap50_cal=add["ap50_cal"],
                    fp=raw["fp_frame"], small=raw["recall_small"],
                    delta_vs_locked_B=raw["recall"] - mL["recall"], ci_locked_B=ci(bootL),
                    rv_online_at_achieved=mOA, rv_locked_at_achieved=mLA, policy_at_achieved=infoA,
                    delta_vs_online_achieved=raw["recall"] - mOA["recall"], ci_online_achieved=ci(bootO),
                    delta_vs_posthoc_achieved=add["delta_at_achieved"],
                    recall_at_matched_fp=add["recall_at_matched_fp"], fp_at_thr=add["fp_at_thr"],
                    small_at_matched_fp=add["small_at_matched_fp"])
            res["pools"][p][bkey] = row
            print(f"{p} B={bkey}: locked {mL['recall']:.2f}@{mL['mbit']:.3f} online {mO['recall']:.2f}@{mO['mbit']:.3f}"
                  f" posthoc {row['RV_posthoc']}", flush=True)

        # ---- G-L2 check (full pools as G-L2): predictor at its realised bits vs locked / online baselines
        Ifull = full[p]
        res["gl2"][p] = {}
        for li, (lvl, c) in enumerate(cav[p].items()):
            T = c["pred_bits"] * 1e6
            out = dict(pred=c["pred"], pred_mbit=c["pred_bits"], RV_probe_posthoc_at_pred_bits=c["RV_probe_at_pred_bits"])
            for nm_, bh in (("RV_true", d.bits.astype(float)), ("RV_probe", pred_bits)):
                inf, chL_, chO_ = policies(T, bh, full["val"], Ifull, p, [565, li, pi])
                out[f"{nm_}_locked"] = met(Ifull, chL_, p, False)
                out[f"{nm_}_online"] = met(Ifull, chO_, p, False)
            fv = [m for m in menu if d.bits[full["val"], m].mean() <= T]
            fm = max(fv, key=lambda m: d.tp[full["val"], m].sum()) if fv else int(np.argmin(d.bits[full["val"]].mean(0)))
            out["F_locked"] = dict(mode=d.modes[fm], **met(Ifull, np.full(len(Ifull), fm), p, False))
            fh = lam_family(Ifull, d.bits.astype(float))
            out["RV_true_posthoc_at_pred_bits"] = float(fh.at(T, np.arange(len(Ifull)), d.ngt[Ifull].sum())[0] * PT)
            res["gl2"][p][lvl] = out
            print(f"G-L2 {p} {lvl}: pred {c['pred']:.2f}@{c['pred_bits']:.3f} | RV_true online "
                  f"{out['RV_true_online']['recall']:.2f}@{out['RV_true_online']['mbit']:.3f} locked "
                  f"{out['RV_true_locked']['recall']:.2f}@{out['RV_true_locked']['mbit']:.3f} posthoc "
                  f"{out['RV_true_posthoc_at_pred_bits']:.2f}", flush=True)

    (C.A5.parent / "r1a" / "vallocked.json").write_text(json.dumps(res, indent=1, default=float))
    print("VALLOCKED_DONE")


if __name__ == "__main__":
    main()
