"""R1a robustness addendum (EXPLORATORY, post hoc; the pre-registered R1a verdict INCONCLUSIVE is unchanged).

  1. achieved-bitrate matching: each val-selected configuration vs RV_probe at the bits it ACTUALLY used on the pool,
     with budget-violation ratios;
  2. RV_probe as an EXECUTED policy: the two-member lambda mixture at a bit level is realised per image with a fixed
     seed; recall / FP / COCO AP50 of that realised selection (AP is not interpolated);
  3. score calibration and source-aware merging, fitted on the A5 TRAIN pool only: per source (YOLO11n tokens at
     640 / 960; BS YOLO11s on each background mode; BS on tiles q50 / q80) a logistic map [logit(score), log area] ->
     P(true positive); merge variants: raw scores + NMS 0.6 (as R1a), calibrated + NMS {0.5, 0.6, 0.7}, calibrated +
     per-class cluster fusion at IoU 0.6 with score max / noisy-OR over sources; one variant per family chosen on VAL
     by mean AP50 over the family's selected configurations, then locked. Every variant drops raw scores < 0.05 before
     merging and keeps the 500 best detections per image (VisDrone maxDets), so the raw variant is close to, not
     identical with, R1a;
  4. recall at matched FP/frame: the threshold of each non-image configuration is set on VAL to the FP/frame of the
     executed RV_probe at the same budget on VAL, then applied unchanged on test / test-dev.
Greedy matching in score order makes a detection's TP flag independent of lower-scored detections, so one matching per
image serves every threshold. Detections are built in parallel (fork) and cached.
Writes runs/uav_semcom/r1a/addendum.json. Usage: python r1a_addendum.py MEASURE_DIR [--procs 6]
"""
import argparse
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import a5_common as C
import g2_common as G
import r1a_eval as E
from a5_fidelity import IOU, coco_eval, in_ignore, iou, read_gt
from g2_predict import Fam, ap50

PT = 100.0
VARIANTS = ("raw_nms0.6", "cal_nms0.5", "cal_nms0.6", "cal_nms0.7", "cal_fuse_max", "cal_fuse_nor")
S = {}      # fork-shared state


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def feats(d):
    area = np.maximum((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]), 1.0)
    return np.column_stack([logit(d[:, 4]), np.log(area)])


def match_idx(d, g, gc):
    """Greedy class-aware matching in the given (score-sorted) order -> matched GT index per detection or -1."""
    out = np.full(len(d), -1)
    if len(d) == 0 or len(g) == 0:
        return out
    M = iou(d[:, :4], g)
    used = np.zeros(len(g), bool)
    cls = d[:, 5].astype(int)
    for i in range(len(d)):
        cand = np.where((gc == cls[i]) & ~used & (M[i] >= IOU))[0]
        if len(cand):
            j = cand[np.argmax(M[i, cand])]
            used[j] = True
            out[i] = j
    return out


def fuse(d, src, thr, how):
    """Per-class greedy clustering at IoU >= thr (pairwise IoU computed once); box of the highest-score member;
    score = max or noisy-OR over the per-source maxima of the cluster."""
    out = []
    for c in np.unique(d[:, 5]):
        idx = np.where(d[:, 5] == c)[0]
        idx = idx[np.argsort(-d[idx, 4], kind="mergesort")]
        dc, sc = d[idx], src[idx]
        Mi = iou(dc[:, :4], dc[:, :4])
        alive = np.ones(len(dc), bool)
        for i in range(len(dc)):
            if not alive[i]:
                continue
            mem = np.where(alive & (Mi[i] >= thr))[0]
            per = {}
            for j in mem:
                per[sc[j]] = max(per.get(sc[j], 0.0), float(dc[j, 4]))
            row = dc[i].copy()
            row[4] = max(per.values()) if how == "max" else 1.0 - float(np.prod([1 - v for v in per.values()]))
            out.append(row)
            alive[mem] = False
    out = np.array(out) if out else np.zeros((0, 6), np.float32)
    return out[np.argsort(-out[:, 4], kind="mergesort")] if len(out) else out


def sources(nm, c):
    fam, u, t, bg, q = c
    M, bgd = S["M"], S["bgd"]
    ob = M[nm]["onboard"][u]
    out = [(f"tok{u}", ob[ob[:, 4] >= t])]
    if bg:
        out.append((f"bg:{bg}", bgd.get((nm, bg), np.zeros((0, 6), np.float32))))
    if fam == "ROI":
        unc = ob[(ob[:, 4] >= 0.05) & (ob[:, 4] < t)]
        cx, cy = (unc[:, 0] + unc[:, 2]) / 2, (unc[:, 1] + unc[:, 3]) / 2
        for tl in M[nm]["tiles"]:
            x0, y0, x1, y1 = tl["box"]
            if np.any((cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)):
                out.append((f"tile:{q}", tl[q]["dets"]))
    return out


def build(nm, c, var):
    parts, src = [], []
    for s, dd in sources(nm, c):
        dd = dd[dd[:, 4] >= 0.05].copy()
        if len(dd) and var != "raw_nms0.6":
            dd[:, 4] = S["cal"][s].predict_proba(feats(dd))[:, 1]
        parts.append(dd)
        src += [s] * len(dd)
    dd = np.concatenate(parts) if parts else np.zeros((0, 6), np.float32)
    if len(dd) == 0:
        return dd
    if "nms" in var:
        out = E.nms(dd, float(var.split("nms")[1]))
    else:
        out = fuse(dd, np.array(src), 0.6, "max" if var.endswith("max") else "nor")
    return out[:500]


def job(args):
    nm, todo = args
    g, gc, ig = S["gts"][nm]
    small_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) < 32 ** 2
    res = {}
    for cn, var in todo:
        dd = build(nm, S["cfg"][cn], var)
        if len(dd):
            dd = dd[~in_ignore(dd[:, :4], ig)]
        mi = match_idx(dd, g, gc)
        res[(cn, var)] = dict(d=dd.astype(np.float32), s=dd[:, 4].copy() if len(dd) else np.zeros(0),
                              tp=mi >= 0, small=np.array([small_g[j] if j >= 0 else False for j in mi], bool))
    return nm, res


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("measure")
    ap_.add_argument("--procs", type=int, default=6)
    a = ap_.parse_args()
    t0 = time.time()
    S["M"] = pickle.load(open(Path(a.measure) / "measure.pkl", "rb"))["images"]
    ev = json.loads((C.A5.parent / "r1a" / "eval.json").read_text())
    d = G.Data(True)
    tab = d.tab
    man = {e["name"]: e for e in C.manifest()["images"]}
    pools = {p: np.asarray(d.pools[p]) for p in ("train", "val", "test", "testdev")}
    gts = {}
    for p, I in pools.items():
        for i in I:
            nm = d.images[i]
            gts[nm] = read_gt(E.ANN["testdev" if man[nm]["split"] == "testdev" else "val"] / (Path(nm).stem + ".txt"))
    S["gts"] = gts
    bgd = {}
    for key, path in E.DETS.items():
        D = np.load(path)
        names = [str(x) for x in D["images"]]
        for m in E.BGS:
            arr = D[m]
            for ii in np.unique(arr[:, 0]).astype(int):
                r = arr[arr[:, 0] == ii]
                bgd[(names[ii], m)] = np.column_stack([r[:, 1], r[:, 2], r[:, 1] + r[:, 3], r[:, 2] + r[:, 4],
                                                       r[:, 5], r[:, 6] - 1]).astype(np.float32)
    S["bgd"] = bgd
    S["cfg"] = {E.name(c): c for c in E.configs()}

    # ---- calibration on TRAIN (each source alone, raw score >= 0.05; tiles pooled per image with NMS 0.6)
    from sklearn.linear_model import LogisticRegression
    srcs = [f"tok{u}" for u in (640, 960)] + [f"bg:{m}" for m in E.BGS] + ["tile:avif_50", "tile:avif_80"]
    X, Y = {s: [] for s in srcs}, {s: [] for s in srcs}
    for i in pools["train"]:
        nm = d.images[i]
        g, gc, ig = gts[nm]
        per = {f"tok{u}": S["M"][nm]["onboard"][u] for u in (640, 960)}
        for m in E.BGS:
            per[f"bg:{m}"] = bgd.get((nm, m), np.zeros((0, 6), np.float32))
        for q in ("avif_50", "avif_80"):
            per[f"tile:{q}"] = E.nms(np.concatenate([tl[q]["dets"] for tl in S["M"][nm]["tiles"]]))
        for s, dd in per.items():
            dd = dd[dd[:, 4] >= 0.05]
            dd = dd[~in_ignore(dd[:, :4], ig)] if len(dd) else dd
            dd = dd[np.argsort(-dd[:, 4], kind="mergesort")]
            if len(dd):
                X[s].append(feats(dd)); Y[s].append(match_idx(dd, g, gc) >= 0)
    S["cal"] = {s: LogisticRegression(C=1.0, max_iter=1000).fit(np.concatenate(X[s]), np.concatenate(Y[s])) for s in srcs}
    print(f"calibrated {time.time() - t0:.0f}s", flush=True)

    sel = dict(ev["selected"])
    chosen = sorted(set(sel.values()))
    todo = [(cn, var) for cn in chosen for var in (VARIANTS if not cn.startswith("TOK") else ("raw_nms0.6", "cal_nms0.6"))]
    imgs = [d.images[i] for p in ("val", "test", "testdev") for i in pools[p]]
    cache = {}
    with Pool(a.procs) as pool:
        for k, (nm, r) in enumerate(pool.imap_unordered(job, [(nm, todo) for nm in imgs], chunksize=4)):
            cache[nm] = r
            if k % 100 == 0:
                print(f"built {k}/{len(imgs)} {time.time() - t0:.0f}s", flush=True)

    def ap_of(key, I):
        names = [d.images[i] for i in I]
        rows = [np.column_stack([np.full(len(v), k), v[:, 0], v[:, 1], v[:, 2] - v[:, 0], v[:, 3] - v[:, 1], v[:, 4],
                                 v[:, 5] + 1]) for k, v in enumerate(cache[n][key]["d"] for n in names)]
        return coco_eval(names, gts, {"x": rows}, set(names))["x"]["AP50"] * PT

    def at_thr(key, I, th):
        tp = fp = tps = 0
        for i in I:
            r = cache[d.images[i]][key]
            k = r["s"] >= th
            tp += int((r["tp"] & k).sum()); fp += int((~r["tp"] & k).sum()); tps += int((r["small"] & k).sum())
        return tp / d.ngt[I].sum() * PT, fp / len(I), tps / d.nsmall[I].sum() * PT

    best_var, val_ap = {}, {}
    for fam in ("TOK", "HYB", "ROI"):
        cns = [cn for cn in chosen if cn.startswith(fam)]
        vs = VARIANTS if fam != "TOK" else ("raw_nms0.6", "cal_nms0.6")
        val_ap[fam] = {v: float(np.mean([ap_of((cn, v), pools["val"]) for cn in cns])) for v in vs}
        best_var[fam] = max(val_ap[fam], key=val_ap[fam].get)
    print("variant chosen on val:", best_var, json.dumps(val_ap), flush=True)

    mu = d.tp[pools["train"]].mean(0)
    pred_bits = np.array([tab.predict(i)[0] for i in range(len(d.images))])
    menu = list(range(len(d.modes)))

    def rvprobe(I):
        Bh = pred_bits[I][:, menu]
        ch = [np.array(menu)[np.argmin(Bh, 1) if np.isinf(l) else np.argmax(mu[menu][None, :] - l * Bh, 1)]
              for l in G.LAMS]
        return Fam(ch, d, I, d.tp, d.fp, d.tps)

    def executed(rv, I, bits, p, seed):
        v, ws = rv.at(bits, np.arange(len(I)), d.ngt[I].sum())
        if not ws:
            return None
        if len(ws) == 1:
            ch = rv.ch[ws[0][0]].copy()
        else:
            (k0, _), (k1, w1) = ws
            ch = np.where(np.random.default_rng(seed).random(len(I)) < w1, rv.ch[k1], rv.ch[k0])
        return dict(recall=float(d.tp[I, ch].sum() / d.ngt[I].sum() * PT),
                    small=float(d.tps[I, ch].sum() / d.nsmall[I].sum() * PT), fp=float(d.fp[I, ch].mean()),
                    mbit=float(d.bits[I, ch].mean() / 1e6), ap50=float(ap50(d, I, ch, p) * PT),
                    interp_recall=float(v * PT))

    # FP-matching thresholds on VAL
    thr = {}
    rv_val = rvprobe(pools["val"])
    for bi, B in enumerate(E.BUDGETS):
        ex = executed(rv_val, pools["val"], B, "val", [560, bi, 9])
        for key, cn in sel.items():
            f, b = key.split("@")
            if abs(float(b) * 1e6 - B) > 1 or ex is None:
                continue
            k = (cn, best_var[f])
            sc = np.unique(np.concatenate([cache[d.images[i]][k]["s"] for i in pools["val"]] + [np.zeros(1)]))
            grid = np.unique(np.quantile(sc, np.linspace(0, 1, 400)))
            fps = np.array([at_thr(k, pools["val"], t)[1] for t in grid])
            thr[key] = float(grid[int(np.argmin(np.abs(fps - ex["fp"])))])
    res = dict(variant_val_ap50=val_ap, variant_chosen=best_var, thresholds=thr, pools={})
    for pi, p in enumerate(("test", "testdev")):
        I = pools[p]
        rv = rvprobe(I)
        res["pools"][p] = {}
        for bi, B in enumerate(E.BUDGETS):
            bkey = f"{B / 1e6:g}"
            exB = executed(rv, I, B, p, [560, bi, pi])
            rows = {}
            for key, cn in sel.items():
                f, b = key.split("@")
                if abs(float(b) * 1e6 - B) > 1:
                    continue
                raw = ev["pools"][p][bkey]["fam"][f]
                exA = executed(rv, I, raw["mbit"] * 1e6, p, [561, bi, pi])
                k = (cn, best_var[f])
                r_th, fp_th, s_th = at_thr(k, I, thr.get(key, 0.25))
                rows[f] = dict(config=cn, variant=best_var[f], mbit=raw["mbit"], budget_violation=raw["mbit"] * 1e6 / B - 1,
                               recall_raw=raw["recall"], ap50_raw=raw["ap50"], fp_raw=raw["fp_frame"],
                               rv_at_achieved=exA, delta_at_achieved=(raw["recall"] - exA["recall"]) if exA else None,
                               ap50_cal=ap_of(k, I), ap50_raw_prefilter=ap_of((cn, "raw_nms0.6"), I),
                               thr=thr.get(key), recall_at_matched_fp=r_th, fp_at_thr=fp_th, small_at_matched_fp=s_th,
                               on_ms=raw["on_ms"], bs_ms=raw["bs_ms"])
            res["pools"][p][bkey] = dict(rv_executed=exB, fam=rows)
        print(f"pool {p} done {time.time() - t0:.0f}s", flush=True)
    (C.A5.parent / "r1a" / "addendum.json").write_text(json.dumps(res, indent=1, default=float))

    print("\n## R1a addendum (khám phá, sau khi xem kết quả)\n")
    print("Biến thể gộp chọn trên val:", best_var)
    print("AP50 val theo biến thể:", json.dumps(val_ap), "\n")
    print("| pool | B | RV_probe thực thi tại B: recall / FP / AP50 / Mbit | họ (biến thể) | Mbit thực (so với B) "
          "| Δ recall tại bit thực [RV thực thi ở bit đó: recall, AP50] | AP50: R1a → lọc 0,05 → hiệu chỉnh "
          "| recall @ cùng FP/khung với RV (FP đạt) | recall nhỏ @ cùng FP | ms UAV / BS |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p, rowsB in res["pools"].items():
        for B, x in rowsB.items():
            ex = x["rv_executed"]
            exs = f"{ex['recall']:.2f} / {ex['fp']:.1f} / {ex['ap50']:.2f} / {ex['mbit']:.3f}" if ex else "—"
            for f, r in x["fam"].items():
                exA = r["rv_at_achieved"]
                dA = (f"{r['delta_at_achieved']:+.2f} [{exA['recall']:.2f}, {exA['ap50']:.2f}]" if exA
                      else "— (ảnh không chạy ở bit này)")
                print(f"| {p} | {B} | {exs} | {f} ({r['variant']}) | {r['mbit']:.3f} ({r['budget_violation'] * 100:+.0f}%) "
                      f"| {dA} | {r['ap50_raw']:.2f} → {r['ap50_raw_prefilter']:.2f} → {r['ap50_cal']:.2f} | "
                      f"{r['recall_at_matched_fp']:.2f} ({r['fp_at_thr']:.1f}) | {r['small_at_matched_fp']:.2f} | "
                      f"{r['on_ms']:.0f} / {r['bs_ms']:.0f} |")
    print(f"\nADDENDUM_DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
