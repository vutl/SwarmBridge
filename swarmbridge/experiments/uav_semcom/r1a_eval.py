"""R1a evaluation (reports/swarmbridge-r1a.md section 0), CPU, from r1a_measure.py outputs + the L1 per-image table
and detections (IMG family, HYB backgrounds).

Per (image, configuration) of TOK / HYB / ROI / ROI-oracle: final BS detection set (union + class-wise NMS 0.6),
ignore-region filter, recall / small recall / FP at conf >= 0.25 (IoU 0.5, class-aware greedy, as A5.2), bits (all
headers / tiles / tokens / backgrounds), onboard ms (YOLO11n CPU all threads + encodes), BS ms (YOLO11s runs).
Selection per family and budget on val (0.5), one evaluation on test / test-dev with video cluster bootstrap,
COCO AP50 of the selected configurations, Pareto table, failure cases, PASS / INCONCLUSIVE / FAIL (0.6).
Usage: python r1a_eval.py MEASURE_DIR [--boot 2000]
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import a5_common as C
import g2_common as G
from a5_fidelity import coco_eval, in_ignore, match, read_gt
from g2_predict import Fam, ap50

PT = 100.0
BUDGETS = (0.005e6, 0.02e6, 0.08e6, 0.3e6, 1.2e6)
REAL = (0.08e6, 0.3e6, 1.2e6)
TOKEN_HDR, TOKEN_BOX, TILE_HDR = 48, 52, 24
ANN = {"val": C.REPO / "data/visdrone/VisDrone2019-DET-val/annotations",
       "testdev": Path("/home/uservutl/data/visdrone_dl/VisDrone2019-DET-test-dev/annotations")}
DETS = {"val": Path("/home/uservutl/work/a6l1/dets_val.npz"), "testdev": Path("/home/uservutl/work/a6l1/dets_testdev.npz")}
BGS = ("avif_5", "mbt2018-mean_q1", "avif_20", "mbt2018-mean_q3")


def configs():
    out = []
    for u in (640, 960):
        for t in (0.1, 0.25, 0.4):
            out.append(("TOK", u, t, None, None))
        for t in (0.25, 0.4):
            for bg in BGS:
                out.append(("HYB", u, t, bg, None))
        for fam in ("ROI", "ROIO"):
            for t in (0.25, 0.4):
                for q in ("avif_50", "avif_80"):
                    for bg in (None, "avif_5"):
                        out.append((fam, u, t, bg, q))
    return out


def name(c):
    f, u, t, bg, q = c
    return f"{f}|u{u}|t{t}" + (f"|bg={bg}" if bg else "") + (f"|q={q}" if q else "")


def nms(d, thr=0.6):
    """Class-wise greedy NMS on rows [x1, y1, x2, y2, score, cls]."""
    if len(d) == 0:
        return d
    keep = []
    for c in np.unique(d[:, 5]):
        idx = np.where(d[:, 5] == c)[0]
        idx = idx[np.argsort(-d[idx, 4], kind="mergesort")]
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            b, r = d[i], d[idx[1:]]
            x1 = np.maximum(b[0], r[:, 0]); y1 = np.maximum(b[1], r[:, 1])
            x2 = np.minimum(b[2], r[:, 2]); y2 = np.minimum(b[3], r[:, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            ar = (b[2] - b[0]) * (b[3] - b[1]) + (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]) - inter
            idx = idx[1:][inter / np.maximum(ar, 1e-9) < thr]
    d = d[keep]
    return d[np.argsort(-d[:, 4], kind="mergesort")]


def score(d, gt):
    g, gc, ig = gt
    d = d[~in_ignore(d[:, :4], ig)] if len(d) else d
    op = d[d[:, 4] >= 0.25] if len(d) else d
    small = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) < 32 ** 2
    mg, md = match(op[:, :4], op[:, 5].astype(int), g, gc) if len(op) else (np.zeros(len(g), bool), np.zeros(0, bool))
    return int(mg.sum()), int(len(op) - md.sum()), int((mg & small).sum()), mg, small, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("measure")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    M = pickle.load(open(Path(a.measure) / "measure.pkl", "rb"))["images"]
    T = json.loads((Path(a.measure) / "timing.json").read_text())
    d = G.Data(True)
    tab = d.tab
    man = {e["name"]: e for e in C.manifest()["images"]}
    pools = {p: np.asarray([i for i in d.pools[p] if d.images[i] in M]) for p in ("train", "val", "test", "testdev")}
    print("images per pool (measured):", {p: len(v) for p, v in pools.items()},
          "of", {p: len(d.pools[p]) for p in pools}, flush=True)
    pool_of = {}
    for p, I in pools.items():
        for i in I:
            pool_of[int(i)] = p
    # background detections (L1, BS detector on decoded backgrounds), keyed by (image name, mode)
    bgd = {}
    for key, path in DETS.items():
        D = np.load(path)
        names = [str(x) for x in D["images"]]
        for m in BGS:
            arr = D[m]
            for ii in np.unique(arr[:, 0]).astype(int):
                r = arr[arr[:, 0] == ii]
                bgd[(names[ii], m)] = np.column_stack([r[:, 1], r[:, 2], r[:, 1] + r[:, 3], r[:, 2] + r[:, 4],
                                                       r[:, 5], r[:, 6] - 1]).astype(np.float32)
    enc = {m: float(tab.enc_s[d.modes.index(m)]) for m in d.modes}
    on_ms = {u: T[f"onboard{u}_cpu{T['cpu']}_ms"] for u in (640, 960)}
    cfgs = configs()
    I_all = np.concatenate([pools[p] for p in pools])
    nI = len(d.images)
    R = {name(c): {k: np.full(nI, np.nan) for k in ("tp", "fp", "tps", "bits", "on_ms", "bs_ms", "ntile")} for c in cfgs}
    dets_for_ap = {name(c): {} for c in cfgs}
    gts = {}
    for i in I_all:
        nm = d.images[i]
        e = man[nm]
        gts[nm] = read_gt(ANN["testdev" if e["split"] == "testdev" else "val"] / (Path(nm).stem + ".txt"))
        rec = M[nm]
        for c in cfgs:
            fam, u, t, bg, q = c
            ob = rec["onboard"][u]
            tok = ob[ob[:, 4] >= t]
            parts = [tok]
            bits = TOKEN_HDR + TOKEN_BOX * len(tok)
            o_ms = on_ms[u]
            b_ms = 0.0
            nt = 0
            if bg:
                parts.append(bgd.get((nm, bg), np.zeros((0, 6), np.float32)))
                bits += d.bits[i, d.modes.index(bg)]
                o_ms += enc[bg] * 1e3
                b_ms += T["server_frame_gpu_ms"]
            if fam in ("ROI", "ROIO"):
                if fam == "ROI":
                    unc = ob[(ob[:, 4] >= 0.05) & (ob[:, 4] < t)]
                    cx, cy = (unc[:, 0] + unc[:, 2]) / 2, (unc[:, 1] + unc[:, 3]) / 2
                else:   # oracle: tiles holding the centre of a GT object missed by the tokens
                    g, gc, ig = gts[nm]
                    op = tok[tok[:, 4] >= 0.25]
                    mg, _ = match(op[:, :4], op[:, 5].astype(int), g, gc) if len(op) else (np.zeros(len(g), bool), None)
                    miss = g[~mg]
                    cx, cy = (miss[:, 0] + miss[:, 2]) / 2, (miss[:, 1] + miss[:, 3]) / 2
                for tl in rec["tiles"]:
                    x0, y0, x1, y1 = tl["box"]
                    if np.any((cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)):
                        parts.append(tl[q]["dets"])
                        bits += TILE_HDR + 8 * tl[q]["bytes"]
                        o_ms += tl[q]["enc_s"] * 1e3
                        b_ms += T["server_tile_gpu_ms"]
                        nt += 1
            dd = nms(np.concatenate(parts)) if fam != "TOK" else tok
            tp, fp, tps, _, _, dkept = score(dd, gts[nm])
            r = R[name(c)]
            r["tp"][i], r["fp"][i], r["tps"][i], r["bits"][i], r["on_ms"][i], r["bs_ms"][i], r["ntile"][i] = \
                tp, fp, tps, bits, o_ms, b_ms, nt
            dets_for_ap[name(c)][nm] = dkept
    print("built", len(cfgs), "configs", flush=True)

    # IMG family: RV_probe (deployable) and fixed modes
    mu = d.tp[pools["train"]].mean(0)
    pred_bits = np.array([tab.predict(i)[0] for i in range(nI)])
    menu = list(range(len(d.modes)))

    def rvprobe(I):
        Bh = pred_bits[I][:, menu]
        ch = [np.array(menu)[np.argmin(Bh, 1) if np.isinf(l) else np.argmax(mu[menu][None, :] - l * Bh, 1)]
              for l in G.LAMS]
        return Fam(ch, d, I, d.tp, d.fp, d.tps)

    def rec_of(cn, I):
        r = R[cn]
        return r["tp"][I].sum() / d.ngt[I].sum()

    # selection on val
    val = pools["val"]
    fams = ("TOK", "HYB", "ROI")
    sel = {}
    for f in fams:
        for B in BUDGETS:
            cands = [name(c) for c in cfgs if c[0] == f and np.nanmean(R[name(c)]["bits"][val]) <= B]
            if cands:
                sel[(f, B)] = max(cands, key=lambda cn: (round(rec_of(cn, val), 12), -np.nanmean(R[cn]["bits"][val])))
    res = dict(timing=T, selected={f"{f}@{B / 1e6:g}": cn for (f, B), cn in sel.items()}, pools={}, configs={})

    # per-config table (all pools) for the Pareto view
    for c in cfgs:
        cn = name(c)
        res["configs"][cn] = {p: dict(recall=float(rec_of(cn, I) * PT),
                                      recall_small=float(R[cn]["tps"][I].sum() / d.nsmall[I].sum() * PT),
                                      fp_frame=float(np.mean(R[cn]["fp"][I])), mbit=float(np.mean(R[cn]["bits"][I]) / 1e6),
                                      on_ms=float(np.mean(R[cn]["on_ms"][I])), bs_ms=float(np.mean(R[cn]["bs_ms"][I])),
                                      tiles=float(np.mean(R[cn]["ntile"][I])))
                              for p, I in pools.items()}

    # final evaluation (test / test-dev)
    verdict_rows = {}
    for pi, p in enumerate(("test", "testdev")):
        I = pools[p]
        pos = np.arange(len(I))
        N = d.ngt[I].sum()
        rv = rvprobe(I)
        fixed = Fam([np.full(len(I), m) for m in menu], d, I, d.tp, d.fp, d.tps)
        seqs = d.seq[I]
        us = np.unique(seqs)
        by = {s: np.where(seqs == s)[0] for s in us}
        aps = coco_eval([d.images[i] for i in I], gts,
                        {cn: [np.column_stack([np.full(len(v), k), v[:, 0], v[:, 1], v[:, 2] - v[:, 0], v[:, 3] - v[:, 1],
                                               v[:, 4], v[:, 5] + 1]) for k, v in
                              enumerate(dets_for_ap[cn][d.images[i]] for i in I)] for cn in set(sel.values())},
                        {d.images[i] for i in I})
        res["pools"][p] = {}
        for bi, B in enumerate(BUDGETS):
            rvB = rv.at(B, pos, N)[0]
            fxB = fixed.fixed_at(B, pos, N)[0]
            row = dict(RV_probe=float(rvB * PT) if np.isfinite(rvB) else None,
                       fixed_best=float(fxB * PT) if np.isfinite(fxB) else None, fam={})
            members = {f: sel[(f, B)] for f in fams if (f, B) in sel}
            rng = np.random.default_rng([559, bi, pi])
            reps = [np.concatenate([by[s] for s in rng.choice(us, len(us), replace=True)]) for _ in range(a.boot)]

            def rv_at(pp):
                v = rv.at(B, pp, d.ngt[I[pp]].sum())[0]
                return v if np.isfinite(v) else 0.0
            rv_reps = np.array([rv_at(pp) for pp in reps])
            for f, cn in members.items():
                r = R[cn]
                recf = lambda pp: r["tp"][I[pp]].sum() / d.ngt[I[pp]].sum()
                dl = (recf(pos) - (rvB if np.isfinite(rvB) else 0.0)) * PT
                boots = np.array([(recf(pp) - rv_reps[k]) * PT for k, pp in enumerate(reps)])
                row["fam"][f] = dict(config=cn, recall=float(recf(pos) * PT), delta=float(dl),
                                     ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                     mbit=float(np.mean(r["bits"][I]) / 1e6),
                                     recall_small=float(r["tps"][I].sum() / d.nsmall[I].sum() * PT),
                                     fp_frame=float(np.mean(r["fp"][I])), on_ms=float(np.mean(r["on_ms"][I])),
                                     bs_ms=float(np.mean(r["bs_ms"][I])), ap50=float(aps[cn]["AP50"] * PT) if cn in aps else None)
            if members:
                best_non = lambda pp: max(R[cn]["tp"][I[pp]].sum() / d.ngt[I[pp]].sum() for cn in members.values())
                dp = ((rvB if np.isfinite(rvB) else 0.0) - best_non(pos)) * PT
                bp = np.array([(rv_reps[k] - best_non(pp)) * PT for k, pp in enumerate(reps)])
                row["img_over_best_nonimg"] = dict(delta=float(dp), ci=[float(np.percentile(bp, 2.5)),
                                                                         float(np.percentile(bp, 97.5))])
            # RV_probe small recall / FP at B (expected over its mixture)
            v, ws = rv.at(B, pos, N)
            if ws:
                row["RV_probe_small"] = float(sum(w * rv.S[k, pos].sum() for k, w in ws) / d.nsmall[I].sum() * PT)
                row["RV_probe_fp"] = float(sum(w * rv.F[k, pos].mean() for k, w in ws))
                dom = max(ws, key=lambda kw: kw[1])[0]       # AP50 of the dominant (deterministic) RV_probe member
                row["RV_probe_ap50_dominant"] = float(ap50(d, I, rv.ch[dom], p) * PT)
            res["pools"][p][f"{B / 1e6:g}"] = row

    # verdict (0.6)
    feas = {u: 1e3 / on_ms[u] >= 5.0 for u in (640, 960)}
    c1, c1_one, c1_pos, c2 = [], [], [], []
    for B in REAL:
        k = f"{B / 1e6:g}"
        for f in fams:
            rows = [res["pools"][p][k]["fam"].get(f) for p in ("test", "testdev")]
            if all(rows):
                ok = [r["delta"] >= 2.0 and r["ci"][0] > 0 for r in rows]
                pos_ok = [r["ci"][0] > 0 for r in rows]
                if all(ok):
                    c1.append((k, f, rows[0]["config"]))
                elif any(ok):
                    c1_one.append((k, f))
                if all(pos_ok):
                    c1_pos.append((k, f))
        rr = [res["pools"][p][k].get("img_over_best_nonimg") for p in ("test", "testdev")]
        if all(rr) and all(r["delta"] >= 2.0 and r["ci"][0] > 0 for r in rr):
            c2.append(k)
    c1_ok = [x for x in c1 if feas[int(x[2].split("|u")[1].split("|")[0])]]
    c2_ok = [k for k in c2 if any(k != x[0] for x in c1)]
    if c1_ok and c2_ok:
        verdict = "PASS"
    elif c1 or c1_one or c1_pos:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    res["verdict"] = dict(verdict=verdict, c1=c1, c1_feasible=c1_ok, c1_one_pool=c1_one, c1_positive=c1_pos,
                          c2=c2, feasible_5fps=feas)

    # failure cases: selected config vs RV_probe dominant member, small TP, at each realistic budget (test-dev)
    fails = {}
    I = pools["testdev"]
    pos = np.arange(len(I))
    rv = rvprobe(I)
    for B in REAL:
        v, ws = rv.at(B, pos, d.ngt[I].sum())
        if not ws:
            continue
        dom = max(ws, key=lambda kw: kw[1])[0]
        for f in fams:
            if (f, B) not in sel:
                continue
            cn = sel[(f, B)]
            loss = R[cn]["tps"][I] - rv.S[dom]
            w = np.argsort(loss)[:6]
            fails[f"{f}@{B / 1e6:g}"] = [dict(image=d.images[I[j]], seq=str(d.seq[I[j]]), small_loss=float(loss[j]),
                                              n_gt=float(d.ngt[I[j]]), n_small=float(d.nsmall[I[j]]),
                                              res=f"{man[d.images[I[j]]]['w']}x{man[d.images[I[j]]]['h']}") for j in w]
    res["failures_testdev"] = fails

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    out = C.A5.parent / "r1a"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(clean(res), indent=1))
    report(res, d)


def report(res, d):
    print("\n## Kết quả R1a (test / test-dev; điểm recall)\n")
    print("| pool | B (Mbit/khung) | RV_probe (IMG) | cố định tốt nhất | họ | cấu hình (chọn trên val) | recall | **Δ_B** [CI] |"
          " Mbit thực | recall nhỏ (RV_probe) | FP/khung (RV_probe) | AP50 (RV_probe trội) | ms UAV | ms BS |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p, rows in res["pools"].items():
        for B, row in rows.items():
            for f, r in row["fam"].items():
                fmt = lambda x: f"{x:.2f}" if x is not None else "—"
                print(f"| {p} | {B} | {fmt(row['RV_probe'])} | {fmt(row['fixed_best'])} | {f} | {r['config']} | "
                      f"{r['recall']:.2f} | **{r['delta']:+.2f}** [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['mbit']:.4f} | "
                      f"{r['recall_small']:.2f} ({fmt(row.get('RV_probe_small'))}) | {r['fp_frame']:.1f} "
                      f"({fmt(row.get('RV_probe_fp'))}) | {fmt(r['ap50'])} ({fmt(row.get('RV_probe_ap50_dominant'))}) | "
                      f"{r['on_ms']:.0f} | {r['bs_ms']:.0f} |")
            if "img_over_best_nonimg" in row:
                x = row["img_over_best_nonimg"]
                print(f"| {p} | {B} | | | IMG − tốt nhất không-ảnh | | | **{x['delta']:+.2f}** [{x['ci'][0]:+.2f}, "
                      f"{x['ci'][1]:+.2f}] | | | | | | |")
    print("\n### Tất cả cấu hình (test-dev): Mbit, recall, recall nhỏ, FP/khung, ms UAV, ms BS, số ô\n")
    print("| cấu hình | Mbit | recall | recall nhỏ | FP/khung | ms UAV | ms BS | ô |")
    print("|---|---|---|---|---|---|---|---|")
    for cn, v in res["configs"].items():
        x = v["testdev"]
        print(f"| {cn} | {x['mbit']:.4f} | {x['recall']:.2f} | {x['recall_small']:.2f} | {x['fp_frame']:.1f} | "
              f"{x['on_ms']:.0f} | {x['bs_ms']:.0f} | {x['tiles']:.1f} |")
    print(f"\nTiming: {json.dumps(res['timing'])}")
    print(f"\n**Kết luận R1a: {res['verdict']['verdict']}** — {json.dumps(res['verdict'])}")


if __name__ == "__main__":
    main()
