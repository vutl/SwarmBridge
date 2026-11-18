"""Review #46 step C summary and the PRE-REGISTERED C1 gate (reports/phases/2026-11-17_paperA-scheduling/swarmbridge-atc-protocol.md).
Run in the WSL env (workload AP50 uses the cached per-image matching of a6l1_ap on the stored test-dev detections).
Usage: python atc_summary.py [--boot 10000]
"""
import argparse
import collections
import gzip
import json

import numpy as np

import a5_common as C
import atc_run as R
from ax2_summary import ci, fmt

PT = 100.0


def two_way(tp, gt, other=None, boot=10000, seed=0):
    """Two-way (episode x video) bootstrap of a pooled ratio sum(tp) / sum(gt) over [E, S] cell matrices, or of the
    difference of two such ratios on the same draws. Returns (mean of the point estimate, lo, hi) like ax2_summary.ci."""
    rng = np.random.default_rng(seed)
    E, S = tp.shape
    we = rng.multinomial(E, np.full(E, 1 / E), size=boot).astype(float)
    ws = rng.multinomial(S, np.full(S, 1 / S), size=boot).astype(float)

    def ratio(t, g):
        return np.einsum("de,es,ds->d", we, t, ws) / np.einsum("de,es,ds->d", we, g, ws)
    r = ratio(tp, gt)
    pt = tp.sum() / gt.sum()
    if other is not None:
        r = r - ratio(*other)
        pt = pt - other[0].sum() / other[1].sum()
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    D = R.OUT
    dev = json.loads((D / "dev.json").read_text())
    te = json.loads((D / "test.json").read_text())
    dl = json.load(gzip.open(D / "test.deliv.json.gz", "rt"))
    sc = R.Scorer()
    T, rows = sc.t, sc.rows
    man = C.manifest()["images"]
    A, seeds, S = te["arms"], te["seeds"], te["sched"]
    P = te["P_star"]
    res = {r["combo"].split("|")[1]: r for r in te["results"]}
    seqs = sorted({man[mi]["seq"] for ep in dl[f"{S}|{A['q3']}"] for mi, *_ in ep})
    sidx = {s: i for i, s in enumerate(seqs)}

    def deliv(arm):
        return dl[f"{S}|{A[arm]}"]

    def cells(arm, qmin=False):
        eps = deliv(arm)
        tp, gt = np.zeros((len(eps), len(seqs))), np.zeros((len(eps), len(seqs)))
        for e, ep in enumerate(eps):
            for mi, mode, ok, ot in ep:
                r, j = rows[mi], sidx[man[mi]["seq"]]
                gt[e, j] += T["ngt"][r]
                if mode >= 0 and (ok if qmin else ot):
                    tp[e, j] += T["gt"][r, mode]
        return tp, gt

    def met(arm, k):
        r = res[A[arm]]
        assert r["seeds"] == seeds
        return np.array([x[k] for x in r["rows"]], float)

    # ---- dev recap ----
    print(f"== dev (frozen before test): C* = {dev['C_star']}, Q-C* = {dev['QC_star']}, P* = {P}")
    for s, v in dev["val_offline"].items():
        print(f"   {s}: val spearman {v['spearman']:.3f}, rmse log1p {v['rmse_log1p']:.3f} (constant "
              f"{v['rmse_log1p_const']:.3f})")
    print("   val recall: " + ", ".join(f"{k} {PT * v:.2f}" for k, v in dev["val_recall"].items()))
    print("   " + "; ".join(dev["log"]))

    # ---- AP50 (stored detections) ----
    try:
        import a6l1_ap as AP
        b = AP.build("testdev")
        aidx = {n: i for i, n in enumerate(b["images"])}

        def ap50(eps):
            return AP.ap50([("testdev", aidx[man[mi]["name"]], C.MODES[mode] if (mode >= 0 and ot) else None)
                            for ep in eps for mi, mode, ok, ot in ep], {"testdev": b})
    except Exception as ex:                                              # pragma: no cover
        print("AP50 unavailable:", ex)
        ap50 = None

    # ---- arms ----
    out = dict(dev=dict(C_star=dev["C_star"], QC_star=dev["QC_star"], P_star=P, log=dev["log"]), arms={},
               gate={}, c0={}, secondary={})
    print(f"== test (test-dev pool, fresh seeds {seeds[0]}..{seeds[-1]}, {len(seqs)} videos)")
    print(f"   {'arm':8s} {'recall':>7s} {'macro':>6s} {'FP/fr':>6s} {'AP50':>6s} {'Qmin-r':>7s} {'ontime':>7s} "
          f"{'drop':>6s} {'Mbit':>6s} {'E kJ':>6s} {'p99ms':>6s}  combo | modes")
    per = {}
    for arm in A:
        eps = deliv(arm)
        per[arm] = dict(recall=sc.per_ep(eps), macro=sc.per_ep(eps, "macro"), fp=sc.per_ep(eps, "fp"),
                        recall_qmin=sc.per_ep(eps, "recall_qmin"),
                        ontime=np.array([np.mean([x[3] for x in ep]) for ep in eps]))
        tp, gt = cells(arm)
        tq, gq = cells(arm, True)
        cnt = collections.Counter(C.MODES[x[1]] if x[1] >= 0 else "none" for ep in eps for x in ep)
        n = sum(cnt.values())
        t = dict(recall=float(tp.sum() / gt.sum()), macro=float(per[arm]["macro"].mean()),
                 fp=float(per[arm]["fp"].mean()), recall_qmin=float(tq.sum() / gq.sum()),
                 ontime=float(per[arm]["ontime"].mean()), drop=float(met(arm, "drop_rate").mean()),
                 mbit=float(met(arm, "delivered_mbit").mean()), energy_kj=float(met(arm, "energy_kj").mean()),
                 int_p99=float(met(arm, "ms_int_max_p99").mean()), int_p99_max=float(met(arm, "ms_int_max_p99").max()),
                 ap50=float(ap50(eps)) if ap50 else None, combo=A[arm],
                 modes={k: v / n for k, v in cnt.most_common(4)})
        out["arms"][arm] = t
        print(f"   {arm:8s} {PT * t['recall']:7.2f} {PT * t['macro']:6.2f} {t['fp']:6.2f} "
              f"{(PT * t['ap50']) if ap50 else float('nan'):6.2f} {PT * t['recall_qmin']:7.2f} {t['ontime']:7.3f} "
              f"{t['drop']:6.3f} {t['mbit']:6.1f} {t['energy_kj']:6.1f} {t['int_p99']:6.1f}  {A[arm]} | "
              + ", ".join(f"{k} {v:.2f}" for k, v in t["modes"].items()))

    def d2(x, y, qmin=False):
        return [PT * v for v in two_way(*cells(x, qmin), other=cells(y, qmin), boot=a.boot)]

    def dseed(x, y, k="recall"):
        return [PT * v for v in ci(per[x][k] - per[y][k])]

    # ---- C1 gate ----
    ps, cs = f"RV-{P}", "C*"
    dR, dR_s = d2(ps, cs), dseed(ps, cs)
    H = d2("RV-count", cs)
    arm_p, arm_c = out["arms"][ps], out["arms"][cs]
    conds = {
        "1  dR >= +1.0 point": dR[0] >= 1.0,
        "2  two-way 95% CI lower bound > 0": dR[1] > 0,
        "3  dR >= 0.5 * H_cnt": dR[0] >= 0.5 * H[0],
        "4  macro recall not lower by > 1.0 pt": PT * (arm_p["macro"] - arm_c["macro"]) >= -1.0,
        "5  FP per frame not higher by > 10 %": arm_p["fp"] <= 1.10 * arm_c["fp"],
        "6  on-time rate not lower by > 1 pt": PT * (arm_p["ontime"] - arm_c["ontime"]) >= -1.0,
        "7  energy not higher by > 2 %": arm_p["energy_kj"] <= 1.02 * arm_c["energy_kj"],
        "8  integrated decision latency p99 <= 50 ms": arm_p["int_p99"] <= 50.0,
    }
    conds = {k: bool(v) for k, v in conds.items()}
    dec = "GO" if all(conds.values()) else "STOP the task-value direction"
    print(f"== C1 GATE: P* = {P}: RV-{P} vs C* ({dev['C_star']})")
    print(f"   dR  two-way {fmt(dR)}   seed-paired {fmt(dR_s)}")
    print(f"   H_cnt (RV-count - C*) two-way {fmt(H)}; recovery {dR[0] / H[0] if H[0] else float('nan'):.2f}")
    print(f"   macro {PT * (arm_p['macro'] - arm_c['macro']):+.2f} pts; FP/frame {arm_p['fp']:.2f} vs {arm_c['fp']:.2f}; "
          f"on-time {PT * (arm_p['ontime'] - arm_c['ontime']):+.2f} pts; energy "
          f"{PT * (arm_p['energy_kj'] / arm_c['energy_kj'] - 1):+.2f} %; p99 {arm_p['int_p99']:.1f} ms")
    for k, v in conds.items():
        print(f"      {'PASS' if v else 'FAIL'}  {k}")
    print(f"   DECISION: {dec}")
    out["gate"] = dict(dR=dR, dR_seed=dR_s, H_cnt=H, recovery=dR[0] / H[0] if H[0] else None, conditions=conds,
                       decision=dec)

    # ---- secondary contrasts ----
    print("== secondary contrasts (recall points; two-way CI | seed CI)")
    for x, y in (("RV-pA", "C*"), ("RV-pB", "C*"), ("RV-pC", "C*"), (ps, "RV-pA"), ("RV-mean", "C*"),
                 ("RV-count", "RV-mean"), ("RV-gt", "C*"), ("RV-gt", "RV-count"), ("q3+MPC", "q3")):
        if x == y:
            continue
        c2, cs_ = d2(x, y), dseed(x, y)
        out["secondary"][f"{x} - {y}"] = dict(two_way=c2, seed=cs_,
                                              macro=PT * (out["arms"][x]["macro"] - out["arms"][y]["macro"]))
        print(f"   {x:9s} - {y:8s} {fmt(c2)} | {fmt(cs_)}   macro {out['secondary'][f'{x} - {y}']['macro']:+.2f}")

    # ---- C0 ----
    print(f"== C0 (Qmin-constrained; Q-C* = {dev['QC_star']}), Qmin-compliant recall points")
    for s in ("count", P, "gt", "mean"):
        c2 = d2(f"RVQ-{s}", "Q-C*", True)
        det = d2(f"RVQ-{s}", "C*")
        out["c0"][s] = dict(H_Q=c2, detector_only_vs_Cstar=det)
        print(f"   RVQ-{s:5s} - Q-C*  H_Q {fmt(c2)}   (its detector-only recall vs C*: {fmt(det)})")
    hq = out["c0"]["count"]["H_Q"]
    close = hq[1] > 0 and hq[0] >= 0.5 * H[0]
    out["c0"]["reading"] = ("close: count knowledge also helps under the quality floor" if close else
                            "materially different: the detector-only headroom depends on dropping the floor")
    print(f"   reading (fixed in the protocol): {out['c0']['reading']}")

    # ---- offline count accuracy on test-dev (after the test) ----
    from scipy.stats import spearmanr
    pr = json.loads((D / "pred.json").read_text())["n_hat"]
    names = [e["name"] for e in man if e["split"] == "testdev"]
    by = {}
    for line in open(C.REPO / "runs" / "uav_semcom" / "a6l1" / "fidelity" / "per_image_testdev.jsonl"):
        r = json.loads(line)
        if r["mode"] == "original":
            by[r["image"]] = r["n_gt"]
    y = np.log1p([by[n] for n in names])
    out["secondary"]["testdev_offline"] = {}
    for s in ("pA", "pB", "pC"):
        p = np.log1p([pr[s][n] for n in names])
        out["secondary"]["testdev_offline"][s] = dict(spearman=float(spearmanr(p, y)[0]),
                                                      rmse_log1p=float(np.sqrt(np.mean((p - y) ** 2))))
        print(f"   test-dev offline {s}: spearman {out['secondary']['testdev_offline'][s]['spearman']:.3f}, rmse log1p "
              f"{out['secondary']['testdev_offline'][s]['rmse_log1p']:.3f}")

    # ---- concentration of RV-P* - C* ----
    tpp, gtt = cells(ps)
    tpc, _ = cells(cs)
    g = (tpp - tpc).sum(0)
    order = np.argsort(-np.abs(g))
    print("== concentration of RV-P* - C* (TP summed over episodes)")
    print("   per video (largest |gain|): " + ", ".join(f"{seqs[i]} {g[i]:+.0f}" for i in order[:6])
          + f"; total {g.sum():+.0f}, videos with gain > 0: {(g > 0).sum()}/{len(g)}")

    def tp_img(arm):
        o = []
        for ep in deliv(arm):
            dd = collections.defaultdict(float)
            for mi, mode, ok, ot in ep:
                dd[mi] += T["gt"][rows[mi], mode] if (mode >= 0 and ot) else 0.0
            o.append(dd)
        return o
    gains, ngt = [], []
    for x, y_ in zip(tp_img(ps), tp_img(cs)):
        for mi in set(x) | set(y_):
            gains.append(x.get(mi, 0.0) - y_.get(mi, 0.0))
            ngt.append(T["ngt"][rows[mi]])
    gains, ngt = np.array(gains), np.array(ngt)
    q = np.quantile(ngt, [1 / 3, 2 / 3])
    for lab, msk in (("few objects", ngt <= q[0]), ("mid", (ngt > q[0]) & (ngt <= q[1])), ("many", ngt > q[1])):
        print(f"   n_gt tercile {lab:12s} gain {gains[msk].sum():8.0f} TP")
    drop_c = met(cs, "drop_rate")
    qd = np.quantile(drop_c, [1 / 3, 2 / 3])
    dd = per[ps]["recall"] - per[cs]["recall"]
    for lab, msk in (("low congestion", drop_c <= qd[0]), ("mid", (drop_c > qd[0]) & (drop_c <= qd[1])),
                     ("high congestion", drop_c > qd[1])):
        print(f"   C* drop-rate tercile {lab:15s} dR {fmt([PT * v for v in ci(dd[msk])])}")
    out["secondary"]["per_video"] = {seqs[i]: float(g[i]) for i in range(len(g))}
    json.dump(out, open(D / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
