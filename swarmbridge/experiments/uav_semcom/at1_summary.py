"""Review #45 (A-T1) summary and the PRE-REGISTERED gate (reports/scheduling/swarmbridge-at1-protocol.md).
Usage: python at1_summary.py runs/uav_semcom/at1
"""
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

import a5_common as C
import a5_eval as AE
import at1_task as AT
from ax2_summary import ci, fmt

CAUSAL = ("q3 (H*)", "F_task", "q3+MPC (I*+MPC, A5 objective)", "RV-mean", "TB-mean", "TMPC-mean")
ORACLE = ("RV-gt", "TB-gt", "TMPC-gt")
FLIPOF = {"RV-gt": "RV-flip", "TB-gt": "TB-flip", "TMPC-gt": "TMPC-flip"}


def pts(c):
    return [100 * x for x in c]


def main(d):
    d = Path(d)
    AE.init(False)
    T = AT.tables(AE.G["table"], AE.G["pools"])
    rows_of = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}
    man = C.manifest()["images"]
    tu = json.loads((d / "tune.json").read_text())
    te = json.loads((d / "test.json").read_text())
    dl = json.load(gzip.open(d / "test.deliv.json.gz", "rt"))
    A, seeds = te["arms"], te["seeds"]
    res = {r["combo"].split("|")[1]: r for r in te["results"]}
    S = te["sched"]

    def rec(arm, src="gt", ontime=True):
        return np.array([AT.score([ep], T[src], T["ngt"], rows_of, ontime) for ep in dl[f"{S}|{A[arm]}"]])

    def met(arm, k):
        r = res[A[arm]]
        assert r["seeds"] == seeds
        return np.array([x[k] for x in r["rows"]], float)

    # ---- tune ----
    trec = tu["recall_ontime"]
    tr = {"q3 (H*)": trec["h=fixed_mbt2018-mean_q3"], "F_task": trec[tu["best_fixed"]],
          "q3+MPC (I*+MPC, A5 objective)": trec["h=fixed_mbt2018-mean_q3+mpc"],
          "RV-mean": trec[tu["best_rules"]["rv:mean"]], "TB-mean": trec[tu["best_rules"]["tb:mean"]],
          "TMPC-mean": trec[f"{tu['best_fixed']}+tmpc=mean"], "RV-gt": trec[tu["best_rules"]["rv:gt"]],
          "TB-gt": trec[tu["best_rules"]["tb:gt"]], "TMPC-gt": trec[f"{tu['best_fixed']}+tmpc=gt"]}
    Cs = max(CAUSAL, key=tr.get)
    Os = max(ORACLE, key=tr.get)
    print(f"== tune (train seeds 0..29): F_task = {tu['best_fixed']}; rules {tu['best_rules']}")
    for k, v in tr.items():
        print(f"   {k:32s} {A[k]:40s} train on-time recall {100 * v:6.2f}")
    print(f"   C* = {Cs}   O* = {Os}  (selection rules of the protocol)")

    # ---- test arms ----
    out = dict(C_star=Cs, O_star=Os, arms={}, contrasts={})
    print(f"== test (fresh seeds {seeds[0]}..{seeds[-1]}), scheduler {S}")
    print(f"   {'arm':32s} {'rec_gt':>7s} {'rec_flip':>8s} {'rec_Qmin':>8s} {'ontime':>7s} {'drop':>6s} {'Mbit':>6s} "
          f"{'E kJ':>6s} {'A5 ret':>7s} {'int p99':>7s}")
    ontime = {}
    for arm in A:
        eps = dl[f"{S}|{A[arm]}"]
        ontime[arm] = np.array([np.mean([x[3] for x in ep]) for ep in eps])
        t = dict(rec_gt=float(rec(arm).mean()), rec_flip=float(rec(arm, "flip").mean()),
                 rec_qmin=float(rec(arm, ontime=False).mean()), ontime=float(ontime[arm].mean()),
                 drop=float(met(arm, "drop_rate").mean()), mbit=float(met(arm, "delivered_mbit").mean()),
                 energy_kj=float(met(arm, "energy_kj").mean()), a5_return=float(met(arm, "return").mean()),
                 int_p99=float(met(arm, "ms_int_max_p99").mean()))
        cnt = collections.Counter(C.MODES[x[1]] if x[1] >= 0 else "none" for ep in eps for x in ep)
        n = sum(cnt.values())
        t["modes"] = {k: v / n for k, v in cnt.most_common(5)}
        out["arms"][arm] = t
        print(f"   {arm:32s} {100 * t['rec_gt']:7.2f} {100 * t['rec_flip']:8.2f} {100 * t['rec_qmin']:8.2f} "
              f"{t['ontime']:7.3f} {t['drop']:6.3f} {t['mbit']:6.1f} {t['energy_kj']:6.1f} {t['a5_return']:7.2f} "
              f"{t['int_p99']:7.1f} | " + ", ".join(f"{k} {v:.2f}" for k, v in t["modes"].items()))

    # ---- gate ----
    H = rec(Os) - rec(Cs)
    Hx = 0.5 * (rec(FLIPOF[Os]) + rec(Os, "flip")) - 0.5 * (rec(Cs) + rec(Cs, "flip"))
    d_ontime = ontime[Os] - ontime[Cs]
    e_rel = met(Os, "energy_kj").mean() / met(Cs, "energy_kj").mean() - 1
    out["contrasts"]["H (optimistic)"] = pts(ci(H))
    out["contrasts"]["H_x (cross-checked)"] = pts(ci(Hx))
    print(f"== PRIMARY: O* = {Os} vs C* = {Cs}  (recall points, paired CI over env seeds)")
    print(f"   H   (select and score with the same measurement): {fmt(pts(ci(H)))}")
    print(f"   H_x (select with one, score with the other):      {fmt(pts(ci(Hx)))}")
    print(f"   frame on-time rate diff {fmt(pts(ci(d_ontime)))} pts; energy {100 * e_rel:+.2f} %")
    conds = {"H_x >= +1.0 point": ci(Hx)[0] * 100 >= 1.0, "H_x CI lower bound > 0": ci(Hx)[1] > 0,
             "matched: on-time rate not lower by > 1 pt": ci(d_ontime)[0] * 100 >= -1.0,
             "matched: energy not higher by > 2 %": e_rel <= 0.02}
    for k, v in conds.items():
        print(f"      {'PASS' if v else 'FAIL'}  {k}")
    dec = "GO to step C (causal utility predictor)" if all(conds.values()) else "STOP the task-utility direction"
    conds = {k: bool(v) for k, v in conds.items()}
    out["gate"] = dict(conditions=conds, decision=dec, energy_rel=float(e_rel), ontime_diff=pts(ci(d_ontime)))
    print(f"   DECISION: {dec}")

    # ---- secondary ----
    print("== secondary: same-controller information value (gt / cross-checked), recall points")
    for o, c in (("TMPC-gt", "TMPC-mean"), ("RV-gt", "RV-mean"), ("TB-gt", "TB-mean")):
        g = rec(o) - rec(c)
        x = 0.5 * (rec(FLIPOF[o]) + rec(o, "flip")) - 0.5 * (rec(c) + rec(c, "flip"))
        out["contrasts"][f"{o} - {c}"] = dict(gt=pts(ci(g)), x=pts(ci(x)))
        print(f"   {o:8s} - {c:9s}  gt {fmt(pts(ci(g)))}   cross-checked {fmt(pts(ci(x)))}")
    for arm in A:
        if arm not in (Cs,) and arm not in FLIPOF.values():
            print(f"   {arm:32s} - C*  gt {fmt(pts(ci(rec(arm) - rec(Cs))))}  Qmin-recall "
                  f"{fmt(pts(ci(rec(arm, ontime=False) - rec(Cs, ontime=False))))}")

    # ---- concentration: per (episode, image) TP gain of O* over C* ----
    def tp_by_img(arm):
        out_ = []
        for ep in dl[f"{S}|{A[arm]}"]:
            dd = collections.defaultdict(float)
            for mi, mode, ok, ot in ep:
                dd[mi] += T["gt"][rows_of[mi], mode] if (mode >= 0 and ot) else 0.0
            out_.append(dd)
        return out_
    to, tc = tp_by_img(Os), tp_by_img(Cs)
    gains, ngt, ep_i, seq = [], [], [], []
    for e, (a, b) in enumerate(zip(to, tc)):
        for mi in set(a) | set(b):
            gains.append(a.get(mi, 0.0) - b.get(mi, 0.0))
            ngt.append(T["ngt"][rows_of[mi]])
            ep_i.append(e)
            seq.append(man[mi]["seq"])
    gains, ngt, ep_i = np.array(gains), np.array(ngt), np.array(ep_i)
    tot = gains.sum()
    pos = np.sort(gains[gains > 0])[::-1]
    top = pos[:max(1, int(0.1 * len(gains)))].sum()
    print(f"== concentration (O* - C*, per episode x image instance, gt TP): total {tot:.0f} TP over {len(gains)} "
          f"instances; positive {pos.sum():.0f}, negative {gains[gains < 0].sum():.0f}; top 10 % of instances carry "
          f"{top:.0f} TP = {100 * top / max(pos.sum(), 1e-9):.0f} % of the positive gain")
    q = np.quantile(ngt, [1 / 3, 2 / 3])
    for lab, msk in (("few objects", ngt <= q[0]), ("mid", (ngt > q[0]) & (ngt <= q[1])), ("many objects", ngt > q[1])):
        print(f"   n_gt tercile {lab:12s} gain {gains[msk].sum():8.0f} TP ({100 * gains[msk].sum() / max(abs(tot), 1e-9):+.0f} % of total)")
    drop_c = met(Cs, "drop_rate")
    qd = np.quantile(drop_c, [1 / 3, 2 / 3])
    per_ep = H
    for lab, msk in (("low congestion", drop_c <= qd[0]), ("mid", (drop_c > qd[0]) & (drop_c <= qd[1])),
                     ("high congestion", drop_c > qd[1])):
        print(f"   C* drop-rate tercile {lab:15s} H {fmt(pts(ci(per_ep[msk])))}")
    sq = collections.defaultdict(float)
    for gsum, s in zip(gains, seq):
        sq[s] += gsum
    big = sorted(sq.items(), key=lambda x: -abs(x[1]))[:5]
    print("   per test video (largest |gain|): " + ", ".join(f"{s} {v:+.0f}" for s, v in big))
    out["concentration"] = dict(total=float(tot), top10_share=float(top / max(pos.sum(), 1e-9)),
                                per_video={k: float(v) for k, v in sq.items()})
    json.dump(out, open(d / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
