"""Paper A R1 summary: per PHY, every arm's A5 return (paired CI vs a reference arm), ok rate, detector recall
(Qmin-compliant: on time and PSNR >= Qmin; and on-time where the delivery tuples carry it), energy, decision latency.
PPO arms: the 3 training-seed checkpoints are averaged per environment seed before pairing.
Usage: python paperA_r1_summary.py   (WSL env) -> runs/uav_semcom/paperA_r1/summary.json + printed tables
"""
import collections
import gzip
import json

import numpy as np

import a5_eval as AE
import at1_task as AT
import paperA_r1 as R
from ax2_summary import ci, fmt


def main():
    AE.init(False)
    T = AT.tables(AE.G["table"], AE.G["pools"])
    rows_of = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}
    te = json.loads((R.OUT / "test.json").read_text())
    dl = json.load(gzip.open(R.OUT / "test.deliv.json.gz", "rt"))
    H, S = te["tune"], te["sched"]
    by = collections.defaultdict(list)
    for r in te["results"]:
        by[(r["phy"], r["combo"])].append(r)

    def recall(ep, ontime):
        tp = gt = 0.0
        for x in ep:
            mi, mode, ok = x[0], x[1], x[2]
            hit = ok if not ontime else (x[3] if len(x) > 3 else None)
            if hit is None:
                return float("nan")
            r = rows_of[mi]
            gt += T["ngt"][r]
            tp += T["gt"][r, mode] if (mode >= 0 and hit) else 0.0
        return tp / gt

    def arm(phy, combo):
        rs = by[(phy, combo)]
        assert all(r["seeds"] == te["seeds"] for r in rs)
        per = lambda k: np.mean([[x.get(k, np.nan) for x in r["rows"]] for r in rs], 0)
        eps = [dl[f"{phy}|{combo}|{r['ckpt']}"] for r in rs]
        rq = np.mean([[recall(e, False) for e in ep] for ep in eps], 0)
        ro = np.mean([[recall(e, True) for e in ep] for ep in eps], 0)
        return dict(ret=per("return"), ok=per("ok_rate"), energy=per("energy_kj"), p99=per("ms_int_max_p99"),
                    rec_q=rq, rec_on=ro)

    blocks = {
        "A (A5 T:best slot rule)": [("Random (untuned)", "lit:std_rand"), ("Round robin (untuned)", "lit:std_rr"),
                                             ("Max rate (untuned)", "lit:std_maxrate"), ("Greedy (untuned)", "lit:greedy"),
                                             ("T:best (tuned on Shannon)", "tbest|heur"), ("PPO-S (trained on Shannon)", "tbest|ppo")],
    }
    out = {}
    for phy in ("S", "L"):
        other = "L" if phy == "S" else "S"
        b = [(f"H*_{phy} = {H[phy]}", f"{S}|h={H[phy]}"), (f"H*_{phy} + MPC (calibrated)", f"{S}|h={H[phy]}+mpc"),
             (f"H*_{phy} + MPC (uncalibrated)", f"{S}|h={H[phy]}+mpcnc")]
        if H[other] != H[phy]:
            b.append((f"H*_{other} = {H[other]} (tuned on other PHY)", f"{S}|h={H[other]}"))
        b += [("PPO-S", f"{S}|ppo"), ("PPO-S + MPC (calibrated)", f"{S}|ppo+mpc")]
        if phy == "L":
            b += [("PPO-LA (trained on LA)", f"{S}|ppola"), ("PPO-LA + MPC (calibrated)", f"{S}|ppola+mpc")]
        blk = dict(blocks, **{"B (frozen JA scheduler @ pacedB3)": b})
        print(f"\n===== PHY {phy} ({'Shannon' if phy == 'S' else 'realistic LA'}), seeds {te['seeds'][0]}..{te['seeds'][-1]}")
        out[phy] = {}
        for bname, arms in blk.items():
            ref_lab, ref_c = (arms[4] if bname.startswith("A") else arms[1])
            ref = arm(phy, ref_c)
            print(f"-- block {bname}; paired deltas vs {ref_lab}")
            print(f"   {'arm':44s} {'return':>8s} {'delta [95% CI]':>24s} {'ok':>6s} {'recQ':>6s} {'recOn':>6s} {'E kJ':>6s} {'p99ms':>6s}")
            for lab, c in arms:
                a = arm(phy, c)
                d = [float(x) for x in ci(a["ret"] - ref["ret"])]
                out[phy][f"{bname}|{lab}"] = dict(combo=c, ret=float(a["ret"].mean()), delta_vs_ref=d, ref=ref_lab,
                                                  ok=float(a["ok"].mean()), rec_qmin=float(np.nanmean(a["rec_q"])),
                                                  rec_ontime=float(np.nanmean(a["rec_on"])), energy_kj=float(a["energy"].mean()),
                                                  p99_ms=float(np.nanmean(a["p99"])))
                o = out[phy][f"{bname}|{lab}"]
                print(f"   {lab:44s} {o['ret']:8.2f} {fmt(d):>24s} {o['ok']:6.3f} {100 * o['rec_qmin']:6.2f} "
                      f"{100 * o['rec_ontime']:6.2f} {o['energy_kj']:6.1f} {o['p99_ms']:6.1f}")
    json.dump(dict(seeds=te["seeds"], H_star=H, sched=S, arms=out), open(R.OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
