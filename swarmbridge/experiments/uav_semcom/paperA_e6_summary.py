"""Paper A E6 summary (pre-registered addendum in reports/phases/2026-11-18_paperA-writeup/paperA-e4-e5-prereg.md):
total bandwidth 0.5 / 1 / 2 MHz at K = 5, M = 2 (the 1 MHz point is N5x2 of E4, same seeds 13000..13099).
H1 cal - H*, H2 cal - RobustMPC-style, H3 cal - nominal (paired 95% t-CI); rule: H1 and H2 at all three bandwidths.
Output: runs/uav_semcom/paperA_r2/summary_e6.json + printed table."""
import gzip
import json

import numpy as np

import a5_eval as AE
import at1_task as AT
import paperA_r2 as R
from ax2_summary import ci


def main():
    AE.init(False)
    T = AT.tables(AE.G["table"], AE.G["pools"])
    rows_of = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}
    src = {}
    for name, key in (("e6", "H_bw"), ("e4", "H_net")):
        te = json.loads((R.OUT / f"{name}.json").read_text())
        dl = json.load(gzip.open(R.OUT / f"{name}.deliv.json.gz", "rt"))
        for r in te["results"]:
            src[(r["phy"], r["combo"])] = (r, dl[f"{r['phy']}|{r['combo']}|{r['ckpt']}"], te["seeds"])
        if name == "e6":
            S, best = te["sched"], dict(te[key])
        else:
            best["N5x2"] = te[key]["N5x2"]
    confs = [("B0.5", 0.5), ("N5x2", 1.0), ("B2", 2.0)]

    def get(n, suf):
        r, d, seeds = src[(n, f"{S}|h={best[n]}{suf}")]
        return r, d, seeds

    def ret(n, suf):
        return np.array([x["return"] for x in get(n, suf)[0]["rows"]])

    def rec(n, suf):
        v = []
        for ep in get(n, suf)[1]:
            tp = gt = 0.0
            for mi, mode, ok, ot in ep:
                rr = rows_of[mi]
                gt += T["ngt"][rr]
                if mode >= 0 and ok:
                    tp += T["gt"][rr, mode]
            v.append(100 * tp / gt)
        return np.array(v)

    def f(v):
        m, lo, hi = v
        return f"{m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]{'*' if lo > 0 or hi < 0 else ' '}"

    out, passes = {}, dict(H1=0, H2=0, H3=0)
    for n, mhz in confs:
        assert get(n, "")[2] == list(range(13000, 13100))
        o = {"MHz": mhz, "H": best[n],
             "H1_cal_minus_H": [float(x) for x in ci(ret(n, "+mpc") - ret(n, ""))],
             "H2_cal_minus_rh": [float(x) for x in ci(ret(n, "+mpc") - ret(n, "+mpcrh"))],
             "H3_cal_minus_nom": [float(x) for x in ci(ret(n, "+mpc") - ret(n, "+mpcnc"))],
             "pool_minus_cal": [float(x) for x in ci(ret(n, "+mpcpl") - ret(n, "+mpc"))],
             "recall_cal_minus_H": [float(x) for x in ci(rec(n, "+mpc") - rec(n, ""))]}
        o["level"] = {a or "H": dict(ret=float(ret(n, a).mean()), recall=float(rec(n, a).mean()),
                                     **{k: float(np.mean([x[k] for x in get(n, a)[0]["rows"]])) for k in
                                        ("drop_rate", "q_viol_rate", "ok_rate", "ms_int_max_p99", "int_over50")})
                      for a in ("", "+mpc", "+mpcnc", "+mpcrh", "+mpcpl")}
        for h, k in (("H1", "H1_cal_minus_H"), ("H2", "H2_cal_minus_rh"), ("H3", "H3_cal_minus_nom")):
            passes[h] += o[k][1] > 0
        out[n] = o
        print(f"== {mhz} MHz ({n}, H*={best[n]})")
        print(f"   H1 cal - H*      {f(o['H1_cal_minus_H'])}   recall {f(o['recall_cal_minus_H'])}")
        print(f"   H2 cal - RobMPC  {f(o['H2_cal_minus_rh'])}")
        print(f"   H3 cal - nominal {f(o['H3_cal_minus_nom'])}")
        print(f"   pooled - cal     {f(o['pool_minus_cal'])}")
        for a, L in o["level"].items():
            print(f"     {a:8s} ret {L['ret']:8.1f} recall {L['recall']:5.1f} ok {L['ok_rate']:.3f} drop {L['drop_rate']:.3f} "
                  f"qviol {L['q_viol_rate']:.3f} int p99 {L['ms_int_max_p99']:5.1f} ms >50ms {L['int_over50']:.3f}")
    out["pass"] = dict(passes, of=3)
    print(f"PASS counts (CI > 0): H1 {passes['H1']}/3  H2 {passes['H2']}/3  H3 {passes['H3']}/3  "
          f"(rule: H1 and H2 at all three bandwidths)")
    json.dump(out, open(R.OUT / "summary_e6.json", "w"), indent=1)


if __name__ == "__main__":
    main()
