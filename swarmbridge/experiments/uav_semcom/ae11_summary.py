"""A-E11 (review #36) summary: calibration adaptation under distribution shift.

Scenarios (test regime x warm-start / drift), from the variant options of a10_hybrid.py:
  matched                  warm start calibrated in the test regime itself
  cal:X                    warm start calibrated in another environment X (stale / mismatched at deployment)
  drift:T+..               matched warm start; the test episode's PHY changes at env time T (in-episode transition)
Methods: ppo, cold (np-any, no warm start), pre (np-any, frozen warm start = A-E9 / A-E10 controller),
win=W (recency window W per bin), tcold=W (split statistics: warm-start calibrator, online-only error tracker with
window W, PPO until it has data), sd=trust (online monitor + switch to an online-only model after an alarm).
Pairing: per environment seed, controller minus PPO with the same checkpoint and the same drift, averaged over the 3
checkpoints, t-interval over the seeds (as A-E9 / A-E10).
Dev selection rule (stated before the dev results were read): within each family {win=30, win=100} and
{tcold=30, sd=trust}, keep the configuration with the highest worst-scenario mean gain over PPO (maximin over the 9
scenarios).
Run from the repo root: python swarmbridge/experiments/uav_semcom/ae11_summary.py TAG   (TAG = dev | final)"""
import json
import sys

import numpy as np
from scipy import stats

A17 = "runs/uav_semcom/a17"
FAMILIES = {"recency": ["win=30", "win=100"], "split": ["tcold=30", "sd=trust"]}


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def fmt(c):
    return f"{c[0]:+6.2f} [{c[1]:+6.2f},{c[2]:+6.2f}]"


def parse(v):
    p = v.split(":")
    opt = dict(x.split("=", 1) for x in p if "=" in x)
    pos = [x for x in p if "=" not in x]
    kind, ckpt = pos[0], int(pos[1])
    if kind == "ppo":
        meth = "ppo"
    elif pos[9] == "cold":
        meth = "cold"
    else:
        ex = [f"{k}={opt[k]}" for k in ("win", "tcold", "sd") if k in opt]
        meth = ex[0] if ex else "pre"
    if "drift" in opt:
        scen = "drift:" + opt["drift"]
    elif "calreg" in opt:
        scen = "cal:" + opt["calreg"]
    else:
        scen = "matched"
    return meth, ckpt, scen, opt.get("drift")


def main(tag):
    out = {}
    for R in ("phy_mcs_blind", "ref"):
        rows = json.load(open(f"{A17}/ae11_{tag}_{R}.json"))["rows"]
        tab = {}
        for r in rows:
            meth, i, scen, dr = parse(r["variant"])
            if meth in ("ppo", "cold"):          # controls depend only on the drift, not on the warm start
                tab[(meth, i, "drift:" + dr if dr else "nodrift")] = r
            else:
                tab[(meth, i, scen)] = r
        scens = sorted({k[2] for k in tab if k[0] not in ("ppo", "cold")}, key=lambda s: (s != "matched", s))
        meths = sorted({k[0] for k in tab if k[0] not in ("ppo", "cold")}, key=lambda m: (m != "pre", m))
        res = {}
        print(f"===== {R}  ({len(rows)} rows)")
        for sc in scens:
            ctl = sc if sc.startswith("drift:") else "nodrift"
            ppo = np.array([tab[("ppo", i, ctl)]["per_seed_return"] for i in range(3)])
            cold = np.array([tab[("cold", i, ctl)]["per_seed_return"] for i in range(3)])
            t_ref = float(sc.split(":")[1].split("+")[0]) if sc.startswith("drift:") else 0.0
            rs = {"cold": dict(vs_ppo=ci((cold - ppo).mean(0)), ok=float(np.mean([tab[("cold", i, ctl)]["ok_rate"] for i in range(3)])),
                               late=float(np.mean([tab[("cold", i, ctl)]["late_rate"] for i in range(3)])))}
            if sc.startswith("drift:"):
                pp = np.array([tab[("ppo", i, ctl)]["per_seed_return_post"] for i in range(3)])
                cp = np.array([tab[("cold", i, ctl)]["per_seed_return_post"] for i in range(3)])
                rs["cold"]["post_vs_ppo"] = ci((cp - pp).mean(0))
            pre = np.array([tab[("pre", i, sc)]["per_seed_return"] for i in range(3)]) if ("pre", 0, sc) in tab else None
            print(f"--- {sc}   PPO {ppo.mean():.2f}   cold vs PPO {fmt(rs['cold']['vs_ppo'])}")
            for m in meths:
                if (m, 0, sc) not in tab:
                    continue
                rr = [tab[(m, i, sc)] for i in range(3)]
                ret = np.array([r["per_seed_return"] for r in rr])
                row = dict(vs_ppo=ci((ret - ppo).mean(0)), ckpt=(ret - ppo).mean(1).tolist(), vs_cold=ci((ret - cold).mean(0)),
                           ok=float(np.mean([r["ok_rate"] for r in rr])), late=float(np.mean([r["late_rate"] for r in rr])),
                           drop=float(np.mean([r["drop_rate"] for r in rr])), switched=float(np.mean([r["switched"] for r in rr])),
                           gated=float(np.mean([r["gated"] for r in rr])), decisions=float(np.mean([r["decisions"] for r in rr])),
                           ms_p99=float(np.mean([r["ms_p99"] for r in rr])), bias_q=rr[0].get("bias_q"),
                           abort_frac=float(np.mean([r.get("abort_frac", 0.0) for r in rr])))
                if pre is not None and m != "pre":
                    row["vs_pre"] = ci((ret - pre).mean(0))
                if sc.startswith("drift:"):
                    pp = np.array([tab[("ppo", i, ctl)]["per_seed_return_post"] for i in range(3)])
                    mp = np.array([r["per_seed_return_post"] for r in rr])
                    row["post_vs_ppo"] = ci((mp - pp).mean(0))
                if "alarm_t" in rr[0]:
                    al = [a for r in rr for a in r["alarm_t"]]
                    fired = [a for a in al if a is not None]
                    false = [a is not None and (sc == "matched" or a < t_ref) for a in al] if (sc == "matched" or sc.startswith("drift:")) else None
                    det = [a - t_ref for a in fired if a >= t_ref]
                    row.update(alarm_frac=float(np.mean([a is not None for a in al])),
                               false_alarm_frac=float(np.mean(false)) if false is not None else None,
                               time_to_alarm_q=[float(x) for x in np.percentile(det, [10, 50, 90])] if det else None,
                               detected_frac=float(len(det) / len(al)) if sc != "matched" else None)
                rs[m] = row
                extra = "".join(f" {t} {fmt(row[t])}" for t in ("vs_pre", "post_vs_ppo") if t in row)
                al = (f" alarm {row['alarm_frac']:.2f} false {row['false_alarm_frac']} tta {row['time_to_alarm_q']}"
                      if "alarm_frac" in row else "")
                print(f"  {m:9s} vs PPO {fmt(row['vs_ppo'])}{extra} ok {row['ok']:.3f} late {row['late']:.3f} "
                      f"sw {row['switched']:.0f} gated {row['gated']:.0f}/{row['decisions']:.0f} abort {row['abort_frac']:.3f} "
                      f"p99 {row['ms_p99']:.0f}{al}")
            res[sc] = rs
        out[R] = res
    # dev selection (maximin over all scenarios of both regimes)
    sel = {}
    for fam, ms in FAMILIES.items():
        worst = {}
        for m in ms:
            v = [out[R][sc][m]["vs_ppo"][0] for R in out for sc in out[R] if m in out[R][sc]]
            if v:
                worst[m] = (min(v), float(np.mean(v)))
        if worst:
            sel[fam] = dict(worst_and_mean=worst, selected=max(worst, key=lambda m: worst[m][0]))
            print(f"selection {fam}: {worst} -> {sel[fam]['selected']}")
    out["selection"] = sel
    json.dump(out, open(f"{A17}/ae11_{tag}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
