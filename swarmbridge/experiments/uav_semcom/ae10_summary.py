"""A-E10 (review #35 optional stress test) summary: stale / mismatched warm start. The 'pre' calibration episodes run in
the OTHER PHY regime (calreg=...), testing in phy_mcs_blind / ref on the A-E9 fresh seeds 8500-8519. Paired at the
environment-seed level with the A-E9 runs of the same seeds (matched warm start, cold, PPO): per seed, the difference is
averaged over the 3 PPO checkpoints (checkpoint i vs checkpoint i), then a t-interval over the 20 seeds.
Run from the repo root: python swarmbridge/experiments/uav_semcom/ae10_summary.py"""
import json

import numpy as np
from scipy import stats

A9, A10 = "runs/uav_semcom/a15", "runs/uav_semcom/a16"
B = ":8:4:causal:1:2:1:"


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def fmt(c):
    return f"{c[0]:+.2f} [{c[1]:+.2f},{c[2]:+.2f}]"


out = {}
for R, X in (("phy_mcs_blind", "ref"), ("ref", "phy_mcs_blind")):
    r9 = {r["variant"]: r for r in json.load(open(f"{A9}/ae9_{R}.json"))["rows"]}
    r10 = {r["variant"]: r for r in json.load(open(f"{A10}/ae10_{R}.json"))["rows"]}
    fam = {
        "ppo": lambda i: r10[f"ppo:{i}"],
        "cals np-any matched": lambda i: r9[f"ahyb:{i}:cals{B}pre:cand=np:trig=any:stack=1:dl=50:fbd=0:fbl=0"],
        "cals np-any cold": lambda i: r9[f"ahyb:{i}:cals{B}cold:cand=np:trig=any:stack=1:dl=50:fbd=0:fbl=0"],
        f"cals np-any calib-in-{X}": lambda i: r10[f"ahyb:{i}:cals{B}pre:cand=np:trig=any:stack=1:dl=50:calreg={X}"],
    }
    if R == "phy_mcs_blind":
        fam["shannon np-pess matched"] = lambda i: r9[f"ahyb:{i}:shannon{B}pre:cand=np:trig=pess:stack=1:dl=50:fbd=0:fbl=0"]
        fam[f"shannon np-pess calib-in-{X}"] = lambda i: r10[f"ahyb:{i}:shannon{B}pre:cand=np:trig=pess:stack=1:dl=50:calreg={X}"]
    ret = {k: np.array([f(i)["per_seed_return"] for i in range(3)]) for k, f in fam.items()}    # (3, 20)
    ppo9 = np.array([r9[f"ppo:{i}"]["per_seed_return"] for i in range(3)])
    res = dict(ppo_identical_to_ae9=bool(np.allclose(ppo9, ret["ppo"])), ppo_mean=float(ret["ppo"].mean()), rows={})
    print("=====", R, "PPO", round(res["ppo_mean"], 2), "identical to A-E9 PPO:", res["ppo_identical_to_ae9"])
    for k in fam:
        if k == "ppo":
            continue
        d = (ret[k] - ret["ppo"]).mean(0)
        rows = [fam[k](i) for i in range(3)]
        row = dict(vs_ppo=ci(d), ckpt=(ret[k] - ret["ppo"]).mean(1).tolist(),
                   switched=float(np.mean([r["switched"] for r in rows])), gated=float(np.mean([r["gated"] for r in rows])),
                   decisions=float(np.mean([r["decisions"] for r in rows])),
                   ms_p99=float(np.mean([r["ms_p99"] for r in rows])), over50=float(np.mean([r["frac_over_50ms"] for r in rows])),
                   bias_q=rows[0].get("bias_q"))
        if "calib-in" in k:
            base = k.split(" calib-in")[0] + " matched"
            row["vs_matched"] = ci((ret[k] - ret[base]).mean(0))
            if k.startswith("cals"):
                row["vs_cold"] = ci((ret[k] - ret["cals np-any cold"]).mean(0))
        res["rows"][k] = row
        extra = "".join(f"  {t} {fmt(row[t])}" for t in ("vs_matched", "vs_cold") if t in row)
        print(f"{k:34s} vs PPO {fmt(row['vs_ppo'])} ckpt {np.round(row['ckpt'], 2)}{extra}  switched {row['switched']:.0f} "
              f"gated {row['gated']:.0f}/{row['decisions']:.0f} p99 {row['ms_p99']:.1f} >50 {row['over50'] * 100:.2f}% bias_q {row['bias_q']}")
    out[R] = res
json.dump(out, open(f"{A10}/ae10_summary.json", "w"), indent=1)
