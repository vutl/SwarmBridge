"""Summary of the literature-baseline evaluation (a) and of PPO trained directly on K5_mcsg (b).

Per regime (K5, ref, K5_mcsg; test pool, fresh seeds 8800..8899): mean return / ok rate / PSNR / energy / mission
time of every policy; PPO groups (joint_best_s*, mode_best_s*, mcsg_joint_best_s*, mcsg_mode_best_s*) are averaged over
their three training seeds per environment seed. Paired contrasts, t-interval over environment seeds, plus the
contrast per training seed: every PPO group vs T:best, vs the best UNTUNED literature-style rule (std_rr, std_maxrate,
std_rand, greedy; best by mean return in that regime, chosen post hoc, which favours the rule) and vs T:fixed.
Usage (repo root): python swarmbridge/experiments/uav_semcom/ax3_summary.py"""
import json
import re

import numpy as np

import a5_common as C
from ax2_summary import ci, fmt

D = C.REPO / "runs" / "uav_semcom" / "ax3"
FILES = {"K5": "std_K5.json", "ref": "std_ref.json", "K5_mcsg": "mcsg_rl.json"}
UNTUNED = ("H:std_rand", "H:std_rr", "H:std_maxrate", "H:greedy")


def main():
    out = {}
    for R, f in FILES.items():
        d = json.load(open(D / f))["results"][R]
        groups = {}
        for spec, eps in d.items():
            m = re.match(r"P:(\w+?)_s(\d)=", spec)
            key = m.group(1) if m else spec
            groups.setdefault(key, {})[int(m.group(2)) if m else -1] = eps

        def arr(g, metric="return", i=None):
            ck = groups[g]
            sel = [i] if i is not None else sorted(ck)
            return np.mean([[e[metric] for e in ck[j]] for j in sel], 0)

        print(f"===== {R}  seeds {len(next(iter(d.values())))}")
        tab = {}
        for g in groups:
            tab[g] = {k: float(np.nanmean(arr(g, k))) for k in ("return", "ok_rate", "mean_psnr", "energy_kj", "mission_s")}
            tab[g]["per_training_seed"] = [float(np.mean(arr(g, i=i))) for i in sorted(groups[g])]
            print(f"  {g:18s} ret {tab[g]['return']:+8.2f} {np.round(tab[g]['per_training_seed'], 2)} "
                  f"ok {tab[g]['ok_rate']:.3f} psnr {tab[g]['mean_psnr']:.1f} E {tab[g]['energy_kj']:.1f} "
                  f"mission {tab[g]['mission_s']:.0f}s")
        best_untuned = max(UNTUNED, key=lambda g: tab[g]["return"])
        con = {}
        for g in groups:
            if not g.endswith("_best"):
                continue
            for ref in ("T:best", best_untuned, "T:fixed"):
                c = dict(ret=ci(arr(g) - arr(ref)),
                         per_training_seed=[float(np.mean(arr(g, i=i) - arr(ref))) for i in sorted(groups[g])])
                con[f"{g} - {ref}"] = c
                print(f"  {g:18s} - {ref:13s} {fmt(c['ret'])} per training seed {np.round(c['per_training_seed'], 2)}")
        out[R] = dict(policies=tab, best_untuned=best_untuned, contrasts=con)
    json.dump(out, open(D / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
