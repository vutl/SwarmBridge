"""A-E9 summary. Uncertainty at the ENVIRONMENT-SEED level: for each of the 20 fresh seeds, the controller-minus-PPO
return difference is averaged over the 3 PPO checkpoints (paired: checkpoint i's controller vs checkpoint i's PPO),
then a t-interval over the 20 seed means (19 df). Checkpoint spread reported separately (per-checkpoint means)."""
import json
import re
import sys

import numpy as np
from scipy import stats

D = "runs/uav_semcom/a15"


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def fmt(c):
    return f"{c[0]:+.2f} [{c[1]:+.2f},{c[2]:+.2f}]"


out = {}
for R in ("phy_mcs_blind", "ref"):
    rows = json.load(open(f"{D}/ae9_{R}.json"))["rows"]
    by = {r["variant"]: r for r in rows}
    ppo = {i: np.array(by[f"ppo:{i}"]["per_seed_return"]) for i in range(3)}
    fam = {}
    for v in by:
        if v.startswith("ppo"):
            continue
        f = re.sub(r"^ahyb:\d", "ahyb:I", v)
        fam.setdefault(f, []).append(v)
    print("=====", R, "PPO", np.mean([ppo[i].mean() for i in range(3)]).round(2))
    res = {}
    for f, vs in fam.items():
        vs = sorted(vs)
        diff = np.array([np.array(by[v]["per_seed_return"]) - ppo[int(v.split(":")[1])] for v in vs])   # (3, 20)
        seed_mean = diff.mean(0)
        g = lambda k: float(np.mean([by[v].get(k, np.nan) if by[v].get(k) is not None else np.nan for v in vs]))
        ab = {}
        n = sum(by[v]["decisions"] for v in vs)
        for v in vs:
            for k, c in by[v].get("aborts", {}).items():
                ab[k] = ab.get(k, 0) + c
        fb = by[vs[0]].get("feedback") or {}
        lost = (sum((by[v].get("feedback") or {}).get("lost", 0) for v in vs) /
                max(sum((by[v].get("feedback") or {}).get("sent", 0) for v in vs), 1))
        m = re.search(r"fbd=(\d+):fbl=([\d.]+)", f)
        key = f.replace(":8:4:causal:1:2:1", "").replace("ahyb:I:", "")
        res[key] = dict(seed_ci=ci(seed_mean), seed_means=seed_mean.tolist(), ckpt_means=diff.mean(1).tolist(),
                        ms=g("ms_mean"), p99=g("ms_p99"), max=max(by[v]["ms_p99"] * 0 + by[v].get("miss_ms_max", np.nan) for v in vs),
                        over50=g("frac_over_50ms"), miss=g("miss_frac") if "miss_frac" in by[vs[0]] else None,
                        full=g("full_frac"), switched=g("switched"), gated=g("gated"),
                        aborts={k: c / n for k, c in ab.items()}, lost_frac=lost,
                        delay=int(m.group(1)) if m else 0, loss=float(m.group(2)) if m else 0.0)
        r_ = res[key]
        print(f"{key:62s} {fmt(r_['seed_ci'])} ckpt {np.round(r_['ckpt_means'], 2)} ms {r_['ms']:5.1f} p99 {r_['p99']:5.1f} "
              f"max {r_['max']:6.1f} >50 {r_['over50'] * 100:5.2f}% full {r_['full']:.2f} gated {r_['gated']:.0f} "
              f"lost {lost:.2f} ab {{{', '.join(f'{k}:{c * 100:.1f}%' for k, c in r_['aborts'].items())}}}")
    # degradation vs the ideal-feedback run of the same controller (paired per seed, averaged over checkpoints)
    print("--- degradation vs fbd=0:fbl=0 (same controller)")
    base = {}
    for f, vs in fam.items():
        ctrl = re.sub(r":fbd=\d+:fbl=[\d.]+", "", f)
        if ":fbd=0:fbl=0" in f and not f.endswith("fbl=0.3") and not f.endswith("fbl=0.6"):
            base[ctrl] = f
    deg = {}
    for f, vs in sorted(fam.items()):
        ctrl = re.sub(r":fbd=\d+:fbl=[\d.]+", "", f)
        if ctrl not in base or base[ctrl] == f:
            continue
        b = np.mean([np.array(by[v]["per_seed_return"]) for v in sorted(fam[base[ctrl]])], 0)
        c = np.mean([np.array(by[v]["per_seed_return"]) for v in sorted(vs)], 0)
        key = f.replace(":8:4:causal:1:2:1", "").replace("ahyb:I:", "")
        deg[key] = ci(c - b)
        print(f"{key:62s} {fmt(deg[key])}")
    # paired comparisons between controllers within a condition
    print("--- paired controller differences (seed level)")
    pair = {}
    conds = sorted({re.search(r"fbd=\d+:fbl=[\d.]+", f).group(0) for f in fam})
    for cnd in conds:
        def get(prefix):
            fs = [f for f in fam if f.startswith(prefix) and f.endswith(cnd)]
            return np.mean([np.array(by[v]["per_seed_return"]) for v in sorted(fam[fs[0]])], 0) if fs else None
        B = ":8:4:causal:1:2:1:pre:"
        cands = {"cals_np-any": get(f"ahyb:I:cals{B}cand=np:trig=any:stack=1:dl=50"),
                 "cals_n2": get(f"ahyb:I:cals{B}cand=n2:stack=1:dl=50"),
                 "cals_all": get(f"ahyb:I:cals{B}cand=all"),
                 "sh_np-pess": get(f"ahyb:I:shannon{B}cand=np:trig=pess:stack=1:dl=50"),
                 "sh_n2": get(f"ahyb:I:shannon{B}cand=n2:stack=1:dl=50"),
                 "sh_all": get(f"ahyb:I:shannon{B}cand=all")}
        for a_, b_ in (("cals_np-any", "cals_n2"), ("cals_np-any", "cals_all"), ("sh_np-pess", "sh_n2"), ("sh_np-pess", "sh_all")):
            if cands[a_] is not None and cands[b_] is not None:
                pair[f"{cnd}:{a_}-{b_}"] = ci(cands[a_] - cands[b_])
                print(f"{cnd:14s} {a_}-{b_:12s} {fmt(pair[f'{cnd}:{a_}-{b_}'])}")
    out[R] = dict(controllers=res, degradation=deg, paired=pair)
json.dump(out, open(f"{D}/ae9_summary.json", "w"), indent=1)
