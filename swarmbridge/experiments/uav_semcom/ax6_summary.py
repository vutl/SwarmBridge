"""Review #41 (A-X6) summary.

invariance: A-X5 policies, old (non-atomic) vs atomic accounting, paired per environment seed: mean return of each,
            paired difference (CI), and the A-X5 headline contrasts (vs T:best@K5_la_rand) recomputed under both.
eval      : per variant, every policy's metrics; decomposition of the scheduling opportunity (paired CIs):
              selection gain = JA-NR - NR           (which UAVs transmit, no reuse)
              reuse gain     = JA-R  - JA-NR        (co-channel reuse on top of the best no-reuse choice)
            for causal (statistical-model) and oracle (realised fading, UPPER BOUND) information; plus RU / RUG - NR,
            and how often the reuse-allowed scheduler chose reuse in contended slots (> M UAVs with traffic).
Usage: python ax6_summary.py runs/uav_semcom/ax6
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from ax2_summary import ci, fmt

KEYS = ("return", "ok_rate", "drop_rate", "late_rate", "bler", "shared_bler", "unpred_shared_frac",
        "cochannel_air_frac", "goodput_kbps", "energy_kj", "jain", "min_uav_ok", "pad_air_s", "bits_rolled_back_frac",
        "tasks_reopened", "ja_reuse_frac", "ja_pred_gain")


def invariance(path):
    d = json.load(gzip.open(path, "rt") if str(path).endswith(".gz") else open(path))
    print(f"reproduction of stored A-X5 rows (la_atomic False): max |return diff| = {d['repro_max_abs_return_diff']:.3g}")
    by = {}
    for key, v in d["results"].items():
        env, spec, ck, acc = key.split("|")
        by.setdefault((env, spec, acc), {})[ck] = dict(zip(v["seeds"], v["rows"]))

    def arr(env, spec, acc, k="return"):
        cks = by[(env, spec, acc)]
        seeds = sorted(next(iter(cks.values())))
        return np.mean([[cks[c][s][k] for s in seeds] for c in sorted(cks)], 0)
    out = {}
    for env in sorted({e for e, _, _ in by}):
        print(f"===== {env}: old -> atomic (paired atomic - old)")
        o = {}
        for spec in sorted({s for e, s, _ in by if e == env}):
            a, b = arr(env, spec, "old"), arr(env, spec, "atomic")
            r = dict(old=float(a.mean()), atomic=float(b.mean()), diff=ci(b - a),
                     ok_old=float(arr(env, spec, "old", "ok_rate").mean()),
                     ok_atomic=float(arr(env, spec, "atomic", "ok_rate").mean()),
                     goodput_old=float(arr(env, spec, "old", "goodput_kbps").mean()),
                     goodput_atomic=float(arr(env, spec, "atomic", "goodput_kbps").mean()),
                     late_old=float(arr(env, spec, "old", "late_rate").mean()),
                     late_atomic=float(arr(env, spec, "atomic", "late_rate").mean()),
                     reopened=float(arr(env, spec, "atomic", "tasks_reopened").mean()),
                     rolled_back=float(arr(env, spec, "atomic", "bits_rolled_back_frac").mean()))
            ref = "T:best@K5_la_rand"
            if spec != ref:
                r["vs_ref_old"] = ci(a - arr(env, ref, "old"))
                r["vs_ref_atomic"] = ci(b - arr(env, ref, "atomic"))
            o[spec] = r
            print(f"  {spec:26s} {r['old']:8.2f} -> {r['atomic']:8.2f}  d {fmt(r['diff'])}  ok {r['ok_old']:.3f}->"
                  f"{r['ok_atomic']:.3f} goodput {r['goodput_old']:6.1f}->{r['goodput_atomic']:6.1f} late "
                  f"{r['late_old']:.3f}->{r['late_atomic']:.3f} reopened {r['reopened']:.1f} rolledback "
                  f"{r['rolled_back']:.4f}" + (f"  vs ref old {fmt(r['vs_ref_old'])} atomic {fmt(r['vs_ref_atomic'])}"
                                              if spec != ref else ""))
        out[env] = o
    return out


def evaluation(path):
    d = json.load(gzip.open(path, "rt"))
    by = {}
    for r in d["results"]:
        by.setdefault(r["variant"], {})[r["policy"]] = r
    out = {}
    for v in sorted(by):
        P = by[v]
        print(f"===== {v}")
        tab = {}
        for p in ("NR", "RU", "RUG", "JA-NR-c", "JA-R-c", "JA-NR-o", "JA-R-o"):
            if p not in P:
                continue
            rows = P[p]["rows"]
            t = {k: float(np.nanmean([x.get(k, np.nan) for x in rows])) for k in KEYS}
            t["member"] = P[p]["member"]
            tab[p] = t
            print(f"  {p:8s} {t['member']:28s} ret {t['return']:8.2f} ok {t['ok_rate']:.3f} drop {t['drop_rate']:.3f} "
                  f"bler {t['bler']:.3f} shbler {t['shared_bler']:.3f} unpred {t['unpred_shared_frac']:.2f} "
                  f"cochan {t['cochannel_air_frac']:.3f} goodput {t['goodput_kbps']:6.1f} E {t['energy_kj']:.1f} "
                  f"pad {t['pad_air_s']:.0f} jain {t['jain']:.3f} minUAV {t['min_uav_ok']:.3f} "
                  f"reusefrac {t['ja_reuse_frac']:.2f}")
        R = lambda p: np.array([x["return"] for x in P[p]["rows"]])
        con = {}
        for name, a, b in (("selection gain, causal (JA-NR-c - NR)", "JA-NR-c", "NR"),
                           ("reuse gain, causal (JA-R-c - JA-NR-c)", "JA-R-c", "JA-NR-c"),
                           ("JA-R-c - NR", "JA-R-c", "NR"),
                           ("selection gain, ORACLE (JA-NR-o - NR)", "JA-NR-o", "NR"),
                           ("reuse gain, ORACLE (JA-R-o - JA-NR-o)", "JA-R-o", "JA-NR-o"),
                           ("information gain, no reuse (JA-NR-o - JA-NR-c)", "JA-NR-o", "JA-NR-c"),
                           ("RU - NR", "RU", "NR"), ("RUG - NR", "RUG", "NR")):
            if a in P and b in P:
                assert P[a]["seeds"] == P[b]["seeds"]
                con[name] = ci(R(a) - R(b))
                print(f"  {name:48s} {fmt(con[name])}")
        out[v] = dict(policies=tab, contrasts=con)
    return out


def main(d):
    d = Path(d)
    res = {}
    for f in ("invariance.json.gz", "invariance.json"):
        if (d / f).exists():
            res["invariance"] = invariance(d / f)
            break
    if (d / "eval.json.gz").exists():
        res["eval"] = evaluation(d / "eval.json.gz")
    if (d / "eval_mcsg.json.gz").exists():
        res["eval_mcsg"] = evaluation(d / "eval_mcsg.json.gz")
    json.dump(res, open(d / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
