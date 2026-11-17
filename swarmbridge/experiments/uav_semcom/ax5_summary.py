"""Review #39 pilot summary (ax5_eval output).

Per env regime: every policy's mean metrics (return, ok, late, BLER, shared-block BLER, shared-block fraction,
co-channel airtime, failed airtime, goodput, energy, Jain, min-UAV ok, slot latency p99); learned policies averaged over
their 3 training seeds per environment seed. Paired contrasts (t-interval over environment seeds, plus per training
seed) against the reference REF (default T:best@K5_la_rand, the tuned heuristic of the training distribution).
Adaptation (the review's criterion "adapt, not merely disable reuse"): on K5_la_rand (hidden parameters drawn per
episode), Spearman correlation of each policy's per-episode co-channel airtime with the hidden CQI error, feedback
delay, OLLA step and CQI bias, and its mean co-channel airtime per fixed condition (good -> bad).
Usage: python ax5_summary.py runs/uav_semcom/ax5/eval.json.gz [REF]   (plain .json also accepted)"""
import gzip
import json
import sys

import numpy as np
from scipy import stats

from ax2_summary import ci, fmt

KEYS = ("return", "ok_rate", "late_rate", "bler", "shared_bler", "shared_block_frac", "cochannel_air_frac",
        "fail_air_frac", "goodput_kbps", "energy_kj", "jain", "min_uav_ok", "gate_flags")


def main(path, ref="T:best@K5_la_rand"):
    d = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    by = {}
    for r in d["results"]:
        by.setdefault(r["env_regime"], {}).setdefault(r["spec"], {})[r["ckpt"]] = r["rows"]

    def arr(env, spec, k="return", i=None):
        ck = by[env][spec]
        sel = [i] if i is not None else sorted(ck)
        return np.mean([[x[k] for x in ck[j]] for j in sel], 0)

    out = {}
    for env in by:
        print(f"===== {env}")
        tab = {}
        for spec in sorted(by[env]):
            t = {k: float(np.nanmean(arr(env, spec, k))) for k in KEYS}
            rows = [x for ck in by[env][spec].values() for x in ck]
            if "ms_slot_p99" in rows[0]:
                t["ms_slot_p99"] = float(np.mean([x["ms_slot_p99"] for x in rows]))
            t["per_ckpt_return"] = [float(np.mean([x["return"] for x in by[env][spec][i]])) for i in sorted(by[env][spec])]
            tab[spec] = t
            print(f"  {spec:26s} ret {t['return']:8.2f} {np.round(t['per_ckpt_return'], 1)} ok {t['ok_rate']:.3f} "
                  f"bler {t['bler']:.3f} shbler {t['shared_bler']:.3f} cochan {t['cochannel_air_frac']:.3f} "
                  f"failair {t['fail_air_frac']:.3f} goodput {t['goodput_kbps']:6.1f} E {t['energy_kj']:.1f} "
                  f"jain {t['jain']:.3f} minUAV {t['min_uav_ok']:.3f}"
                  + (f" ms {t['ms_slot_p99']:.1f}" if "ms_slot_p99" in t else ""))
        con = {}
        if ref in by[env]:
            for spec in sorted(by[env]):
                if spec == ref:
                    continue
                c = dict(ret=ci(arr(env, spec) - arr(env, ref)),
                         per_training_seed=[float(np.mean(arr(env, spec, i=i) - arr(env, ref))) for i in sorted(by[env][spec])])
                con[spec] = c
                print(f"  {spec:26s} - {ref}: {fmt(c['ret'])} per training seed {np.round(c['per_training_seed'], 2)}")
        out[env] = dict(policies=tab, contrasts_vs_ref=con)
    if "K5_la_rand" in by:
        print("===== adaptation on K5_la_rand: Spearman(co-channel airtime, hidden parameter)")
        adapt = {}
        for spec in sorted(by["K5_la_rand"]):
            rows = [x for ck in by["K5_la_rand"][spec].values() for x in ck]
            co = np.array([x["cochannel_air_frac"] for x in rows])
            if co.std() < 1e-9:
                continue
            a = {}
            for hk in ("la_cqi_sigma", "la_fb_delay", "la_olla_step", "la_cqi_bias"):
                h = np.array([x["la_hidden"][hk] for x in rows])
                rho, p = stats.spearmanr(co, h)
                a[hk] = dict(rho=float(rho), p=float(p))
            adapt[spec] = a
            print(f"  {spec:26s} " + " ".join(f"{k[3:]} {v['rho']:+.2f}(p={v['p']:.3f})" for k, v in a.items()))
        out["adaptation_rand"] = adapt
    conds = [c for c in ("K5_la_good", "K5_la", "K5_la_noisy", "K5_la_slow", "K5_la_bad") if c in by]
    if conds:
        print("===== co-channel airtime by fixed condition (good -> bad)")
        specs = sorted(set.intersection(*[set(by[c]) for c in conds]))
        out["reuse_by_condition"] = {s: {c: float(np.mean(arr(c, s, "cochannel_air_frac"))) for c in conds} for s in specs}
        for s in specs:
            print(f"  {s:26s} " + " ".join(f"{c[6:] or 'nom'}:{out['reuse_by_condition'][s][c]:.3f}" for c in conds))
    json.dump(out, open(path.replace(".gz", "").replace(".json", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
