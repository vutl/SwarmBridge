"""A-X2 summary: combinations table and paired contrasts (environment-seed CI and training-seed spread).

Per combination and environment seed the value is the mean over the paired training seeds I = 0..2; contrasts are
per-seed differences, t-interval over environment seeds. Training-seed variation: the contrast computed with
training seed I only (I = 0, 1, 2) and its spread.
Contrasts (fixed with the A-X2 design, before the evaluation was run):
  X1 c|heur - tbest|heur          learned channel scheduler, heuristic modes
  X2 c|heur - sch|heur            ... against the best heuristic channel rule (A-X1 SCH*)
  X3 cm|cmm - tbest|ppo           co-trained channel + mode vs fixed scheduler + mode-only PPO (A5 standard)
  X4 cm|cmm - c|heur              mode learning on top of the learned scheduler
  X5 cm|cmm - jc|jm               properly trained heads vs the A5 joint heads with heuristic move / power (A-X1)
  X6 tbest|ppo+mpc - tbest|ppo    selective PPO-MPC (cold) under the fixed scheduler
  X7 cm|cmm+mpc - cm|cmm          the same on top of the learned scheduler
  X8 cm|cmm+mpc - tbest|ppo+mpc   full stack vs the current Paper A controller (cold)
Diagnostics (out-of-distribution pairings, labelled OOD):
  I1 (cm|cmm - cm|ppo) - (tbest|cmm - tbest|ppo)   scheduler x mode-head interaction (each mode head trained under one
                                                  of the two schedulers)
  I2 (c|ppo - c|heur) - (tbest|ppo - tbest|heur)    does mode learning gain more under the learned scheduler?
Usage (repo root): python swarmbridge/experiments/uav_semcom/ax2_summary.py runs/uav_semcom/ax2/eval_K5.json"""
import gzip
import json
import sys

import numpy as np
from scipy import stats

import a5_common as C

FID = C.REPO / "runs" / "uav_semcom" / "a5" / "fidelity" / "per_image_val.jsonl"
CONTRASTS = {"X1": ("c|heur", "tbest|heur"), "X2": ("c|heur", "sch|heur"), "X3": ("cm|cmm", "tbest|ppo"),
             "X4": ("cm|cmm", "c|heur"), "X5": ("cm|cmm", "jc|jm"), "X6": ("tbest|ppo+mpc", "tbest|ppo"),
             "X7": ("cm|cmm+mpc", "cm|cmm"), "X8": ("cm|cmm+mpc", "tbest|ppo+mpc"),
             "S1": ("tbest_env|heur_env", "tbest|heur"), "S2": ("cm|cmm", "tbest_env|heur_env"),
             "S3": ("c|heur", "tbest_env|heur_env"), "S4": ("tbest|ppo", "tbest_env|heur_env")}
INTER = {"I1": (("cm|cmm", "cm|ppo"), ("tbest|cmm", "tbest|ppo")), "I2": (("c|ppo", "c|heur"), ("tbest|ppo", "tbest|heur"))}
METRICS = ("return", "ok_rate", "late_rate", "drop_rate", "q_viol_rate", "mean_psnr", "energy_kj", "jain", "min_uav_ok",
           "cochannel_air_frac", "chan_util", "ms_slot_p99", "ms_mode_p99", "mode_over50")


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def fmt(c):
    return f"{c[0]:+6.2f} [{c[1]:+6.2f},{c[2]:+6.2f}]"


def recall(deliv, per, man):
    tp = gt = 0
    for ep in deliv:
        for mi, mode, ok in ep:
            o = per.get((man[mi]["name"], "original"))
            if o is None:
                continue
            gt += o["n_gt"]
            row = per.get((man[mi]["name"], C.MODES[mode])) if mode >= 0 else None
            if ok and row is not None:
                tp += row["tp"]
    return tp / gt if gt else float("nan")


def main(path):
    d = json.load(open(path))
    dl = json.load(gzip.open(path.replace(".json", ".deliv.json.gz"), "rt"))
    per = {}
    for l in open(FID):
        r = json.loads(l)
        per[(r["image"], r["mode"])] = r
    man = C.manifest()["images"]
    rows = d["results"]
    combos = sorted({r["combo"] for r in rows})
    by = {}                                           # combo -> {ckpt: rows}
    for r in rows:
        by.setdefault(r["combo"], {})[r["ckpt"]] = r["rows"]

    def arr(c, m="return", i=None):
        ck = by[c]
        if -1 in ck:
            return np.array([x[m] for x in ck[-1]], float)
        sel = [i] if i is not None else sorted(ck)
        return np.mean([[x[m] for x in ck[j]] for j in sel], 0)

    tab = {}
    print(f"===== ctrl {d['ctrl_regime']}  env {d['env_regime']}  seeds {len(rows[0]['seeds'])}")
    for c in combos:
        t = {m: float(np.nanmean(arr(c, m))) for m in METRICS}
        allrows = [x for ck in by[c].values() for x in ck]
        sv = [x["service"] for x in allrows if x["service"].get("n")]
        t.update(service_ratio=float(np.mean([s["ratio"] for s in sv])), service_nmae=float(np.mean([s["nmae"] for s in sv])),
                 service_zero=float(np.mean([s["zero_frac"] for s in sv])),
                 recall=float(np.mean([recall(dl[f"{c}|{i}"], per, man) for i in by[c]])),
                 per_ckpt_return=[float(np.mean([x["return"] for x in by[c][i]])) for i in sorted(by[c])])
        if "switched" in allrows[0]:
            tfp = [x["t_first_plan"] for x in allrows if x["t_first_plan"] is not None]
            t.update(switched=float(np.mean([x["switched"] for x in allrows])),
                     gated_frac=float(np.mean([x["gated"] / max(x["gated"] + x["planned"], 1) for x in allrows])),
                     t_first_plan_q=[float(v) for v in np.percentile(tfp, [10, 50, 90])] if tfp else None,
                     abort_frac=float(np.mean([x["aborts"] / max(x["planned"], 1) for x in allrows])))
        tab[c] = t
        print(f"  {c:20s} ret {t['return']:7.2f} {np.round(t['per_ckpt_return'], 2)} ok {t['ok_rate']:.3f} "
              f"late {t['late_rate']:.3f} psnr {t['mean_psnr']:.2f} E {t['energy_kj']:.1f} jain {t['jain']:.3f} "
              f"minUAV {t['min_uav_ok']:.3f} cochan {t['cochannel_air_frac']:.3f} util {t['chan_util']:.2f} "
              f"recall {t['recall']:.3f} svc {t['service_ratio']:.2f}/{t['service_nmae']:.2f} "
              f"ms slot/mode p99 {t['ms_slot_p99']:.1f}/{t['ms_mode_p99']:.1f} >50 {t['mode_over50']:.4f}"
              + (f" sw {t['switched']:.0f} gated {t['gated_frac']:.2f} t1 {t['t_first_plan_q']} abort {t['abort_frac']:.3f}"
                 if "switched" in t else ""))
    con = {}
    for name, (a, b) in CONTRASTS.items():
        if a in by and b in by:
            v = dict(ret=ci(arr(a) - arr(b)), ok=ci(arr(a, "ok_rate") - arr(b, "ok_rate")))
            ks = sorted(set(by[a]) & set(by[b]) - {-1}) or sorted((set(by[a]) | set(by[b])) - {-1})
            v["per_training_seed"] = [float(np.mean(arr(a, i=i if i in by[a] else None) - arr(b, i=i if i in by[b] else None)))
                                      for i in ks]
            con[name] = v
            print(f"  {name} {a} - {b:16s} return {fmt(v['ret'])} per training seed {np.round(v['per_training_seed'], 2)} "
                  f"ok {v['ok'][0]:+.3f}")
    for name, ((a1, a2), (b1, b2)) in INTER.items():
        if all(c in by for c in (a1, a2, b1, b2)):
            x = (arr(a1) - arr(a2)) - (arr(b1) - arr(b2))
            con[name] = dict(ret=ci(x), ood=True)
            print(f"  {name} (OOD) ({a1} - {a2}) - ({b1} - {b2}) {fmt(con[name]['ret'])}")
    json.dump(dict(ctrl_regime=d["ctrl_regime"], env_regime=d["env_regime"], combos=tab, contrasts=con),
              open(path.replace(".json", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
