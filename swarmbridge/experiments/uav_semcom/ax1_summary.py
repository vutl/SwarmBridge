"""A-X1 pilot summary: paired seed-level contrasts between scheduler x mode-controller combinations.

Per combination the per-seed value is the mean over the PPO checkpoints it uses (checkpoint i of the joint and of
the mode-only PPO are paired); contrasts are per-seed differences with a t-interval over the environment seeds.
Contrasts (stated with the pilot design, before the run):
  C1 sch|heur  - tbest|heur      a better-tuned heuristic channel rule
  C2 jc|heur   - sch|heur        learned channel head vs the best heuristic rule, heuristic modes
  C3 jc|ppo    - sch|ppo         the same with mode-only PPO modes
  C4 interaction (jc|ppo - jc|heur) - (sch|ppo - sch|heur): does mode learning gain more under the learned scheduler?
  C5 jc|jm     - tbest|ppo       the A5 post-hoc hypothesis (heuristic move + power, PPO channel + mode) vs the A5
                                 mode-only PPO standard
  C6 joint|jm  - jc|jm           learned movement + power on top
Detector recall of what is delivered on time: a5_fidelity per_image_val.jsonl join (as a5_summarize.fidelity_join).
Run from the repo root: python swarmbridge/experiments/uav_semcom/ax1_summary.py"""
import gzip
import json

import numpy as np
from scipy import stats

import a5_common as C

D = C.REPO / "runs" / "uav_semcom" / "ax1"
FID = C.REPO / "runs" / "uav_semcom" / "a5" / "fidelity" / "per_image_val.jsonl"
METRICS = ("return", "ok_rate", "late_rate", "drop_rate", "q_viol_rate", "mean_psnr", "energy_kj", "jain", "min_uav_ok",
           "cochannel_air_frac", "chan_util", "ms_slot_mean", "ms_mode_mean", "ms_decision_p99")
CONTRASTS = {"C1": ("sch|heur", "tbest|heur"), "C2": ("jc|heur", "sch|heur"), "C3": ("jc|ppo", "sch|ppo"),
             "C5": ("jc|jm", "tbest|ppo"), "C6": ("joint|jm", "jc|jm"), "mode_gain_tbest": ("tbest|ppo", "tbest|heur"),
             "jc|jm-vs-tbest|heur": ("jc|jm", "tbest|heur")}


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def fmt(c):
    return f"{c[0]:+6.2f} [{c[1]:+6.2f},{c[2]:+6.2f}]"


def recall(deliv, per, man):
    tp = gt = tps = ns = 0
    for ep in deliv:
        for mi, mode, ok in ep:
            o = per.get((man[mi]["name"], "original"))
            if o is None:
                continue
            gt += o["n_gt"]; ns += o["n_small"]
            row = per.get((man[mi]["name"], C.MODES[mode])) if mode >= 0 else None
            if ok and row is not None:
                tp += row["tp"]; tps += row["tp_small"]
    return (tp / gt if gt else float("nan"), tps / ns if ns else float("nan"), gt)


def main():
    d = json.load(open(D / "pilot.json"))
    dl = json.load(gzip.open(D / "pilot.deliv.json.gz", "rt"))
    per = {}
    if FID.exists():
        for l in open(FID):
            r = json.loads(l)
            per[(r["image"], r["mode"])] = r
    man = C.manifest()["images"]
    out = {}
    for R in sorted({r["regime"] for r in d["results"]}):
        rows = [r for r in d["results"] if r["regime"] == R]
        combos = sorted({r["combo"] for r in rows})
        val = {}
        tab = {}
        for c in combos:
            rr = [r for r in rows if r["combo"] == c]
            val[c] = {m: np.nanmean([[x.get(m, np.nan) for x in r["rows"]] for r in rr], 0) for m in METRICS}
            rec = [recall(dl[f"{R}|{c}|{r['ckpt']}"], per, man) for r in rr] if per else []
            tab[c] = {m: float(np.nanmean(val[c][m])) for m in METRICS}
            if rec:
                tab[c].update(recall=float(np.mean([x[0] for x in rec])), recall_small=float(np.mean([x[1] for x in rec])))
        print(f"===== {R}   SCH* = {d['sch'][R]}   seeds {len(rows[0]['seeds'])}")
        for c in combos:
            t = tab[c]
            print(f"  {c:11s} ret {t['return']:7.2f} ok {t['ok_rate']:.3f} late {t['late_rate']:.3f} drop {t['drop_rate']:.3f} "
                  f"psnr {t['mean_psnr']:.2f} E {t['energy_kj']:.1f} jain {t['jain']:.3f} minUAV {t['min_uav_ok']:.3f} "
                  f"cochan {t['cochannel_air_frac']:.3f} util {t['chan_util']:.3f} ms slot/mode/p99 {t['ms_slot_mean']:.2f}/"
                  f"{t['ms_mode_mean']:.2f}/{t['ms_decision_p99']:.2f}"
                  + (f" recall {t['recall']:.3f} small {t['recall_small']:.3f}" if "recall" in t else ""))
        con = {}
        for name, (a, b) in CONTRASTS.items():
            if a in val and b in val:
                con[name] = dict(ret=ci(val[a]["return"] - val[b]["return"]), ok=ci(val[a]["ok_rate"] - val[b]["ok_rate"]))
        if all(c in val for c in ("jc|ppo", "jc|heur", "sch|ppo", "sch|heur")):
            dd = (val["jc|ppo"]["return"] - val["jc|heur"]["return"]) - (val["sch|ppo"]["return"] - val["sch|heur"]["return"])
            con["C4_interaction"] = dict(ret=ci(dd))
        for k, v in con.items():
            print(f"  {k:22s} return {fmt(v['ret'])}" + (f"  ok {v['ok'][0]:+.3f} [{v['ok'][1]:+.3f},{v['ok'][2]:+.3f}]" if "ok" in v else ""))
        out[R] = dict(combos=tab, contrasts=con, sch=d["sch"][R])
    json.dump(out, open(D / "pilot_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
