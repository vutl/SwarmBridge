"""Summarize the A4 fresh-seed confirmation (final_a4.py JSONs) as markdown.

Baselines fixed BEFORE the fresh seeds were drawn: the train-tuned member of the base family (tune_v20 JSON) and the
train-tuned member of the union base + budget families (best train return over both JSONs). The best heuristic ON
the fresh seeds is also shown (selected on test: optimistic for the heuristic side).
Paired differences: per scenario (seed), PPO - baseline; mean and 95% t-interval over the scenarios. "pooled" = for
each scenario, the mean over the 3 training seeds' policies.
Usage: python summarize_final_a4.py OUT_MD FINAL_JSON... --tune T.json --tune-budget TB.json [--train-dirs label=dir...]
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import t as _t

    def tcrit(n):
        return float(_t.ppf(0.975, n - 1))
except ImportError:  # pragma: no cover
    def tcrit(n):
        return 1.984 if n >= 100 else 2.045

KEYS = ("return", "mean_util", "ok_rate", "drop_rate", "q_viol_rate", "mean_psnr", "mean_delay", "mission_s",
        "energy_kj")


def ci(x):
    x = np.asarray(x, float)
    h = tcrit(len(x)) * x.std(ddof=1) / math.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def fmt(m, lo, hi):
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def row_metrics(eps):
    return {k: float(np.nanmean([e[k] for e in eps])) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("finals", nargs="+")
    ap.add_argument("--tune", required=True)
    ap.add_argument("--tune-budget", required=True)
    ap.add_argument("--train-dirs", nargs="*", default=[], help="LABEL=EVAL_JSON (for the experience table)")
    args = ap.parse_args()
    res, seeds = {}, None
    for f in args.finals:
        d = json.load(open(f))
        assert seeds is None or d["seeds"] == seeds, "all final JSONs must use the same seeds"
        seeds = d["seeds"]
        for label, combos in d["results"].items():
            res.setdefault(label, {}).update(combos)
    heur = {k[2:]: v["-"] for k, v in res.items() if k.startswith("H:")}
    tb, tbb = json.load(open(args.tune)), json.load(open(args.tune_budget))
    base_tuned = tb["tuned"]
    union = {**{n: r["train_return"] for n, r in tb["rows"].items()},
             **{n: r["train_return"] for n, r in tbb["rows"].items()}}
    union_tuned = max(union, key=union.get)
    best_test = max(heur, key=lambda n: np.mean([e["return"] for e in heur[n]]))
    refs = {"union-tuned " + union_tuned: heur[union_tuned], "base-tuned " + base_tuned: heur[base_tuned],
            "best on fresh seeds " + best_test: heur[best_test]}
    L = [f"Fresh test seeds {seeds[0]}..{seeds[-1]} (n = {len(seeds)}). Baselines tuned on train seeds before these "
         f"seeds were used: base family -> `{base_tuned}`, base+budget families -> `{union_tuned}`. "
         f"Best heuristic on the fresh seeds (selected on test, optimistic): `{best_test}`.", ""]

    # groups of PPO policies: label prefix -> list of labels (one per training seed)
    groups = defaultdict(list)
    for label in res:
        if not label.startswith("H:"):
            groups[label.rsplit("_s", 1)[0]].append(label)
    L += ["| policy | return | ok | PSNR | mission s | energy kJ | delay s | "
          + " | ".join(f"Δ vs {r.split(' ')[0]} {r.split(' ')[-1]}" for r in refs) + " |",
          "|---|---|---|---|---|---|---|" + "---|" * len(refs)]

    def line(name, eps_per_seed):
        m = row_metrics([e for eps in eps_per_seed for e in eps])
        cells = []
        for r in refs.values():
            diff = np.mean([[a["return"] - b["return"] for a, b in zip(eps, r)] for eps in eps_per_seed], axis=0)
            cells.append(fmt(*ci(diff)))
        return (f"| {name} | {m['return']:+.2f} | {m['ok_rate']:.3f} | {m['mean_psnr']:.1f} | {m['mission_s']:.1f} | "
                f"{m['energy_kj']:.1f} | {m['mean_delay']:.2f} | " + " | ".join(cells) + " |")
    for r, eps in refs.items():
        L.append(line(r, [eps]))
    for g in sorted(groups):
        labs = sorted(groups[g])
        for lab in labs:
            L.append(line(lab, [res[lab]["PPPP"]]))
        if len(labs) > 1:
            L.append(line(f"**{g} pooled ({len(labs)} seeds)**", [res[lab]["PPPP"] for lab in labs]))
    L.append("")

    # action-replacement ablation for joint groups
    for g in sorted(groups):
        labs = sorted(groups[g])
        if not all(len(res[lab]) == 16 for lab in labs):
            continue
        L += [f"Action replacement, `{g}` (P = PPO, H = heuristic: slot parts of `{json.load(open(args.finals[0]))['slot_heur']}`, "
              "event-greedy mode); order move / channel / power / mode; return pooled over the training seeds; "
              "Δ = combo - PPPP (paired, pooled).", "",
              "| combo | return | Δ vs PPPP | ok | PSNR | mission s |", "|---|---|---|---|---|---|"]
        full = np.mean([[e["return"] for e in res[lab]["PPPP"]] for lab in labs], axis=0)
        for combo in sorted(res[labs[0]], key=lambda c: (c.count("H"), c)):
            per = [res[lab][combo] for lab in labs]
            m = row_metrics([e for eps in per for e in eps])
            d = np.mean([[e["return"] for e in eps] for eps in per], axis=0) - full
            L.append(f"| {combo} | {m['return']:+.2f} | {fmt(*ci(d)) if combo != 'PPPP' else '-'} | {m['ok_rate']:.3f} | "
                     f"{m['mean_psnr']:.1f} | {m['mission_s']:.1f} |")
        L.append("")

    # experience consumed in training
    if args.train_dirs:
        L += ["Training experience at the final checkpoint (2M PPO decision steps each):", "",
              "| run | episodes | env s | UAV-s | RL slot steps | RL mode steps | heuristic slot | heuristic mode |",
              "|---|---|---|---|---|---|---|---|"]
        for spec in args.train_dirs:
            lab, path = spec.split("=", 1)
            e = json.load(open(path))["log"][-1].get("experience")
            if e:
                L.append(f"| {lab} | {e['episodes']} | {e['env_s']:.0f} | {e['uav_s']:.0f} | {e.get('rl_steps_slot', 0)} | "
                         f"{e.get('rl_steps_mode', 0)} | {e.get('heur_slot', 0)} | {e.get('heur_mode', 0)} |")
        L.append("")
    open(args.out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
