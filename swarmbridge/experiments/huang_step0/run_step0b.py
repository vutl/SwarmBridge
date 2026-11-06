"""Step 0B grid: does temporal coupling leave room for RL over causal heuristics?

Cells: noise {paper -130 dBm, thermal k*chi*B} x M1 payload {fig5 314 kbit, original 3.5 MB} x load lam
{0.5, 1, 2, 4} tasks/s, with all couplings on ('full'), plus single-coupling ablations (queue / visibility /
interference switched off one at a time). Per cell: 10 train seeds (0-9) tune the price policy's lambda; 30 test
seeds (100-129) evaluate every policy.

PRE-REGISTERED GATE (written before running): RL is worth building in a cell only if
  hindsight_tasks (knows future tasks, expected rates) - best causal policy (chosen ON THE TEST SEEDS, which favours
  the heuristics)  >=  0.02 utility per task, with the paired 95% t-interval of per-episode differences above 0.
hindsight_full (also knows the fading draws) is reported as a looser reference, not used by the gate.

Usage: python run_step0b.py OUT_DIR [--procs 6] [--test 30]
"""
import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import model as M
import policies as P
import sim as S

LAMS = (0.5, 1.0, 2.0, 4.0)
PRICES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
T95 = {30: 2.045, 20: 2.093, 10: 2.262}


def cells():
    out = []
    for noise in ("paper", "thermal"):
        for payload in ("fig5", "original"):
            for lam in LAMS:
                base = dict(noise=noise, payload=payload, queue=True, visibility=True, interference=True)
                out.append(("full", lam, base))
                for off in ("queue", "visibility", "interference"):
                    out.append((f"no_{off}", lam, dict(base, **{off: False})))
    return out


def causal_policies(lmb):
    return [P.fixed(k) for k in range(5)] + [P.greedy, P.greedy_area, P.cheapest_feasible, P.price(lmb), P.rollout()]


def tune(job):
    tag, lam, cfg = job
    scores = {}
    for lmb in PRICES:
        scores[lmb] = float(np.mean([S.run(S.Scenario(s, lam, cfg), P.price(lmb))[0] for s in range(10)]))
    return max(scores, key=scores.get), scores


def evaluate(job):
    tag, lam, cfg, lmb, seed = job
    sc = S.Scenario(seed, lam, cfg)
    row = {}
    best_acts, best_u = None, -1e9
    for pol in causal_policies(lmb):
        name = pol.__name__ if not pol.__name__.startswith("price") else "price_tuned"
        u, ok, acts = S.run(sc, pol)
        row[name] = (u, ok)
        if u > best_u:
            best_u, best_acts = u, acts
    t0 = time.time()
    row["hindsight_tasks"] = P.hindsight(sc, best_acts, know_fading=False)[:2]
    row["hindsight_full"] = P.hindsight(sc, best_acts, know_fading=True)[:2]
    row["_hs_seconds"] = (time.time() - t0, 0)
    return tag, lam, cfg["noise"], cfg["payload"], seed, row


def ci(diff):
    d = np.asarray(diff)
    h = T95.get(len(d), 2.0) * d.std(ddof=1) / math.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--test", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cs = cells()
    t0 = time.time()
    with Pool(args.procs) as pool:
        tuned = pool.map(tune, cs)
        print(f"TUNED {len(cs)} cells in {time.time() - t0:.0f}s", flush=True)
        jobs = [(tag, lam, cfg, tuned[ci_][0], seed) for ci_, (tag, lam, cfg) in enumerate(cs)
                for seed in range(100, 100 + args.test)]
        rows = []
        for n, r in enumerate(pool.imap_unordered(evaluate, jobs, chunksize=4)):
            rows.append(r)
            if n % 200 == 0:
                print(f"EVAL {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    with open(out / "episodes.jsonl", "w") as f:
        for tag, lam, noise, payload, seed, row in rows:
            f.write(json.dumps(dict(cell=tag, lam=lam, noise=noise, payload=payload, seed=seed,
                                    **{k: v for k, v in row.items()})) + "\n")
    # summarise
    summary = []
    for ci_, (tag, lam, cfg) in enumerate(cs):
        rs = sorted([r for r in rows if r[0] == tag and r[1] == lam and r[2] == cfg["noise"] and r[3] == cfg["payload"]],
                    key=lambda r: r[4])
        pols = [k for k in rs[0][5] if not k.startswith("_")]
        mean = {p: float(np.mean([r[5][p][0] for r in rs])) for p in pols}
        sat = {p: float(np.mean([r[5][p][1] for r in rs])) for p in pols}
        causal = [p for p in pols if not p.startswith("hindsight")]
        fixed = [p for p in causal if p.startswith("fixed")]
        best_c = max(causal, key=mean.get)
        best_f = max(fixed, key=mean.get)
        d_tasks = ci([r[5]["hindsight_tasks"][0] - r[5][best_c][0] for r in rs])
        d_full = ci([r[5]["hindsight_full"][0] - r[5][best_c][0] for r in rs])
        d_sel = ci([r[5]["greedy"][0] - r[5][best_f][0] for r in rs])
        d_look = ci([max(r[5]["rollout_s3_h6"][0], r[5]["price_tuned"][0]) - r[5]["greedy"][0] for r in rs])
        gate = d_tasks[0] >= 0.02 and d_tasks[1] > 0
        summary.append(dict(cell=tag, lam=lam, noise=cfg["noise"], payload=cfg["payload"], price_lambda=tuned[ci_][0],
                            mean=mean, satisfied=sat, best_causal=best_c, best_fixed=best_f,
                            gap_hindsight_tasks=d_tasks, gap_hindsight_full=d_full,
                            value_of_mode_selection=d_sel, value_of_lookahead_over_greedy=d_look, rl_gate=gate))
    (out / "summary.json").write_text(json.dumps(dict(cells=summary, seconds=time.time() - t0,
                                                      train_seeds=list(range(10)),
                                                      test_seeds=[100, 100 + args.test - 1]), indent=1))
    lines = ["# Step 0B — simplified Huang-inspired SAGIN, temporal coupling", "",
             f"Test seeds 100–{99 + args.test}; utility per task (SCE, −1 if a constraint fails). "
             "Gap columns: mean [95% CI] over paired episodes.", "",
             "| noise | M1 payload | cell | λ | best fixed | greedy | best causal | hindsight tasks − best causal | "
             "hindsight full − best causal | greedy − best fixed | lookahead − greedy | gate |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f3 = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    for s in summary:
        lines.append(f"| {s['noise']} | {s['payload']} | {s['cell']} | {s['lam']} | {s['best_fixed']} "
                     f"{s['mean'][s['best_fixed']]:.3f} | {s['mean']['greedy']:.3f} | {s['best_causal']} "
                     f"{s['mean'][s['best_causal']]:.3f} | {f3(s['gap_hindsight_tasks'])} | "
                     f"{f3(s['gap_hindsight_full'])} | {f3(s['value_of_mode_selection'])} | "
                     f"{f3(s['value_of_lookahead_over_greedy'])} | {'OPEN' if s['rl_gate'] else '-'} |")
    lines += ["", f"Cells passing the RL gate: {sum(s['rl_gate'] for s in summary)}/{len(summary)}.",
              f"Wall time {time.time() - t0:.0f} s, CPU only."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-2:]), flush=True)
    print("RUN_DONE step0b", flush=True)


if __name__ == "__main__":
    main()
