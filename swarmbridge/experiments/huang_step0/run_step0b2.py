"""Step 0B, corrected reference (post-hoc, see report): value of knowing future tasks, closed loop.

The pre-registered gate of run_step0b.py compared against an OPEN-LOOP hindsight plan (future tasks known, expected
rates, never re-planned); under real fading that plan cascades into deadline misses and is often worse than the
heuristics, so it is not an upper reference. Here the reference is a CLOSED-LOOP lookahead that sees the true next
H tasks but only expected rates and re-plans from the realised state at every decision ('foresight'). The matching
causal planner samples futures from the task distribution ('rollout'). Gate: foresight - best causal >= 0.02 with
the paired 95% t-interval above 0 (an upper estimate of what a learned anticipatory policy could add).

Cells: the 16 'full' cells of step 0B (all couplings on), same test seeds 100-129, price lambda reused from 0B.
Usage: python run_step0b2.py STEP0B_DIR OUT_DIR [--procs 6]
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import model as M
import policies as P
import sim as S
from run_step0b import ci


def job(a):
    noise, payload, lam, lmb, seed = a
    cfg = dict(noise=noise, payload=payload, queue=True, visibility=True, interference=True)
    sc = S.Scenario(seed, lam, cfg)
    pols = [P.fixed(k) for k in range(5)] + [P.greedy, P.greedy_area, P.cheapest_feasible, P.price(lmb), P.rollout(),
                                             P.rollout2(base="all"), P.rollout2(base="area"),
                                             P.rollout2(base="all", foresight=True), P.rollout2(base="area", foresight=True)]
    row = {}
    for p in pols:
        name = "price_tuned" if p.__name__.startswith("price") else p.__name__
        row[name] = S.run(sc, p)[:2]
    return noise, payload, lam, seed, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step0b")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prev = json.loads((Path(args.step0b) / "summary.json").read_text())["cells"]
    full = [c for c in prev if c["cell"] == "full"]
    jobs = [(c["noise"], c["payload"], c["lam"], c["price_lambda"], s) for c in full for s in range(100, 130)]
    t0 = time.time()
    with Pool(args.procs) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            rows.append(r)
            if n % 60 == 0:
                print(f"EVAL {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    with open(out / "episodes.jsonl", "w") as f:
        for noise, payload, lam, seed, row in rows:
            f.write(json.dumps(dict(noise=noise, payload=payload, lam=lam, seed=seed, **row)) + "\n")
    summ, lines = [], ["# Step 0B-2 — closed-loop foresight reference (post-hoc correction)", "",
                       "| noise | M1 payload | λ | best causal | foresight best | foresight − best causal | "
                       "foresight − matching rollout | gate |", "|---|---|---|---|---|---|---|---|"]
    f3 = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    for c in full:
        rs = sorted([r for r in rows if (r[0], r[1], r[2]) == (c["noise"], c["payload"], c["lam"])], key=lambda r: r[3])
        names = list(rs[0][4])
        mean = {p: float(np.mean([r[4][p][0] for r in rs])) for p in names}
        causal = [p for p in names if not p.startswith("foresight")]
        fore = [p for p in names if p.startswith("foresight")]
        bc, bf = max(causal, key=mean.get), max(fore, key=mean.get)
        d = ci([r[4][bf][0] - r[4][bc][0] for r in rs])
        match = bf.replace("foresight", "rollout") + "_s3"
        dm = ci([r[4][bf][0] - r[4][match][0] for r in rs])
        gate = d[0] >= 0.02 and d[1] > 0
        summ.append(dict(noise=c["noise"], payload=c["payload"], lam=c["lam"], mean=mean, best_causal=bc,
                         best_foresight=bf, gap=d, gap_vs_matching_rollout=dm, gate=gate))
        lines.append(f"| {c['noise']} | {c['payload']} | {c['lam']} | {bc} {mean[bc]:.3f} | {bf} {mean[bf]:.3f} | "
                     f"{f3(d)} | {f3(dm)} | {'OPEN' if gate else '-'} |")
    lines += ["", f"Cells passing: {sum(s['gate'] for s in summ)}/{len(summ)}; wall {time.time() - t0:.0f} s, CPU."]
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("RUN_DONE step0b2", flush=True)


if __name__ == "__main__":
    main()
