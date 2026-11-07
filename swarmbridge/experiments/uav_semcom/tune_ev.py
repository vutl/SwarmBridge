"""Fair baseline for the two-timescale envs (A4 survey / reference ev): tune the heuristic family on TRAIN images and
train episode seeds 0..29, pick the member with the best mean return, and report every member on the TEST images and
test seeds 1000..1029 (the seeds train_ev.py evaluates PPO on). Slot part from the v2 heuristic family (tune_heur);
mode part = event-greedy (env_ev.greedy_mode), or the fixed mode for fixed_* members. No RL involved: the tuned
heuristic is chosen before any PPO result is seen.
Usage: python tune_ev.py OUT_JSON --env survey --profiles ... --cpu-file ... --modes ... --cfg JSON
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np

import env as E
import env_ev as V
import env_survey as SV
import policies as P
from tune_heur import family

CORES = {"ev": V.UavSemcomEnvEv, "survey": SV.UavSurveyEnv}
G = {}


def init(profiles, modes, cpu_file, env, cfg):
    G["table"] = E.ModeTable(profiles, modes, cpu_file)
    G["tr"], G["te"] = G["table"].split()
    G["modes"], G["core"], G["cfg"] = modes, CORES[env], cfg


def run(a):
    name, which = a
    pool, seeds = (G["tr"], range(30)) if which == "train" else (G["te"], range(1000, 1030))
    if name.startswith("fixed_"):
        j = G["modes"].index(name[6:])
        sp, mp = P.fixed(j), V.fixed_mode(j)
    else:
        sp, mp = P.from_name(name, G["modes"]), V.greedy_mode
    return a, [V.run_episode_ev(G["core"](G["table"], pool, G["cfg"]), sp, mp, s) for s in seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--env", default="survey", choices=list(CORES))
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--procs", type=int, default=5)
    args = ap.parse_args()
    modes, cfg = args.modes.split(","), json.loads(args.cfg)
    jobs = [(n, w) for n in family(modes) for w in ("train", "test")]
    with Pool(args.procs, initializer=init, initargs=(args.profiles, modes, args.cpu_file, args.env, cfg)) as pool:
        res = pool.map(run, jobs)
    rows = {}
    for (name, which), eps in res:
        r = rows.setdefault(name, {})
        r[f"{which}_return"] = float(np.mean([e["return"] for e in eps]))
        if which == "test":
            r.update(test_util=float(np.mean([e["mean_util"] for e in eps])),
                     test_ok=float(np.mean([e["ok_rate"] for e in eps])),
                     test_mission=float(np.mean([e["mission_s"] for e in eps])),
                     test_return_all=[e["return"] for e in eps])
    best = max(rows, key=lambda n: rows[n]["train_return"])
    json.dump(dict(env=args.env, cfg=cfg, tuned=best, rows=rows), open(args.out, "w"), indent=1)
    for n in sorted(rows, key=lambda n: -rows[n]["train_return"])[:15]:
        r = rows[n]
        print(f"{n:24s} train {r['train_return']:+8.2f}  test {r['test_return']:+8.2f}  ok {r['test_ok']:.3f}  "
              f"mission {r['test_mission']:.0f}s")
    print("TUNED", best)


if __name__ == "__main__":
    main()
