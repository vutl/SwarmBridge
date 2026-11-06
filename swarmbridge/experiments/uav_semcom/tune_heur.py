"""Fair-baseline check for A3: tune the heuristic family on TRAIN images + train episode seeds, pick the best member
by mean episode return (the quantity PPO maximises), then compare it with a trained PPO on the shared TEST seeds.

Family (env v2): every fixed mode, per-image greedy (with / without BS detour), queue pacing q_thr in {0..5}, and
channel-matched staggered surveys (start order idx / near / far; optional pacing).
Usage: python tune_heur.py OUT_JSON --profiles ... --cpu-file ... --modes ... --cfg JSON --ppo-eval EVAL.json
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np

import env as E
import policies as P

G = {}


def family(modes):
    out = [f"fixed_{m}" for m in modes] + ["greedy", "greedy_detour"] + [f"paced_q{q}" for q in range(6)]
    for order in ("idx", "near", "far"):
        out += [f"stag_{order}"] + [f"stag_{order}_q{q}" for q in (0, 1, 2, 3)]
    return out


def run(args):
    name, which, seeds = args
    pool = G["tr"] if which == "train" else G["te"]
    pol = P.from_name(name, G["modes"])
    return name, which, [P.run_episode(E.UavSemcomEnv(G["table"], pool, G["cfg"]), pol, s) for s in seeds]


def init(profiles, modes, cpu_file, cfg):
    G["table"] = E.ModeTable(profiles, modes, cpu_file)
    G["tr"], G["te"] = G["table"].split()
    G["cfg"] = cfg
    G["modes"] = list(modes)


def ci(a, b):
    d = np.asarray(a) - np.asarray(b)
    h = 2.045 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) == 30 else 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--ppo-eval", required=True)
    ap.add_argument("--n-tune", type=int, default=30)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    modes = args.modes.split(",")
    ev = json.load(open(args.ppo_eval))
    assert ev["cfg"] == cfg, (ev["cfg"], cfg)
    test_seeds = ev["seeds"]
    train_seeds = list(range(args.n_tune))   # disjoint from test seeds 1000+
    jobs = [(s, w, train_seeds if w == "train" else test_seeds) for s in family(modes) for w in ("train", "test")]
    with Pool(args.procs, initializer=init, initargs=(args.profiles, modes, args.cpu_file, cfg)) as pool:
        res = pool.map(run, jobs)
    tab = {}
    for name, which, eps in res:
        tab.setdefault(name, {})[which] = eps

    def mean(eps, k):
        return float(np.mean([m[k] for m in eps]))
    best = max(tab, key=lambda n: mean(tab[n]["train"], "return"))
    ppo = ev["results"]["ppo"]
    rows = {n: dict(train_return=mean(v["train"], "return"), test_return=mean(v["test"], "return"),
                    test_util=mean(v["test"], "mean_util"), test_ok=mean(v["test"], "ok_rate"),
                    test_mission=mean(v["test"], "mission_s")) for n, v in tab.items()}
    oracle_best = max(tab, key=lambda n: mean(tab[n]["test"], "return"))   # picked on test: optimistic for the heuristic
    cmp = {}
    for tag, n in (("tuned", best), ("test_best", oracle_best)):
        cmp[tag] = dict(name=n, d_return=ci([m["return"] for m in ppo], [m["return"] for m in tab[n]["test"]]),
                        d_util=ci([m["mean_util"] for m in ppo], [m["mean_util"] for m in tab[n]["test"]]),
                        d_mission=ci([m["mission_s"] for m in ppo], [m["mission_s"] for m in tab[n]["test"]]))
    ppo_row = dict(test_return=mean(ppo, "return"), test_util=mean(ppo, "mean_util"), test_ok=mean(ppo, "ok_rate"),
                   test_mission=mean(ppo, "mission_s"))
    json.dump(dict(cfg=cfg, ppo_eval=args.ppo_eval, train_seeds=train_seeds, test_seeds=test_seeds, rows=rows,
                   ppo=ppo_row, compare=cmp), open(args.out, "w"), indent=1)
    print("cfg", cfg)
    for n, r in sorted(rows.items(), key=lambda x: -x[1]["train_return"]):
        print(f"  {n:16s} train_ret {r['train_return']:+7.2f} test_ret {r['test_return']:+7.2f} util {r['test_util']:+.3f} "
              f"ok {r['test_ok']:.2f} mission {r['test_mission']:.0f}s")
    print(f"  {'PPO':16s} {'':17s} test_ret {ppo_row['test_return']:+7.2f} util {ppo_row['test_util']:+.3f} "
          f"ok {ppo_row['test_ok']:.2f} mission {ppo_row['test_mission']:.0f}s")
    for tag, c in cmp.items():
        print(f"  PPO - {tag} ({c['name']}): return {c['d_return'][0]:+.2f} [{c['d_return'][1]:+.2f},{c['d_return'][2]:+.2f}]"
              f"  util {c['d_util'][0]:+.3f} [{c['d_util'][1]:+.3f},{c['d_util'][2]:+.3f}]")
    print("TUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
