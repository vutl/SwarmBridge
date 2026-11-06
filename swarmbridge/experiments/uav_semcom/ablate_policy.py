"""Which part of PPO's action carries its edge over the tuned heuristic? Mix the two policies component-wise on the
same test seeds: each of {move, channel+power, mode} is taken from PPO or from the heuristic (2^3 combinations;
"HHH" is the heuristic itself, "PPP" is PPO).

Usage: python ablate_policy.py MODEL.zip OUT_JSON --heur paced_q3 --profiles ... --cpu-file ... --modes ... --cfg JSON
"""
import argparse
import itertools
import json

import numpy as np
from stable_baselines3 import PPO

import env as E
import gym_env as G
import policies as P


def heur_from_name(n):
    if n.startswith("stag_"):
        parts = n.split("_")
        return P.staggered(int(parts[2][1:]) if len(parts) > 2 else None, parts[1])
    if n.startswith("pacedT"):
        return P.paced_time(float(n[6:]))
    if n.startswith("paced_q"):
        return P.paced(int(n[7]), True, queue_aware=n.endswith("_qa"))
    return P.greedy(True, queue_aware=n == "greedy_qa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--heur", required=True)
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    table = E.ModeTable(args.profiles, args.modes.split(","), args.cpu_file)
    _, te = table.split()
    model = PPO.load(args.model, device="cpu")
    g = G.GymUavSemcom(table, te, cfg, seed=0)
    K, per = g.core.c["K"], g._per
    heur = heur_from_name(args.heur)

    def mixed(src):   # src = (move, chpow, mode) each 'P' or 'H'
        def pol(env):
            a, _ = model.predict(env.obs(), deterministic=True)
            pa = [tuple(int(x) for x in row) for row in np.asarray(a).reshape(K, per)]
            ha = heur(env)
            out = []
            for p, h in zip(pa, ha):
                mv = p[0] if src[0] == "P" else h[0]
                ch, pl = (p[1], p[2]) if src[1] == "P" else (h[1], h[2])
                md = p[3] if src[2] == "P" else h[3]
                out.append((mv, ch, pl, md))
            return out
        return pol

    seeds = list(range(1000, 1000 + args.n))
    res = {}
    for src in itertools.product("PH", repeat=3):
        tag = "".join(src)
        eps = [P.run_episode(E.UavSemcomEnv(table, te, cfg), mixed(src), s) for s in seeds]
        res[tag] = eps
        print(f"move/chpow/mode={tag}  return {np.mean([m['return'] for m in eps]):+7.2f}  util "
              f"{np.mean([m['mean_util'] for m in eps]):+.3f}  ok {np.mean([m['ok_rate'] for m in eps]):.2f}  "
              f"mission {np.mean([m['mission_s'] for m in eps]):.0f}s", flush=True)
    json.dump(dict(cfg=cfg, heur=args.heur, seeds=seeds, results=res), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
