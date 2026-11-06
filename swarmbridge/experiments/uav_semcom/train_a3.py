"""A3: train PPO (centralised, MultiDiscrete hybrid action discretised as in Li et al. JSAC 2026) on the A2 environment
and evaluate it next to the heuristic baselines on the SAME held-out images and episode seeds.

Train images / test images: 70/30 split of the A1 image set (split seed 0). Train episode seeds random; test seeds
1000..1000+N_TEST-1, shared by every policy (common random numbers).
Usage: python train_a3.py OUT_DIR --profiles P1.jsonl [P2.jsonl] --modes m1,m2,... [--steps 2e6] [--cfg JSON]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

import env as E
import gym_env as G
import policies as P


def make(table, pool, cfg, seed):
    def f():
        return G.GymUavSemcom(table, pool, cfg, seed)
    return f


def evaluate_rl(model, table, pool, cfg, seeds):
    out = []
    for s in seeds:
        e = G.GymUavSemcom(table, pool, cfg, seed=s, seeds=[s])
        obs, _ = e.reset()
        done, tot = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = e.step(a)
            tot += r
        info["return"] = tot
        out.append(info)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file", default=None)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--steps", type=float, default=2e6)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--n-env", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.cfg)
    modes = args.modes.split(",")
    table = E.ModeTable(args.profiles, modes, args.cpu_file)
    tr, te = table.split()
    seeds = list(range(1000, 1000 + args.n_test))
    venv = SubprocVecEnv([make(table, tr, cfg, args.seed * 100 + i) for i in range(args.n_env)])
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=1024, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.01, policy_kwargs=dict(net_arch=[256, 256]), seed=args.seed, verbose=0, device="cpu")
    t0 = time.time()
    log = []
    chunk = int(2e5)
    done_steps = 0
    while done_steps < args.steps:
        model.learn(chunk, reset_num_timesteps=False)
        done_steps += chunk
        ev = evaluate_rl(model, table, te, cfg, seeds[:10])
        rec = dict(steps=done_steps, sec=time.time() - t0, util=float(np.mean([m["mean_util"] for m in ev])),
                   ok=float(np.mean([m["ok_rate"] for m in ev])))
        log.append(rec)
        print(f"STEP {done_steps} util {rec['util']:+.3f} ok {rec['ok']:.2f} {rec['sec']:.0f}s", flush=True)
    model.save(out / f"ppo_s{args.seed}")
    res = {"ppo": evaluate_rl(model, table, te, cfg, seeds)}
    pols = [P.fixed(j) for j in range(len(modes))] + [P.greedy(), P.greedy(detour=True)] + \
        [P.paced(q) for q in (0, 1, 2, 4)] + [P.staggered(None, "idx"), P.staggered(0, "far")]
    for p in pols:
        name = p.__name__ if not p.__name__.startswith("fixed") else "fixed_" + modes[int(p.__name__.split("_")[1])]
        res[name] = [P.run_episode(E.UavSemcomEnv(table, te, cfg), p, s) for s in seeds]
    (out / f"eval_s{args.seed}.json").write_text(json.dumps(dict(cfg=cfg, modes=modes, seeds=seeds, log=log,
                                                                 results=res), indent=1))
    for k, v in res.items():
        print(f"{k:28s} util {np.mean([m['mean_util'] for m in v]):+.3f} ok {np.mean([m['ok_rate'] for m in v]):.2f} "
              f"cap {np.mean([m['captured'] for m in v]):.2f} E {np.mean([m['energy_kj'] for m in v]):.0f}kJ "
              f"mission {np.mean([m['mission_s'] for m in v]):.0f}s return {np.mean([m['return'] for m in v]):+.1f}")
    print("RUN_DONE a3", flush=True)


if __name__ == "__main__":
    main()
