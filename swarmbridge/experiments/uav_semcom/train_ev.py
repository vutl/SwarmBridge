"""Train MaskablePPO on the two-timescale env (A2 v2.2 reference workload, or the A4 survey workload) and evaluate it
next to heuristics on the same held-out images and test seeds.

  --env ev | survey          workload (env_ev.UavSemcomEnvEv / env_survey.UavSurveyEnv)
  --rl-part joint | movement | mode
        joint    : the policy takes slot decisions (move / channel / power) and event mode decisions
        movement : the policy takes slot decisions; modes by the event-greedy heuristic (env_ev.greedy_mode)
        mode     : the policy takes mode decisions; slot decisions by --slot-heur (a v2 heuristic name)
Monitoring during training uses VALIDATION seeds 2000..2009 (never the test seeds); the final checkpoint is fixed
in advance (last one). Test seeds 1000..1000+N_TEST-1 are used once, at the end, for every policy.
Usage: python train_ev.py OUT --profiles ... --modes ... [--cpu-file F] [--env ev] [--rl-part joint] [--cfg JSON]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

import env as E
import env_ev as V
import env_survey as SV
import gym_env_ev as GE
import policies as P

CORES = {"ev": V.UavSemcomEnvEv, "survey": SV.UavSurveyEnv}


def parts(args, modes):
    if args.rl_part == "movement":
        return dict(mode_pol=V.greedy_mode)
    if args.rl_part == "mode":
        return dict(slot_pol=P.from_name(args.slot_heur, modes))
    return {}


def make(table, pool, cfg, seed, core, kw):
    def f():
        return GE.GymUavSemcomEv(table, pool, cfg, seed, core_cls=core, **kw)
    return f


def evaluate_rl(model, table, pool, cfg, seeds, core, kw):
    out = []
    for s in seeds:
        e = GE.GymUavSemcomEv(table, pool, cfg, seed=s, seeds=[s], core_cls=core, **kw)
        obs, _ = e.reset()
        done, tot = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True, action_masks=e.action_masks())
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
    ap.add_argument("--env", default="ev", choices=list(CORES))
    ap.add_argument("--rl-part", default="joint", choices=["joint", "movement", "mode"])
    ap.add_argument("--slot-heur", default="greedy")
    ap.add_argument("--steps", type=float, default=2e6)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--n-env", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=float, default=2e5)
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.cfg)
    modes = args.modes.split(",")
    core = CORES[args.env]
    table = E.ModeTable(args.profiles, modes, args.cpu_file)
    tr, te = table.split()
    kw = parts(args, modes)
    test_seeds = list(range(1000, 1000 + args.n_test))
    val_seeds = list(range(2000, 2010))
    venv = SubprocVecEnv([make(table, tr, cfg, args.seed * 100 + i, core, kw) for i in range(args.n_env)])
    model = MaskablePPO("MlpPolicy", venv, n_steps=1024, batch_size=1024, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                        learning_rate=3e-4, ent_coef=0.01, policy_kwargs=dict(net_arch=[256, 256]), seed=args.seed,
                        verbose=0, device="cpu")
    t0, log, done_steps = time.time(), [], 0
    while done_steps < args.steps:
        model.learn(int(args.eval_every), reset_num_timesteps=False)
        done_steps += int(args.eval_every)
        ev = evaluate_rl(model, table, te, cfg, val_seeds, core, kw)
        rec = dict(steps=done_steps, sec=time.time() - t0, val_return=float(np.mean([m["return"] for m in ev])),
                   val_util=float(np.mean([m["mean_util"] for m in ev])))
        log.append(rec)
        print(f"STEP {done_steps} val_return {rec['val_return']:+.2f} util {rec['val_util']:+.3f} {rec['sec']:.0f}s",
              flush=True)
    model.save(out / f"ppo_s{args.seed}")
    res = {"ppo": evaluate_rl(model, table, te, cfg, test_seeds, core, kw)}
    heur = [("fixed_" + m, P.fixed(j), V.fixed_mode(j)) for j, m in enumerate(modes)]
    for name in ("greedy", "paced_q4", "pacedB0.5", "pacedB1", "pacedB2", "stag_near", "stag_near_q3"):
        heur.append((name, P.from_name(name, modes), V.greedy_mode))
    for name, sp, mp in heur:
        res[name] = [V.run_episode_ev(core(table, te, cfg), sp, mp, s) for s in test_seeds]
    (out / f"eval_s{args.seed}.json").write_text(json.dumps(dict(cfg=cfg, env=args.env, rl_part=args.rl_part,
                                                                 slot_heur=args.slot_heur, modes=modes,
                                                                 test_seeds=test_seeds, val_seeds=val_seeds, log=log,
                                                                 results=res), indent=1))
    for k, v in res.items():
        print(f"{k:28s} return {np.mean([m['return'] for m in v]):+8.2f} util {np.mean([m['mean_util'] for m in v]):+.3f}"
              f" ok {np.mean([m['ok_rate'] for m in v]):.3f} delay {np.nanmean([m['mean_delay'] for m in v]):.2f}s"
              f" mission {np.mean([m['mission_s'] for m in v]):.0f}s")
    print("RUN_DONE ev", flush=True)


if __name__ == "__main__":
    main()
