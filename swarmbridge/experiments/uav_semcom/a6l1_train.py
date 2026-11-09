"""A6.0-L1: train mode-only TimeMaskablePPO under the TASK reward (a6l1_task.TaskSurveyEnv).

Identical to a5_train.py (network, PPO hyper-parameters, 6 envs, train pool, env seeds 100 * seed + i, 2M steps,
evaluation every 2e5 steps on the val pool seeds 3000..3019, best-val checkpoint, ties -> earlier) except: the core env
(task reward) and the slot heuristic (slot part of T:best_task, a6l1_eval tune). Validation return = task return.
Usage: python a6l1_train.py OUT --regime ref --seed 0 [--steps 2e6]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

import a5_common as C
import a55_heur as H
import a6l1_eval as LE
import a6l1_task as TK
import gym_env_ev as GE
import timeppo
from train_ev import Experience, evaluate_rl


def make(table, pool, cfg, seed, kw):
    def f():
        return GE.GymUavSemcomEv(table, pool, cfg, seed, core_cls=TK.TaskSurveyEnv, **kw)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--regime", required=True)
    ap.add_argument("--heur", default="T6:best")
    ap.add_argument("--steps", type=float, default=2e6)
    ap.add_argument("--n-env", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=float, default=2e5)
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = C.regime_cfg(args.regime)
    table, pools = C.load_table(False)
    assert TK.task_table(table).covered(np.concatenate([pools["train"], pools["val"]]))
    heur = LE.resolve(args.regime, args.heur)
    sp, _ = H.from_name(heur, C.MODES)
    kw = dict(slot_pol=sp)
    venv = SubprocVecEnv([make(table, pools["train"], cfg, args.seed * 100 + i, kw) for i in range(args.n_env)])
    model = timeppo.TimeMaskablePPO("MlpPolicy", venv, n_steps=1024, batch_size=1024, n_epochs=10, gamma=0.99,
                                    gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.01,
                                    policy_kwargs=dict(net_arch=[256, 256]), seed=args.seed, verbose=0, device="cpu")
    meta = dict(regime=args.regime, cfg=cfg, rl_part="mode", heur=heur, seed=args.seed, steps=args.steps,
                val_seeds=C.PPO_VAL_SEEDS, val_pool="val", train_pool="train", time_discount=True,
                reward="task", beta=TK.BETA)
    t0, log, done_steps, exp = time.time(), [], 0, Experience()
    while done_steps < args.steps:
        model.learn(int(args.eval_every), reset_num_timesteps=False, callback=exp)
        done_steps += int(args.eval_every)
        model.save(out / f"ckpt_{done_steps}")
        ev = evaluate_rl(model, table, pools["val"], cfg, C.PPO_VAL_SEEDS, TK.TaskSurveyEnv, kw)
        rec = dict(steps=done_steps, sec=time.time() - t0, val_return=float(np.mean([m["return"] for m in ev])),
                   val_ok=float(np.mean([m["ok_rate"] for m in ev])), experience=dict(exp.c))
        log.append(rec)
        best = max(log, key=lambda r: (r["val_return"], -r["steps"]))
        (out / "train.json").write_text(json.dumps(dict(meta, log=log, best_steps=best["steps"],
                                                        last_steps=done_steps), indent=1))
        print(f"STEP {done_steps} val_return {rec['val_return']:+.2f} ok {rec['val_ok']:.3f} {rec['sec']:.0f}s "
              f"episodes {exp.c['episodes']} best@{best['steps']}", flush=True)
    print("RUN_DONE a6l1", flush=True)


if __name__ == "__main__":
    main()
