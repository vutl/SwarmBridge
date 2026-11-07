"""A5.4: train TimeMaskablePPO (semi-MDP discounting, gamma 0.99 / GAE lambda 0.95 per second) on one A5 regime.

Same network, PPO hyper-parameters, 6 parallel envs and 2M decision steps as the A4 corrected runs (train_ev.py
--time-discount); differences, all about data hygiene:
  * episodes use the A5 TRAIN pool (seeds = 100 * seed + env index, as A4);
  * every --eval-every steps a checkpoint is saved and evaluated on the VAL pool, seeds 3000..3019 (A4 monitored on
    test images); the checkpoint with the best validation return is recorded (selection rule fixed in advance:
    best mean val return, ties -> earlier), and so is the last one;
  * no test evaluation here (a5_eval.py evaluates the selected checkpoints on the test pools, paired with the
    heuristics).
  --rl-part joint | mode | movement; --heur NAME | T:FAMILY: for mode-only RL its slot part drives the slots
  (A5: the slot part of the regime's best tuned heuristic, T:best), for movement-only RL its mode part drives the modes.
Usage: python a5_train.py OUT --regime ref --rl-part joint --seed 0 [--heur T:reuse] [--steps 2e6]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

import a5_common as C
import a5_eval as AE
import env_survey as SV
import gym_env_ev as GE
import mode_heur as MH
import timeppo
from train_ev import Experience, evaluate_rl


def make(table, pool, cfg, seed, kw):
    def f():
        return GE.GymUavSemcomEv(table, pool, cfg, seed, core_cls=SV.UavSurveyEnv, **kw)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--regime", required=True)
    ap.add_argument("--rl-part", default="joint", choices=["joint", "movement", "mode"])
    ap.add_argument("--heur", default="T:best")
    ap.add_argument("--steps", type=float, default=2e6)
    ap.add_argument("--n-env", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=float, default=2e5)
    ap.add_argument("--resume", default="", help="run dir: continue its last checkpoint (extended-training "
                    "experiment); episodes use env seeds offset by +50 so the start of training is not replayed")
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = C.regime_cfg(args.regime)
    table, pools = C.load_table(False)
    heur = AE.tuned(args.regime, args.heur[2:]) if args.heur.startswith("T:") else args.heur
    sp, mp = MH.from_name(heur, C.MODES)
    kw = dict(joint={}, mode=dict(slot_pol=sp), movement=dict(mode_pol=mp))[args.rl_part]
    off = 50 if args.resume else 0
    venv = SubprocVecEnv([make(table, pools["train"], cfg, args.seed * 100 + off + i, kw) for i in range(args.n_env)])
    model = timeppo.TimeMaskablePPO("MlpPolicy", venv, n_steps=1024, batch_size=1024, n_epochs=10, gamma=0.99,
                                    gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.01,
                                    policy_kwargs=dict(net_arch=[256, 256]), seed=args.seed, verbose=0, device="cpu")
    meta = dict(regime=args.regime, cfg=cfg, rl_part=args.rl_part, heur=heur, seed=args.seed, steps=args.steps,
                val_seeds=C.PPO_VAL_SEEDS, val_pool="val", train_pool="train", time_discount=True)
    t0, log, done_steps, exp = time.time(), [], 0, Experience()
    if args.resume:
        prev = json.loads((Path(args.resume) / "train.json").read_text())
        done_steps = prev["last_steps"]
        log = prev["log"]
        exp.c.update(log[-1]["experience"])
        model = timeppo.TimeMaskablePPO.load(Path(args.resume) / f"ckpt_{done_steps}.zip", env=venv, device="cpu")
        meta["resumed_from"] = dict(dir=args.resume, steps=done_steps)
    while done_steps < args.steps:
        model.learn(int(args.eval_every), reset_num_timesteps=False, callback=exp)
        done_steps += int(args.eval_every)
        model.save(out / f"ckpt_{done_steps}")
        ev = evaluate_rl(model, table, pools["val"], cfg, C.PPO_VAL_SEEDS, SV.UavSurveyEnv, kw)
        rec = dict(steps=done_steps, sec=time.time() - t0, val_return=float(np.mean([m["return"] for m in ev])),
                   val_ok=float(np.mean([m["ok_rate"] for m in ev])), experience=dict(exp.c))
        log.append(rec)
        best = max(log, key=lambda r: (r["val_return"], -r["steps"]))
        (out / "train.json").write_text(json.dumps(dict(meta, log=log, best_steps=best["steps"],
                                                        last_steps=done_steps), indent=1))
        print(f"STEP {done_steps} val_return {rec['val_return']:+.2f} ok {rec['val_ok']:.3f} {rec['sec']:.0f}s "
              f"episodes {exp.c['episodes']} best@{best['steps']}", flush=True)
    print("RUN_DONE a5", flush=True)


if __name__ == "__main__":
    main()
