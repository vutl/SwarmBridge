"""A-X2 (review #37): train a learned channel scheduler on one A5 regime, movement + power fixed to the T:best slot
rule (ax2_env.GymSchedEv).

  --variant c   C-only: the policy chooses the channel of every UAV at each slot step; modes by the T:best mode rule
  --variant cm  C+Mo:   the policy chooses channels at slot steps and the mode at mode steps
Everything else is the A5 setup (a5_train.py): TimeMaskablePPO (semi-MDP discounting, gamma 0.99 / GAE lambda 0.95
per second), MLP 256-256, n_steps 1024, batch 1024, 10 epochs, lr 3e-4, entropy 0.01, 6 parallel envs on the TRAIN
pool (env seeds 100 * seed + i), 2M decision steps, a checkpoint every 2e5 steps evaluated on the VAL pool seeds
3000..3019 with the same wrapper; selection rule fixed in advance: best mean val return (ties -> earlier).
Usage: python ax2_train.py OUT --regime K5 --variant c --seed 0 [--steps 2e6]
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
import ax2_env as X
import env_survey as SV
import mode_heur as MH
import timeppo
from train_ev import Experience


def env_kw(regime, variant):
    sp, mp = MH.from_name(AE.tuned(regime, "best"), C.MODES)
    return dict(sp_heur=sp, mp_heur=mp, learn_mode=(variant == "cm"))


def make(table, pool, cfg, seed, regime, variant):
    def f():
        return X.GymSchedEv(table, pool, cfg, seed, core_cls=SV.UavSurveyEnv, **env_kw(regime, variant))
    return f


def evaluate(model, table, pool, cfg, seeds, regime, variant):
    out = []
    for s in seeds:
        e = X.GymSchedEv(table, pool, cfg, seed=s, seeds=[s], core_cls=SV.UavSurveyEnv, **env_kw(regime, variant))
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
    ap.add_argument("--regime", required=True)
    ap.add_argument("--variant", required=True, choices=["c", "cm"])
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
    AE.G["table"], AE.G["pools"] = table, pools
    venv = SubprocVecEnv([make(table, pools["train"], cfg, args.seed * 100 + i, args.regime, args.variant)
                          for i in range(args.n_env)])
    model = timeppo.TimeMaskablePPO("MlpPolicy", venv, n_steps=1024, batch_size=1024, n_epochs=10, gamma=0.99,
                                    gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.01,
                                    policy_kwargs=dict(net_arch=[256, 256]), seed=args.seed, verbose=0, device="cpu")
    meta = dict(regime=args.regime, cfg=cfg, variant=args.variant, heur=AE.tuned(args.regime, "best"), seed=args.seed,
                steps=args.steps, val_seeds=C.PPO_VAL_SEEDS, val_pool="val", train_pool="train", time_discount=True)
    t0, log, done_steps, exp = time.time(), [], 0, Experience()
    while done_steps < args.steps:
        model.learn(int(args.eval_every), reset_num_timesteps=False, callback=exp)
        done_steps += int(args.eval_every)
        model.save(out / f"ckpt_{done_steps}")
        ev = evaluate(model, table, pools["val"], cfg, C.PPO_VAL_SEEDS, args.regime, args.variant)
        rec = dict(steps=done_steps, sec=time.time() - t0, val_return=float(np.mean([m["return"] for m in ev])),
                   val_ok=float(np.mean([m["ok_rate"] for m in ev])), experience=dict(exp.c))
        log.append(rec)
        best = max(log, key=lambda r: (r["val_return"], -r["steps"]))
        (out / "train.json").write_text(json.dumps(dict(meta, log=log, best_steps=best["steps"],
                                                        last_steps=done_steps), indent=1))
        print(f"STEP {done_steps} val_return {rec['val_return']:+.2f} ok {rec['val_ok']:.3f} {rec['sec']:.0f}s "
              f"episodes {exp.c['episodes']} best@{best['steps']}", flush=True)
    print("RUN_DONE ax2", flush=True)


if __name__ == "__main__":
    main()
