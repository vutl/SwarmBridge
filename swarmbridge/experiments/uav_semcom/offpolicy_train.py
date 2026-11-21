"""Paper A E7: train QR-DQN or Discrete SAC on the mode-only task of PPO-LA (A-X9), same environment and protocol:
regime K5_la + scheduler-aware LA, persistent interference, causal LA observation features (obs_la), slots by the
frozen JA scheduler on the pacedB3 movement rule, TRAIN pool (env seeds 100 * seed + env index), 2M RL decisions,
checkpoint + validation (VAL pool, seeds C.PPO_VAL_SEEDS) every 200k decisions, selected checkpoint = best mean
validation return (ties -> earlier), exactly as a5_train.py.
Usage: python offpolicy_train.py OUT --algo qrdqn|dsac --seed 0 [--lr 1e-4] [--ent-ratio 0.6]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

import a5_common as C
import mode_heur as MH
import offpolicy_rl as OP
import env_survey as SV
from a5_train import make
from train_ev import evaluate_rl

BASE_EXTRA = dict(la_mode="sched", la_interf="persistent", obs_la=True)


def setup():
    f = json.loads((C.REPO / "runs" / "uav_semcom" / "ax9" / "sched.json").read_text())["frozen"]
    cfg = C.regime_cfg("K5_la", **BASE_EXTRA)
    table, pools = C.load_table(False)
    os.environ["SB_JA_LOG"] = "0"
    import ja_sched as JA
    JA.LOG = False
    sp = MH.from_name(f["move"], C.MODES)[0]
    sp = JA.from_name(f["member"], sp)
    return cfg, table, pools, dict(slot_pol=sp), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--algo", required=True, choices=("qrdqn", "dsac"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=float, default=2e6)
    ap.add_argument("--n-env", type=int, default=2)
    ap.add_argument("--eval-every", type=float, default=2e5)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ent-ratio", type=float, default=0.6)
    ap.add_argument("--buffer", type=int, default=250_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--learning-starts", type=int, default=20_000)
    ap.add_argument("--train-every", type=int, default=4, help="RL decisions (transitions) per gradient step")
    ap.add_argument("--eps-frac", type=float, default=0.1, help="QR-DQN: epsilon 1 -> 0.05 over this share of steps")
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, table, pools, kw, frozen = setup()
    venv = SubprocVecEnv([make(table, pools["train"], cfg, a.seed * 100 + i, kw) for i in range(a.n_env)])
    obs_dim = venv.observation_space.shape[0]
    nvec = list(venv.action_space.nvec)
    n_act = nvec[-1]
    if a.algo == "qrdqn":
        agent = OP.QRDQN(obs_dim, n_act, lr=a.lr or 1e-4, seed=a.seed)
    else:
        agent = OP.DiscreteSAC(obs_dim, n_act, lr=a.lr or 3e-4, ent_ratio=a.ent_ratio, seed=a.seed)
    rep = OP.Replay(a.buffer, obs_dim, seed=a.seed)
    rng = np.random.default_rng(1000 + a.seed)
    meta = dict(algo=a.algo, hp=agent.hp, seed=a.seed, steps=a.steps, n_env=a.n_env, buffer=a.buffer, batch=a.batch,
                learning_starts=a.learning_starts, train_every=a.train_every, eps_frac=a.eps_frac, cfg=cfg,
                slot_move=frozen["move"], slot_ja=frozen["member"], val_seeds=C.PPO_VAL_SEEDS, val_pool="val",
                train_pool="train", time_discount=True, obs_dim=obs_dim, nvec=[int(x) for x in nvec])
    obs = venv.reset()
    t = upd = 0
    next_eval, log, t0, episodes, losses = int(a.eval_every), [], time.time(), 0, []
    while t < a.steps:
        masks = np.stack(venv.env_method("action_masks"))[:, -n_act:]
        assert masks.all(), "mode-phase masks are expected to allow every mode"
        if t < a.learning_starts:
            act = rng.integers(0, n_act, size=a.n_env)
        else:
            eps = max(0.05, 1.0 - 0.95 * t / (a.eps_frac * a.steps)) if a.algo == "qrdqn" else 0.0
            act = agent.act(obs, rng, eps=eps, mask=masks)
        full = np.zeros((a.n_env, len(nvec)), dtype=np.int64)
        full[:, -1] = act
        nobs, rew, done, infos = venv.step(full)
        nreal = nobs.copy()
        for j, d in enumerate(done):
            if d:
                nreal[j] = infos[j]["terminal_observation"]
                episodes += 1
        rep.add(obs, act, rew, nreal, done.astype(np.float32), [inf["dt"] for inf in infos])
        obs = nobs
        t += a.n_env
        while t >= a.learning_starts and upd * a.train_every < t - a.learning_starts:
            losses.append(agent.update(rep.sample(a.batch)))
            upd += 1
        if t >= next_eval:
            steps = next_eval
            next_eval += int(a.eval_every)
            torch.save(agent.state(), out / f"ckpt_{steps}.pt")
            ev = evaluate_rl(OP.ModePolicy(agent, nvec), table, pools["val"], cfg, C.PPO_VAL_SEEDS, SV.UavSurveyEnv, kw)
            rec = dict(steps=steps, sec=time.time() - t0, val_return=float(np.mean([m["return"] for m in ev])),
                       val_ok=float(np.mean([m["ok_rate"] for m in ev])), episodes=episodes, updates=upd,
                       loss=float(np.mean(losses[-1000:])) if losses else None,
                       alpha=float(agent.log_alpha.exp()) if a.algo == "dsac" else None)
            log.append(rec)
            best = max(log, key=lambda r: (r["val_return"], -r["steps"]))
            (out / "train.json").write_text(json.dumps(dict(meta, log=log, best_steps=best["steps"], last_steps=steps),
                                                       indent=1))
            print(f"STEP {steps} val_return {rec['val_return']:+.2f} ok {rec['val_ok']:.3f} {rec['sec']:.0f}s "
                  f"episodes {episodes} updates {upd} loss {rec['loss']} best@{best['steps']}", flush=True)
    venv.close()
    print("RUN_DONE e7", flush=True)


if __name__ == "__main__":
    main()
