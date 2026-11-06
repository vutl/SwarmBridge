"""Where does a trained PPO policy's advantage come from? Roll out PPO and a heuristic on the same seeds and log
behaviour: flight speed / macro-move usage, capture times (mission duration), channel / power / mode choices,
queue length and per-task delay.

Usage: python analyze_policy.py MODEL.zip OUT_JSON --profiles ... --cpu-file ... --modes ... [--cfg JSON] [--n 10]
"""
import argparse
import json
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

import env as E
import gym_env as G
import policies as P


def rollout(core, act_fn, seed):
    core.reset(seed)
    log = dict(moves=Counter(), ch=Counter(), pw=Counter(), modes=Counter(), qlen=[], cap_t=[], speed=[], dist_bs=[], hover_q=[])
    done = False
    while not done:
        acts = act_fn(core)
        for (mv, ch, pl, md), u in zip(acts, core.uavs):
            log["moves"][int(mv)] += 1
            log["ch"][int(ch)] += 1
            if ch > 0 and u.queue:
                log["pw"][int(pl)] += 1
            log["qlen"].append(len(u.queue))
        prev = [u.wp_i for u in core.uavs]
        pos0 = [u.pos.copy() for u in core.uavs]
        _, _, done = core.step(acts)
        for k, u in enumerate(core.uavs):
            log["speed"].append(float(np.linalg.norm(u.pos - pos0[k])))
            log["dist_bs"].append(float(np.linalg.norm(u.pos[:2] - np.array(core.c["bs"][:2]))))
            if u.wp_i < len(u.wps):
                log["hover_q"].append((len(u.queue), float(np.linalg.norm(u.pos - pos0[k])) < 0.5))
            if u.wp_i > prev[k]:
                log["cap_t"].append(core.t)
    m = core.metrics()
    fin = [tk for tk in core.finished if np.isfinite(tk.delay)]
    for tk in core.tasks:
        if tk.mode >= 0:
            log["modes"][core.tab.modes[tk.mode]] += 1
    return dict(metrics=m, moves=dict(log["moves"]), ch=dict(log["ch"]), pw=dict(log["pw"]),
                modes=dict(log["modes"]), mean_q=float(np.mean(log["qlen"])),
                last_capture=float(max(log["cap_t"]) if log["cap_t"] else -1),
                mean_speed=float(np.mean(log["speed"])), mean_dist_bs=float(np.mean(log["dist_bs"])),
                hover_by_q={q: float(np.mean([h for qq, h in log["hover_q"] if qq == q])) for q in range(6)
                            if any(qq == q for qq, _ in log["hover_q"])}, mean_delay=float(np.mean([t.delay for t in fin])) if fin else None,
                mean_psnr=float(np.mean([core.tab.psnr[t.img, t.mode] for t in fin])) if fin else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--heur", default="greedy", help="heuristic name, see policies.from_name")
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    table = E.ModeTable(args.profiles, args.modes.split(","), args.cpu_file)
    tr, te = table.split()
    model = PPO.load(args.model, device="cpu")
    g = G.GymUavSemcom(table, te, cfg, seed=0)

    def ppo_act(core):
        a, _ = model.predict(core.obs(), deterministic=True)
        return G.decode_action(core, a)

    heur = P.from_name(args.heur, args.modes.split(","))
    res = {"ppo": [], args.heur: []}
    for s in range(1000, 1000 + args.n):
        res["ppo"].append(rollout(E.UavSemcomEnv(table, te, cfg), ppo_act, s))
        res[args.heur].append(rollout(E.UavSemcomEnv(table, te, cfg), heur, s))
    summ = {}
    for k, rs in res.items():
        summ[k] = dict(util=np.mean([r["metrics"]["mean_util"] for r in rs]),
                       ok=np.mean([r["metrics"]["ok_rate"] for r in rs]),
                       energy_kj=np.mean([r["metrics"]["energy_kj"] for r in rs]),
                       last_capture_s=np.mean([r["last_capture"] for r in rs]),
                       mean_speed=np.mean([r["mean_speed"] for r in rs]),
                       mean_dist_bs=np.mean([r["mean_dist_bs"] for r in rs]),
                       hover_by_q={q: np.mean([r["hover_by_q"][q] for r in rs if q in r["hover_by_q"]]) for q in range(6)}, mean_queue=np.mean([r["mean_q"] for r in rs]),
                       mean_delay=np.mean([r["mean_delay"] for r in rs]), mean_psnr=np.mean([r["mean_psnr"] for r in rs]),
                       moves=sum((Counter(r["moves"]) for r in rs), Counter()),
                       ch=sum((Counter(r["ch"]) for r in rs), Counter()),
                       pw=sum((Counter(r["pw"]) for r in rs), Counter()),
                       modes=sum((Counter(r["modes"]) for r in rs), Counter()).most_common(8))
    json.dump(dict(summary=summ, episodes=res), open(args.out, "w"), indent=1, default=float)
    for k, v in summ.items():
        print(k, json.dumps(v, default=float))


if __name__ == "__main__":
    main()
