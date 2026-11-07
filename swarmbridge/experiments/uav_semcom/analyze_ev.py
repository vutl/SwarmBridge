"""What does a two-timescale PPO policy do differently from a heuristic? Roll both out on the same test seeds and log
mode choices, slot moves, PSNR margin over Qmin, and the return decomposition
    return = sum of utilities - w_time * (UAV-seconds active) - w_energy * energy[kJ].
Usage: python analyze_ev.py MODEL.zip OUT_JSON --env survey --heur NAME --profiles ... --cpu-file ... --modes ... --cfg JSON
"""
import argparse
import json
from collections import Counter

import numpy as np
from sb3_contrib import MaskablePPO

import env as E
import env_ev as V
import env_survey as SV
import gym_env_ev as GE
import policies as P

CORES = {"ev": V.UavSemcomEnvEv, "survey": SV.UavSurveyEnv}


def summarize(core, moves, ret):
    c = core.c
    deliv = [tk for tk in core.finished if not tk.dropped]
    m = core.metrics()
    n_total = len(core.uavs) * c["n_wp"] * core.L
    return dict(ret=ret, util_sum=m["mean_util"] * n_total, time_cost=c["w_time"] * m["mission_s"] * len(core.uavs),
                energy_cost=c["w_energy"] * m["energy_kj"], modes=Counter(core.tab.modes[mo] for (_, _, _, _, mo) in core.decisions),
                moves=Counter(int(x) for x in moves),
                margin=float(np.mean([core.tab.psnr[tk.img, tk.mode] - tk.qmin for tk in deliv])) if deliv else float("nan"),
                **m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--env", default="survey", choices=list(CORES))
    ap.add_argument("--heur", required=True)
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    cfg, modes = json.loads(args.cfg), args.modes.split(",")
    core_cls = CORES[args.env]
    table = E.ModeTable(args.profiles, modes, args.cpu_file)
    _, te = table.split()
    model = MaskablePPO.load(args.model, device="cpu")
    heur = P.from_name(args.heur, modes)
    res = {"ppo": [], args.heur: []}
    for s in range(1000, 1000 + args.n):
        e = GE.GymUavSemcomEv(table, te, cfg, seed=s, seeds=[s], core_cls=core_cls)
        obs, _ = e.reset()
        done, tot, moves = False, 0.0, []
        while not done:
            slot = e.phase == "slot"
            a, _ = model.predict(obs, deterministic=True, action_masks=e.action_masks())
            if slot:
                moves += [a[3 * k] for k, u in enumerate(e.core.uavs) if u.active]
            obs, r, done, _, _ = e.step(a)
            tot += r
        res["ppo"].append(summarize(e.core, moves, tot))
        core = core_cls(table, te, cfg)
        hm = []

        def sp(env):
            acts = heur(env)
            hm.extend(a[0] for a, u in zip(acts, env.uavs) if u.active)
            return acts
        m = V.run_episode_ev(core, sp, V.greedy_mode, s)
        res[args.heur].append(summarize(core, hm, m["return"]))
    summ = {}
    for k, rs in res.items():
        summ[k] = {f: float(np.mean([r[f] for r in rs])) for f in
                   ("ret", "util_sum", "time_cost", "energy_cost", "mean_util", "ok_rate", "drop_rate", "mean_psnr",
                    "margin", "mean_delay", "mission_s", "energy_kj")}
        summ[k]["modes"] = sum((r["modes"] for r in rs), Counter()).most_common()
        summ[k]["moves"] = sorted(sum((r["moves"] for r in rs), Counter()).items())
    json.dump(dict(cfg=cfg, heur=args.heur, seeds=list(range(1000, 1000 + args.n)), summary=summ), open(args.out, "w"),
              indent=1, default=float)
    for k, v in summ.items():
        print(k, json.dumps(v, default=float))


if __name__ == "__main__":
    main()
