"""How does a joint PPO policy use the channels differently from the heuristic (assign_channels: the M channels to the
UAVs with the least head-of-queue slack, one UAV per channel)? At every slot decision of PPO's own rollout, log PPO's
channel / power entries and what the heuristic would do in the same state. Analysis only (old test seeds 1000..).
Usage: python channel_a4.py MODEL.zip OUT_JSON --profiles ... --cpu-file ... --modes ... --cfg JSON [--n 20]
"""
import argparse
import json
from collections import Counter

import numpy as np
from sb3_contrib import MaskablePPO

import env as E
import env_survey as SV
import gym_env_ev as GE
import policies as P
import timeppo  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    cfg, modes = json.loads(args.cfg), args.modes.split(",")
    table = E.ModeTable(args.profiles, modes, args.cpu_file)
    _, te = table.split()
    model = MaskablePPO.load(args.model, device="cpu")
    c = Counter()
    for s in range(1000, 1000 + args.n):
        e = GE.GymUavSemcomEv(table, te, cfg, seed=s, seeds=[s], core_cls=SV.UavSurveyEnv)
        obs, _ = e.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True, action_masks=e.action_masks())
            if e.phase == "slot":
                core = e.core
                cand = [k for k, u in enumerate(core.uavs) if u.active and u.queue]
                if cand:
                    ppo_ch = [int(a[3 * k + 1]) for k in range(e.K)]
                    ppo_pw = [int(a[3 * k + 2]) for k in range(e.K)]
                    h_ch = P.assign_channels(core)
                    on = [k for k in cand if ppo_ch[k] > 0]
                    used = Counter(ppo_ch[k] for k in on)
                    c["slots_with_queue"] += 1
                    c[f"n_cand={len(cand)}"] += 1
                    c[f"n_cand={len(cand)}_ppo_on={len(on)}"] += 1
                    c["ppo_shared_channel"] += any(v > 1 for v in used.values())
                    c["ppo_idle_while_queue"] += len(on) < min(len(cand), core.c["M"])
                    c["same_set_as_heur"] += set(on) == {k for k in cand if h_ch[k] > 0}
                    c["ppo_power_max"] += sum(ppo_pw[k] == len(core.c["p_levels"]) - 1 for k in on)
                    c["ppo_tx_uav_slots"] += len(on)
                    if len(cand) > core.c["M"]:
                        # who is left out: by PPO vs heuristic (gain rank 0 = best expected gain, slack rank 0 = least)
                        g = {k: core.expected_gain(core.uavs[k].pos) for k in cand}
                        sl = {k: core.latest_tx_end(core.uavs[k].queue[0]) - core.t for k in cand}
                        grank = sorted(cand, key=lambda k: -g[k])
                        srank = sorted(cand, key=lambda k: sl[k])
                        for k in cand:
                            if ppo_ch[k] == 0:
                                c[f"contended_ppo_off_gainrank{grank.index(k)}"] += 1
                                c[f"contended_ppo_off_slackrank{srank.index(k)}"] += 1
                            if h_ch[k] == 0:
                                c[f"contended_heur_off_gainrank{grank.index(k)}"] += 1
                        c["contended"] += 1
            obs, _, done, _, _ = e.step(a)
    json.dump(dict(cfg=cfg, model=args.model, counts=dict(c)), open(args.out, "w"), indent=1)
    for k in sorted(c):
        print(f"{k:40s} {c[k]}")


if __name__ == "__main__":
    main()
