"""Heuristic baselines for the A2 environment (all causal: current state + expected link, no future knowledge).

Movement: follow the survey mission (next waypoint), optionally detour toward the BS when the head image cannot meet
its deadline from the current position. Channels: the M most urgent non-empty queues get the M channels (no
co-channel sharing). Power: max level. Mode: fixed, or chosen per image by predicted utility.
"""
import math

import numpy as np


def _dir_to(vec):
    if np.linalg.norm(vec) < 1e-6:
        return 0
    a = math.atan2(vec[1], vec[0]) % (2 * math.pi)
    return 1 + int(round(a / (math.pi / 4))) % 8


def mission_move(env, k):
    u = env.uavs[k]
    return 0 if u.wp_i >= len(u.wps) else 9   # macro: exact heading to the next waypoint


def assign_channels(env):
    """Give the M channels to the UAVs whose head task has the least slack."""
    cand = []
    for k, u in enumerate(env.uavs):
        if u.queue:
            h = u.queue[0]
            cand.append((h.dmax - (env.t - h.t_cap), k))
    ch = [0] * len(env.uavs)
    for i, (_, k) in enumerate(sorted(cand)[:env.c["M"]]):
        ch[k] = i + 1
    return ch


def expected_rate(env, k, p_level=-1):
    c = env.c
    g = env.expected_gain(env.uavs[k].pos)
    return c["B_total"] / c["M"] * math.log2(1 + c["p_levels"][p_level] * g / env.noise_w())


def predicted_util(env, k, task, m, table_bits=None, table_q=None, rate=None):
    rate = rate or expected_rate(env, k)
    bits = (table_bits if table_bits is not None else env.tab.bits[task.img])[m]
    q = (table_q if table_q is not None else env.tab.psnr[task.img])[m]
    wait = max(env.t - task.t_cap, 0)
    delay = wait + env.tab.enc_s[m] + bits / rate + env.tab.dec_s[m]
    return env.utility(task, q, delay)[0]


def fixed(m):
    def pol(env):
        ch = assign_channels(env)
        return [(mission_move(env, k), ch[k], len(env.c["p_levels"]) - 1, m) for k in range(len(env.uavs))]
    pol.__name__ = f"fixed_{m}"
    return pol


def greedy(per_image=True, detour=False):
    """Per head image, pick the mode with the best predicted utility at the current expected rate (rate shared by
    the queue: divided by the number of UAVs competing for the M channels)."""
    def pol(env):
        ch = assign_channels(env)
        n_busy = sum(1 for u in env.uavs if u.queue)
        share = min(1.0, env.c["M"] / max(n_busy, 1))
        acts = []
        mb = env.tab.bits[env.pool].mean(0)
        mq = env.tab.psnr[env.pool].mean(0)
        for k, u in enumerate(env.uavs):
            md, mv = 0, mission_move(env, k)
            if u.queue:
                h = u.queue[0]
                r = expected_rate(env, k) * share
                if per_image:
                    us = [predicted_util(env, k, h, m, rate=r) for m in range(env.n_modes)]
                else:
                    us = [predicted_util(env, k, h, m, mb, mq, rate=r) for m in range(env.n_modes)]
                md = int(np.argmax(us))
                if detour and max(us) < 0:
                    mv = 10   # macro: toward the BS
            acts.append((mv, ch[k], len(env.c["p_levels"]) - 1, md))
        return acts
    pol.__name__ = f"greedy{'_img' if per_image else '_mean'}{'_detour' if detour else ''}"
    return pol


def run_episode(env, pol, seed):
    env.reset(seed)
    done, tot = False, 0.0
    while not done:
        _, r, done = env.step(pol(env))
        tot += r
    m = env.metrics()
    m["return"] = tot
    return m
