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


def queue_util(env, u, m, rate):
    """Predicted utility of the whole own queue if the head uses mode m and every later image then picks its own
    greedy-best mode, served in order (encode-then-send, as in the env) at the expected rate. Accounts for the waiting
    a slow / large head mode imposes on the images behind it, which the head-only greedy ignores."""
    tab = env.tab
    h = u.queue[0]
    cur = max(env.t, h.t_cap) + tab.enc_s[m] + tab.bits[h.img, m] / rate
    tot = env.utility(h, tab.psnr[h.img, m], cur + tab.dec_s[m] - h.t_cap)[0]
    for tk in u.queue[1:]:
        best, best_end = -9.0, None
        for m2 in range(env.n_modes):
            end = max(cur, tk.t_cap) + tab.enc_s[m2] + tab.bits[tk.img, m2] / rate
            v = env.utility(tk, tab.psnr[tk.img, m2], end + tab.dec_s[m2] - tk.t_cap)[0]
            if v > best:
                best, best_end = v, end
        tot += best
        cur = best_end
    return tot


def greedy(per_image=True, detour=False, queue_aware=False):
    """Per head image, pick the mode with the best predicted utility at the current expected rate (rate shared by
    the queue: divided by the number of UAVs competing for the M channels). queue_aware: maximise the predicted
    utility of the whole own queue instead of the head alone."""
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
                if queue_aware:
                    us = [queue_util(env, u, m, r) for m in range(env.n_modes)]
                elif per_image:
                    us = [predicted_util(env, k, h, m, rate=r) for m in range(env.n_modes)]
                else:
                    us = [predicted_util(env, k, h, m, mb, mq, rate=r) for m in range(env.n_modes)]
                md = int(np.argmax(us))
                if detour and max(us) < 0:
                    mv = 10   # macro: toward the BS
            acts.append((mv, ch[k], len(env.c["p_levels"]) - 1, md))
        return acts
    pol.__name__ = f"greedy{'_qa' if queue_aware else ('_img' if per_image else '_mean')}{'_detour' if detour else ''}"
    return pol


def paced(q_thr=0, per_image=True, queue_aware=False):
    """Greedy mode + queue-aware pacing: hover at the current position (do not fly on to the next waypoint) while the
    own queue holds more than q_thr images, so captures never outrun the link. Uses the slack in the mission time
    exactly like a learned policy can; costs nothing in this environment unless a mission-time penalty is set."""
    base = greedy(per_image, queue_aware=queue_aware)

    def pol(env):
        acts = base(env)
        out = []
        for k, (mv, ch, pl, md) in enumerate(acts):
            u = env.uavs[k]
            time_left = env.c["T"] - env.t
            wps_left = len(u.wps) - u.wp_i
            # never pace so long that the remaining waypoints become unreachable
            need = sum(np.linalg.norm(u.wps[i] - (u.wps[i - 1] if i > u.wp_i else u.pos))
                       for i in range(u.wp_i, len(u.wps))) / env.c["v_max"] + 2 * wps_left
            if len(u.queue) > q_thr and time_left > need + 5:
                mv = 0
            out.append((mv, ch, pl, md))
        return out
    pol.__name__ = f"paced_q{q_thr}{'' if per_image else '_mean'}{'_qa' if queue_aware else ''}"
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


def paced_time(tau, queue_aware=False):
    """Rate-aware pacing: hover while the predicted time to drain the own queue (encode + send each queued image in
    its greedy mode at the current expected, channel-shared rate) exceeds tau seconds. Unlike paced(q_thr), a UAV far
    from the BS (slow link) waits more than a nearby one."""
    base = greedy(True, queue_aware=queue_aware)

    def pol(env):
        acts = base(env)
        n_busy = sum(1 for u in env.uavs if u.queue)
        share = min(1.0, env.c["M"] / max(n_busy, 1))
        out = []
        for k, (mv, ch, pl, md) in enumerate(acts):
            u = env.uavs[k]
            if u.queue and u.wp_i < len(u.wps):
                r = expected_rate(env, k) * share
                tab = env.tab
                backlog = 0.0
                for i, tk in enumerate(u.queue):
                    m = md if i == 0 and tk.mode < 0 else (tk.mode if tk.mode >= 0 else
                                                          int(np.argmax([predicted_util(env, k, tk, j, rate=r)
                                                                         for j in range(env.n_modes)])))
                    if i == 0 and tk.mode >= 0:
                        backlog += max(tk.ready - env.t, 0) + tk.rem_bits / r
                    else:
                        backlog += tab.enc_s[m] + tab.bits[tk.img, m] / r
                time_left = env.c["T"] - env.t
                wps_left = len(u.wps) - u.wp_i
                need = sum(np.linalg.norm(u.wps[i] - (u.wps[i - 1] if i > u.wp_i else u.pos))
                           for i in range(u.wp_i, len(u.wps))) / env.c["v_max"] + 2 * wps_left
                if backlog > tau and time_left > need + 5:
                    mv = 0
            out.append((mv, ch, pl, md))
        return out
    pol.__name__ = f"pacedT{tau}{'_qa' if queue_aware else ''}"
    return pol


def staggered(q_thr=None, order="idx"):
    """Channel-matched survey scheduling (the rule PPO was found to learn): at most M UAVs (one per channel) run their
    survey at a time; the others hover at their start point and begin only when an active UAV has captured its last
    waypoint AND emptied its queue. Active UAVs fly the mission at full speed (q_thr=None) or with queue pacing.
    order: which UAVs start first - 'idx' (index), 'near' / 'far' (expected link gain at the start position)."""
    base = greedy(True) if q_thr is None else paced(q_thr)

    def pol(env):
        acts = base(env)
        K, M = len(env.uavs), env.c["M"]
        if order == "idx":
            rank = list(range(K))
        else:
            gains = [env.expected_gain(u.wps[0]) for u in env.uavs]
            rank = sorted(range(K), key=lambda k: -gains[k] if order == "near" else gains[k])
        started = [u.wp_i > 0 for u in env.uavs]
        busy = [(u.wp_i > 0 and (u.wp_i < len(u.wps) or len(u.queue) > 0)) for u in env.uavs]
        n_busy = sum(busy)
        allowed = set(k for k in range(K) if started[k])
        for k in rank:   # admit not-yet-started UAVs in rank order while fewer than M are busy
            if not started[k] and n_busy < M:
                allowed.add(k)
                n_busy += 1
        out = []
        for k, (mv, ch, pl, md) in enumerate(acts):
            u = env.uavs[k]
            if k not in allowed:
                time_left = env.c["T"] - env.t
                need = sum(np.linalg.norm(u.wps[i] - (u.wps[i - 1] if i > 0 else u.pos))
                           for i in range(len(u.wps))) / env.c["v_max"] + 2 * len(u.wps)
                if time_left > need + 10:
                    mv = 0
            out.append((mv, ch, pl, md))
        return out
    pol.__name__ = f"stag_{order}" + ("" if q_thr is None else f"_q{q_thr}")
    return pol
