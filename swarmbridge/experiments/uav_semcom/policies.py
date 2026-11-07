"""Heuristic baselines for the A2 v2 environment (all causal: current state + expected link, no future knowledge).

Information budget: per-image x per-mode payload / quality is read ONLY through env.mode_info (same knowledge as
the learned controller's observation); everything else used here (time, positions, waypoints / remaining path,
queue slack, encoder backlog, expected SNR) is in env.obs().

Movement: follow the survey mission (next waypoint), optionally detour toward the BS. Channels: the M most urgent
non-empty queues get the M channels (no co-channel sharing). Power: max level. Modes: fixed, or chosen per image
(capture order) by predicted utility through the UAV's serial encoder and FIFO transmitter.
"""
import math

import numpy as np


def mission_move(env, k):
    u = env.uavs[k]
    return 0 if u.wp_i >= len(u.wps) else 9   # macro: exact heading to the next waypoint


def assign_channels(env):
    """Give the M channels to the active UAVs whose head image has the least slack."""
    cand = []
    for k, u in enumerate(env.uavs):
        if u.active and u.queue:
            cand.append((env.latest_tx_end(u.queue[0]) - env.t, k))
    ch = [0] * len(env.uavs)
    for i, (_, k) in enumerate(sorted(cand)[:env.c["M"]]):
        ch[k] = i + 1
    return ch


def expected_rate(env, k, p_level=-1):
    c = env.c
    g = env.expected_gain(env.uavs[k].pos)
    sig = c["p_levels"][p_level] * g
    return env.link_rate(sig, env.noise_w(), 0.0, sig / env.noise_w())


def channel_share(env):
    n_busy = sum(1 for u in env.uavs if u.active and u.queue)
    return min(1.0, env.c["M"] / max(n_busy, 1))


def plan_modes(env, k, rate, select):
    """Decide the modes of UAV k's images awaiting a decision, in capture order. For each image, predict when it is
    encoded (serial encoder after the images ahead) and when its transmission ends (FIFO transmitter at `rate` after
    the images ahead), then let `select(task, bits_hat, psnr_hat, enc_free, tx_free)` pick its mode."""
    u, tab = env.uavs[k], env.tab
    tx_free = env.t
    for tk in u.queue:
        if tk.mode >= 0:
            b = tk.rem_bits if tk.ready <= env.t else env.mode_info(tk)[0][tk.mode] - tk.sent_bits
            tx_free = max(tx_free, tk.ready) + max(b, 0.0) / rate
    enc_free = max(env.t, u.enc_free)
    modes = []
    for tk in env.awaiting(u):
        b_hat, q_hat = env.mode_info(tk)
        m = select(tk, b_hat, q_hat, enc_free, tx_free)
        enc_free = enc_free + tab.enc_s[m]
        tx_free = max(tx_free, enc_free) + b_hat[m] / rate
        modes.append(int(m))
    return tuple(modes + [0] * (env.L - len(modes)))


def greedy_select(env, rate):
    tab = env.tab

    def sel(tk, b_hat, q_hat, enc_free, tx_free):
        best, best_u = 0, -math.inf
        for m in range(env.n_modes):
            ready = enc_free + tab.enc_s[m]
            end = max(tx_free, ready) + b_hat[m] / rate
            if end > tk.t_cap + tk.dmax - tab.dec_s[m]:     # predicted to miss the deadline -> discarded
                v = -1.0
            else:
                v = env.utility(tk, q_hat[m], end + tab.dec_s[m] - tk.t_cap)[0]
            if v > best_u:
                best, best_u = m, v
        return best
    return sel


def fixed(m):
    def pol(env):
        ch = assign_channels(env)
        return [(mission_move(env, k), ch[k], len(env.c["p_levels"]) - 1, (m,) * env.L) for k in range(len(env.uavs))]
    pol.__name__ = f"fixed_{m}"
    return pol


def greedy(detour=False):
    """Per image (capture order), the mode with the best predicted utility at the current expected rate, shared by
    the UAVs competing for the M channels."""
    def pol(env):
        ch = assign_channels(env)
        share = channel_share(env)
        acts = []
        for k, u in enumerate(env.uavs):
            mv = mission_move(env, k)
            modes = (0,) * env.L
            if u.active and u.queue:
                r = max(expected_rate(env, k) * share, 1.0)
                modes = plan_modes(env, k, r, greedy_select(env, r))
                if detour and any(env.latest_tx_end(tk) - env.t < 1.0 for tk in u.queue):
                    mv = 10   # macro: toward the BS
            acts.append((mv, ch[k], len(env.c["p_levels"]) - 1, modes))
        return acts
    pol.__name__ = f"greedy{'_detour' if detour else ''}"
    return pol


def _time_to_finish(env, u):
    """Flight time for the remaining survey at v_max plus a 2 s margin per waypoint (observable: remaining path)."""
    return env.remaining_path(u) / env.c["v_max"] + 2 * (len(u.wps) - u.wp_i)


def paced(q_thr=0):
    """Greedy modes + queue-aware pacing: hover while the own queue holds more than q_thr images, as long as the
    remaining survey stays feasible."""
    base = greedy()

    def pol(env):
        out = []
        for k, (mv, ch, pl, md) in enumerate(base(env)):
            u = env.uavs[k]
            if u.wp_i < len(u.wps) and len(u.queue) > q_thr and env.c["T"] - env.t > _time_to_finish(env, u) + 5:
                mv = 0
            out.append((mv, ch, pl, md))
        return out
    pol.__name__ = f"paced_q{q_thr}"
    return pol


def staggered(q_thr=None, order="idx"):
    """At most M UAVs (one per channel) run their survey at a time; the others hover at their start point until an
    active UAV has captured its last waypoint and emptied its queue. order: 'idx' / 'near' / 'far' (expected link
    gain at the first waypoint). Optional queue pacing for the surveying UAVs."""
    base = greedy() if q_thr is None else paced(q_thr)

    def pol(env):
        acts = base(env)
        K, M = len(env.uavs), env.c["M"]
        if order == "idx":
            rank = list(range(K))
        else:
            gains = [env.expected_gain(u.wps[0]) for u in env.uavs]
            rank = sorted(range(K), key=lambda k: -gains[k] if order == "near" else gains[k])
        started = [u.wp_i > 0 for u in env.uavs]
        n_busy = sum(1 for u in env.uavs if u.wp_i > 0 and u.active)
        allowed = {k for k in range(K) if started[k]}
        for k in rank:
            if not started[k] and n_busy < M:
                allowed.add(k)
                n_busy += 1
        out = []
        for k, (mv, ch, pl, md) in enumerate(acts):
            u = env.uavs[k]
            if k not in allowed and env.c["T"] - env.t > _time_to_finish(env, u) + 10:
                mv = 0
            out.append((mv, ch, pl, md))
        return out
    pol.__name__ = f"stag_{order}" + ("" if q_thr is None else f"_q{q_thr}")
    return pol


def backlog_s(env, k, rate):
    """Predicted time (s) until UAV k's whole queue is sent at `rate`: images already decided (exact payload once
    encoded, else as known), then the undecided ones in their greedy mode (info as known; images whose probe is not
    done yet are costed with the train-mean table) through the serial encoder."""
    u, tab = env.uavs[k], env.tab
    tx_free = env.t
    for tk in u.queue:
        if tk.mode >= 0:
            b = tk.rem_bits if tk.ready <= env.t else env.mode_info(tk)[0][tk.mode] - tk.sent_bits
            tx_free = max(tx_free, tk.ready) + max(b, 0.0) / rate
    enc_free = max(env.t, u.enc_free)
    sel = greedy_select(env, rate)
    for tk in u.queue:
        if tk.mode < 0:
            known = tk.info_t <= env.t + 1e-9
            b_hat, q_hat = env.mode_info(tk) if known else (tab.mean_bits, tab.mean_psnr)
            start = max(enc_free, tk.info_t)
            m = sel(tk, b_hat, q_hat, start, tx_free)
            enc_free = start + tab.enc_s[m]
            tx_free = max(tx_free, enc_free) + b_hat[m] / rate
    return tx_free - env.t


def paced_backlog(tau):
    """Finer pacing than a queue-length threshold: greedy modes, and hover (do not fly on to the next waypoint)
    while the predicted time to drain the own queue exceeds tau seconds, as long as the survey stays feasible."""
    base = greedy()

    def pol(env):
        out = []
        share = channel_share(env)
        for k, (mv, ch, pl, md) in enumerate(base(env)):
            u = env.uavs[k]
            if u.wp_i < len(u.wps) and u.queue and env.c["T"] - env.t > _time_to_finish(env, u) + 5:
                r = max(expected_rate(env, k) * share, 1.0)
                if backlog_s(env, k, r) > tau:
                    mv = 0
            out.append((mv, ch, pl, md))
        return out
    pol.__name__ = f"pacedB{tau:g}"
    return pol


def from_name(name, modes=None):
    """Rebuild a heuristic from its __name__ (fixed_<mode name> needs the mode list)."""
    if name.startswith("fixed_"):
        return fixed(modes.index(name[len("fixed_"):]))
    if name.startswith("pacedB"):
        return paced_backlog(float(name[len("pacedB"):]))
    if name.startswith("stag_"):
        parts = name.split("_")
        return staggered(int(parts[2][1:]) if len(parts) > 2 else None, parts[1])
    if name.startswith("paced_q"):
        return paced(int(name[len("paced_q"):]))
    return greedy(detour=name.endswith("_detour"))


def run_episode(env, pol, seed):
    env.reset(seed)
    done, tot = False, 0.0
    while not done:
        _, r, done = env.step(pol(env))
        tot += r
    m = env.metrics()
    m["return"] = tot
    return m
