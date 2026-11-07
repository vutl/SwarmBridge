"""Extra mode heuristic for the two-timescale envs (added after A4 batch 1 showed PPO keeps flying and downgrades
modes per image instead of hovering): backlog-budgeted greedy.

budget_mode(tau): at the image's decision instant, for each mode predict (as env_ev.greedy_mode does: info known at
that instant, serial encoder, FIFO transmitter backlog at the expected channel-shared rate) when its transmission
would end, and value it with greedy's own criterion (-1 if predicted to miss the deadline, else the predicted
utility, which is < -1 when Qmin is violated). Among the modes that leave the UAV's transmit backlog at most `tau`
seconds from now, take the best-valued one (ties: lowest index, as greedy). If no mode fits the budget, take the
deadline-feasible mode predicted to meet Qmin that ends earliest, else the earliest-ending mode.
With tau = inf this is exactly env_ev.greedy_mode.
It is a mode policy; combine it with any v2 slot heuristic ("SLOT+budgetTAU", e.g. "greedy+budget1.5").
"""
import math

import env_ev as V
import policies as P


def budget_mode(tau):
    def f(env, k, task):
        u, tab = env.uavs[k], env.tab
        rate = max(P.expected_rate(env, k) * P.channel_share(env), 1.0)
        tx_free = env.t
        for tk in u.queue:
            if tk is task:
                break
            if tk.mode >= 0:
                b = tk.rem_bits if tk.ready <= env.t else env.mode_info(tk)[0][tk.mode] - tk.sent_bits
                tx_free = max(tx_free, tk.ready) + max(b, 0.0) / rate
        b_hat, q_hat = env.mode_info(task)
        enc_free = max(env.t, u.enc_free)
        best, best_u, early, early_end, early_q, early_q_end = None, -math.inf, 0, math.inf, None, math.inf
        for m in range(env.n_modes):
            end = max(tx_free, enc_free + tab.enc_s[m]) + b_hat[m] / rate
            if end < early_end:
                early, early_end = m, end
            feasible = end <= task.t_cap + task.dmax - tab.dec_s[m]
            if feasible and q_hat[m] >= task.qmin and end < early_q_end:
                early_q, early_q_end = m, end
            if end - env.t <= tau:
                v = env.utility(task, q_hat[m], end + tab.dec_s[m] - task.t_cap)[0] if feasible else -1.0
                if v > best_u:
                    best, best_u = m, v
        if best is not None:
            return best
        return early_q if early_q is not None else early
    f.__name__ = f"budget{tau:g}"
    return f


def reuse(slot_pol):
    """POST-HOC diagnostic (designed after the A4 channel analysis showed joint PPO puts all UAVs on air): the base slot
    policy's channel assignment (M channels to the least-slack UAVs), plus EVERY other active UAV with a queue also
    transmits, on the channel whose occupants have the least expected received power at the BS (least interference
    to it). Interference in the env is time-resolved (co-channel UAVs only interfere while both are actually sending),
    so a shared channel is used alone whenever the other UAV has nothing ready. Power unchanged (max).
    (A first version that only shared when the instantaneous predicted sum rate rose never shared at all: with both
    UAVs sending, sharing always lowers the sum rate at these SNRs; it was identical to the base heuristics.)"""
    def pol(env):
        acts = [list(a) for a in slot_pol(env)]
        c = env.c
        g = [env.expected_gain(u.pos) for u in env.uavs]
        for k, u in enumerate(env.uavs):
            if not (u.active and u.queue) or acts[k][1] > 0:
                continue
            load = {ch: sum(c["p_levels"][acts[j][2]] * g[j] for j in range(len(env.uavs))
                            if j != k and acts[j][1] == ch) for ch in range(1, c["M"] + 1)}
            acts[k][1] = min(load, key=lambda ch: (load[ch], ch))
        return [tuple(a) for a in acts]
    pol.__name__ = slot_pol.__name__ + "+reuse"
    return pol


def from_name(name, modes):
    """'SLOT+budgetTAU' -> (slot policy, mode policy); any other name -> (v2 heuristic, event-greedy / fixed);
    a '+reuse' suffix wraps the slot policy with reuse() (post-hoc diagnostic)."""
    if name.endswith("+reuse"):
        sp, mp = from_name(name[:-len("+reuse")], modes)
        return reuse(sp), mp
    if "+budget" in name:
        slot, tau = name.split("+budget")
        return P.from_name(slot, modes), budget_mode(float(tau))
    if name.startswith("fixed_"):
        j = modes.index(name[6:])
        return P.fixed(j), V.fixed_mode(j)
    return P.from_name(name, modes), V.greedy_mode


def family():
    return [f"{s}+budget{t:g}" for s in ("greedy", "pacedB1.5", "pacedB2", "pacedB3")
            for t in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)]


def reuse_family(modes):
    """Post-hoc diagnostic family: the base (tune_heur) and budget families, each with channel reuse."""
    from tune_heur import family as base
    return [n + "+reuse" for n in base(modes) + family() if not n.startswith("fixed_")]
