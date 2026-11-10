"""A5.5 / A6.0-L1 heuristics: the A5 budget / paced-backlog / reuse heuristics (mode_heur.py, policies.py) rebuilt
with two switches (pre-registered in reports/archive/uav-semcom-a/uav-semcom-a5.5.md 0.1 and reports/archive/uav-semcom-a/uav-semcom-a6-l1.md 0.3):

  cal   ONLINE SELF-CALIBRATION of the three model quantities the A5 heuristics take from the model, from what the
        team has observed so far (strictly past data):
          rho   service rate: the A5 heuristics predict a UAV's drain rate as expected_rate x channel_share (Shannon,
                no interference). rho = (sum of bits actually sent + c) / (sum of model-predicted bits + c) over the
                slots in which a UAV had at least one slot's worth of predicted backlog (so the model rate was the
                limit), pooled over the team, per-slot decay lambda = 0.95, prior c = 5e6 bits (rho = 1).
          rho_b payload: (sum of true payloads + c_b) / (sum of predicted payloads + c_b) over the images whose
                encode has FINISHED (the true size is known only then), c_b = 2e6 bits.
          s     safety factor (AIMD on outcomes): each team image dropped or late (known at its expiry / delivery):
                s <- min(1.3 s, 3); each one delivered in time: s <- max(0.995 s, 1). Predicted transmit times are
                multiplied by s.
        Effective rate = rho * expected_rate * share / s; predicted payloads x rho_b.
  value 'psnr' (A5: the predicted utility of a mode uses its predicted PSNR) or 'task' (A6.0-L1: the PSNR term is
        replaced by xi_q * v_hat, the probe-predicted task value of the env -- a6l1_task.TaskSurveyEnv).

With cal off and value 'psnr' the actions are those of the A5 heuristic of the same name (test_a55.py checks this
step by step). Names: PREFIX:SLOT+budgetTAU[+reuse], PREFIX in {a5x (off, psnr), sc (cal, psnr), tv (off, task),
sctv (cal, task)}, SLOT in {greedy, pacedB<tau>}.
"""
import math

import mode_heur as MH
import policies as P

PREFIX = {"a5x": (False, "psnr"), "sc": (True, "psnr"), "tv": (False, "task"), "sctv": (True, "task")}
LAMBDA, C_RATE, C_BITS = 0.95, 5e6, 2e6
S_UP, S_DOWN, S_MAX = 1.3, 0.995, 3.0
SLOTS = ("greedy", "pacedB1.5", "pacedB2", "pacedB3")
TAUS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def family(prefix):
    return [f"{prefix}:{s}+budget{t:g}{r}" for r in ("", "+reuse") for s in SLOTS for t in TAUS]


# ---------------- online calibration state (per episode, stored on the env) ----------------
def state(env, slot_boundary=False):
    st = getattr(env, "_a55", None)
    if st is None or st["tasks"] is not env.tasks:           # new episode
        st = dict(tasks=env.tasks, num=0.0, den=0.0, nb=0.0, db=0.0, s=1.0, seg_i=0, fin_i=0,
                  t_prev=None, pred=None)
        env._a55 = st
    # payload: images whose encode has finished by now (true size known)
    for tk in env.tasks:
        if tk.mode >= 0 and tk.ready <= env.t + 1e-12 and not getattr(tk, "_a55_enc", False):
            tk._a55_enc = True
            st["nb"] += tk.bits
            st["db"] += env.mode_info(tk)[0][tk.mode]
    # outcomes (AIMD): finished images, in completion order
    fin = env.finished
    while st["fin_i"] < len(fin):
        tk = fin[st["fin_i"]]
        st["fin_i"] += 1
        if tk.dropped or tk.delay > tk.dmax:
            st["s"] = min(st["s"] * S_UP, S_MAX)
        else:
            st["s"] = max(st["s"] * S_DOWN, 1.0)
    if slot_boundary and st["t_prev"] != env.t:
        # service rate over the slot that just ended: bits sent per UAV (finished transmission segments)
        sent = [0.0] * len(env.uavs)
        segs = env.segments
        while st["seg_i"] < len(segs):
            k, ch, a, b, interf, r = segs[st["seg_i"]]
            st["seg_i"] += 1
            sent[k] += r * (b - a)
        if st["pred"] is not None:
            dt = env.t - st["t_prev"]
            st["num"] *= LAMBDA
            st["den"] *= LAMBDA
            for k, (rate, backlog) in enumerate(st["pred"]):
                if rate > 0 and backlog >= rate * dt:      # backlogged: the model rate was the limit
                    st["num"] += sent[k]
                    st["den"] += rate * dt
        share = P.channel_share(env)
        st["pred"] = [(P.expected_rate(env, k) * share if u.active else 0.0, _queued_bits(env, u))
                      for k, u in enumerate(env.uavs)]
        st["t_prev"] = env.t
    return st


def _queued_bits(env, u):
    b = 0.0
    for tk in u.queue:
        if tk.mode >= 0:
            b += tk.rem_bits if tk.ready <= env.t else env.mode_info(tk)[0][tk.mode] - tk.sent_bits
    return max(b, 0.0)


class Cal:
    """Multipliers used by a policy: identity when calibration is off."""

    def __init__(self, env, on):
        if on:
            st = state(env)
            self.rho = (st["num"] + C_RATE) / (st["den"] + C_RATE)
            self.rho_b = (st["nb"] + C_BITS) / (st["db"] + C_BITS)
            self.s = st["s"]
        else:
            self.rho, self.rho_b, self.s = 1.0, 1.0, 1.0

    def rate(self, env, k):
        r = P.expected_rate(env, k) * P.channel_share(env)
        if self.rho != 1.0 or self.s != 1.0:
            r = r * self.rho / self.s
        return max(r, 1.0)

    def bits(self, b):
        return b * self.rho_b if self.rho_b != 1.0 else b


# ---------------- value of a mode ----------------
def vterm(env, task, q_hat, value, known=True):
    """Per-mode quality term of the predicted utility: A5 = xi_q * clip((PSNR - Qmin) / q_span); task = xi_q * v_hat."""
    c = env.c
    if value == "psnr":
        return [c["xi_q"] * float(min(max((q - task.qmin) / c["q_span"], -1.0), 1.0)) for q in q_hat]
    v = env.task_value_hat(task.img) if known else env.task_value_mean()
    return [c["xi_q"] * float(x) for x in v]


def util(env, task, vt_m, q_m, delay):
    """Same arithmetic as env.utility with the quality term given."""
    c = env.c
    ok = q_m >= task.qmin and delay <= task.dmax
    u = vt_m + c["xi_d"] * max(task.dmax - delay, 0) / task.dmax
    return u - (0 if ok else 1)


def greedy_select(env, rate, value, cal):
    tab = env.tab

    def sel(tk, b_hat, q_hat, enc_free, tx_free, known=True):
        vt = vterm(env, tk, q_hat, value, known)
        best, best_u = 0, -math.inf
        for m in range(env.n_modes):
            ready = enc_free + tab.enc_s[m]
            end = max(tx_free, ready) + cal.bits(b_hat[m]) / rate
            if end > tk.t_cap + tk.dmax - tab.dec_s[m]:
                v = -1.0
            else:
                v = util(env, tk, vt[m], q_hat[m], end + tab.dec_s[m] - tk.t_cap)
            if v > best_u:
                best, best_u = m, v
        return best
    return sel


def backlog_s(env, k, rate, value, cal):
    """policies.backlog_s with calibrated payloads and the chosen value."""
    u, tab = env.uavs[k], env.tab
    tx_free = env.t
    for tk in u.queue:
        if tk.mode >= 0:
            b = tk.rem_bits if tk.ready <= env.t else cal.bits(env.mode_info(tk)[0][tk.mode]) - tk.sent_bits
            tx_free = max(tx_free, tk.ready) + max(b, 0.0) / rate
    enc_free = max(env.t, u.enc_free)
    sel = greedy_select(env, rate, value, cal)
    for tk in u.queue:
        if tk.mode < 0:
            known = tk.info_t <= env.t + 1e-9
            b_hat, q_hat = env.mode_info(tk) if known else (tab.mean_bits, tab.mean_psnr)
            start = max(enc_free, tk.info_t)
            m = sel(tk, b_hat, q_hat, start, tx_free, known)
            enc_free = start + tab.enc_s[m]
            tx_free = max(tx_free, enc_free) + cal.bits(b_hat[m]) / rate
    return tx_free - env.t


def slot_policy(slot, cal_on, value):
    """greedy (no pacing) or pacedB<tau>: mission movement, M most urgent queues get the channels, max power; pacing:
    hover while the predicted backlog exceeds tau and the survey stays feasible. Mode entries are unused (two-timescale
    envs take modes from the mode policy)."""
    tau = None if slot == "greedy" else float(slot[len("pacedB"):])

    def pol(env):
        if cal_on:
            state(env, slot_boundary=True)
        cal = Cal(env, cal_on)
        ch = P.assign_channels(env)
        acts = []
        for k, u in enumerate(env.uavs):
            mv = P.mission_move(env, k)
            if tau is not None and u.wp_i < len(u.wps) and u.queue and \
                    env.c["T"] - env.t > P._time_to_finish(env, u) + 5:
                if backlog_s(env, k, cal.rate(env, k), value, cal) > tau:
                    mv = 0
            acts.append((mv, ch[k], len(env.c["p_levels"]) - 1, (0,) * env.L))
        return acts
    pol.__name__ = slot
    return pol


def budget_mode(tau, cal_on, value):
    """mode_heur.budget_mode with calibrated rate / payloads and the chosen value."""
    def f(env, k, task):
        cal = Cal(env, cal_on)
        u, tab = env.uavs[k], env.tab
        rate = cal.rate(env, k)
        tx_free = env.t
        for tk in u.queue:
            if tk is task:
                break
            if tk.mode >= 0:
                b = tk.rem_bits if tk.ready <= env.t else cal.bits(env.mode_info(tk)[0][tk.mode]) - tk.sent_bits
                tx_free = max(tx_free, tk.ready) + max(b, 0.0) / rate
        b_hat, q_hat = env.mode_info(task)
        vt = vterm(env, task, q_hat, value)
        enc_free = max(env.t, u.enc_free)
        best, best_u, early, early_end, early_q, early_q_end = None, -math.inf, 0, math.inf, None, math.inf
        for m in range(env.n_modes):
            end = max(tx_free, enc_free + tab.enc_s[m]) + cal.bits(b_hat[m]) / rate
            if end < early_end:
                early, early_end = m, end
            feasible = end <= task.t_cap + task.dmax - tab.dec_s[m]
            if feasible and q_hat[m] >= task.qmin and end < early_q_end:
                early_q, early_q_end = m, end
            if end - env.t <= tau:
                v = util(env, task, vt[m], q_hat[m], end + tab.dec_s[m] - task.t_cap) if feasible else -1.0
                if v > best_u:
                    best, best_u = m, v
        if best is not None:
            return best
        return early_q if early_q is not None else early
    f.__name__ = f"budget{tau:g}"
    return f


def from_name(name, modes):
    """'PREFIX:SLOT+budgetTAU[+reuse]' -> (slot policy, mode policy); any other name -> mode_heur.from_name (A5)."""
    if ":" not in name:
        return MH.from_name(name, modes)
    prefix, rest = name.split(":", 1)
    cal_on, value = PREFIX[prefix]
    reuse = rest.endswith("+reuse")
    if reuse:
        rest = rest[:-len("+reuse")]
    slot, tau = rest.split("+budget")
    sp = slot_policy(slot, cal_on, value)
    if reuse:
        sp = MH.reuse(sp)
    return sp, budget_mode(float(tau), cal_on, value)
