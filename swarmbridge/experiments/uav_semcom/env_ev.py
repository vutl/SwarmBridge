"""A2 v2.2: two-timescale control on top of env v2.1 (env.py is not modified).

  slow timescale (every slot, dt = 1 s): UAV movement, channel, transmit power -- decided at the slot start;
  event timescale: the MODE of an image is decided at the moment its per-mode information becomes available
    (oracle / mean: at capture; predicted / oracle_probe: when its probe encode has finished on the UAV encoder),
    by a mode policy that sees the state at that instant.

The slot is simulated as one continuous-time event loop (transmissions, encoder completions, deadlines, info
arrivals). It pauses whenever an image needs a mode decision and resumes after `decide(m)`; nothing after the
decision instant is visible to the policy. Information levels:
  oracle        exact per-mode (bits, PSNR), available at capture, no cost      (upper-information reference)
  oracle_probe  exact values, but only after the probe encode has run          (isolates the acquisition cost)
  predicted     probe-based prediction, after the probe encode has run          (deployable)
  mean          train-mean table, at capture
oracle -> oracle_probe isolates the cost of acquiring the information (probe time); oracle_probe -> predicted
isolates the prediction error.

Encoder: FIFO, serial. Probes are queued at capture (one per image, back to back); a mode-encode is queued at its
decision instant, behind whatever the encoder already holds.
Usage: core = UavSemcomEnvEv(table, pool, cfg); core.reset(seed); then per slot:
  core.begin_slot([(move, ch, power) per UAV]); loop: ev = core.advance(); if ev[0] == "mode": core.decide(m) ...
"""
import math

import numpy as np

import env as E

INFO_LEVELS = ("oracle", "oracle_probe", "predicted", "mean")


class UavSemcomEnvEv(E.UavSemcomEnv):

    def __init__(self, table, img_pool, cfg=None, seed=0):
        c = dict(E.DEFAULT, **(cfg or {}))
        assert c["mode_info"] in INFO_LEVELS, c["mode_info"]
        if c["mode_info"] == "oracle_probe":
            assert table.fitted and table.probe_j is not None
        super().__init__(table, img_pool, dict(cfg or {}, mode_info=c["mode_info"] if c["mode_info"] != "oracle_probe"
                                               else "predicted"), seed)
        self.c["mode_info"] = c["mode_info"]

    # ---------- information ----------
    def probe_s(self):
        return self.tab.enc_s[self.tab.probe_j] if self.c["mode_info"] in ("predicted", "oracle_probe") else 0.0

    def mode_info(self, task):
        if self.c["mode_info"] == "oracle_probe":
            return self.tab.bits[task.img], self.tab.psnr[task.img]
        return super().mode_info(task)

    def awaiting(self, u):
        return [tk for tk in u.queue if tk.mode < 0 and tk.info_t <= self.t + 1e-9]

    # ---------- episode ----------
    def reset(self, seed=None):
        self._slot = None
        self._r = 0.0
        self.decisions = []        # (time, uav, wp, tile, mode) for checks
        return super().reset(seed)

    def pop_reward(self):
        r, self._r = self._r, 0.0
        return r

    def begin_slot(self, slot_actions):
        """slot_actions: per UAV (move, channel, power level). Mode entries are NOT accepted here."""
        c = self.c
        assert self._slot is None, "previous slot not finished"
        acts = [(int(a[0]), int(a[1]), int(a[2])) for a in slot_actions]
        assert len(acts) == c["K"]
        t0 = self.t
        tx = {k: (ch, c["p_levels"][pl]) for k, (mv, ch, pl) in enumerate(acts) if self.uavs[k].active and ch > 0}
        self._slot = dict(acts=acts, t0=t0, t1=t0 + c["dt"], n_active=sum(u.active for u in self.uavs), e=0.0,
                          gains=[self.slot_gain(k) for k in range(c["K"])], tx=tx, pending=None, n_ev=0)

    def _next_decision(self):
        """Earliest-captured image (over all UAVs) whose info is available now and that has no mode yet."""
        best = None
        for k, u in enumerate(self.uavs):
            for tk in u.queue:
                if tk.mode < 0 and tk.info_t <= self.t + 1e-9:
                    key = (tk.info_t, tk.t_cap, k, tk.tile)
                    if best is None or key < best[0]:
                        best = (key, k, tk)
                    break             # FIFO per UAV: only its first undecided image can be next
        return None if best is None else (best[1], best[2])

    def decide(self, m):
        S = self._slot
        k, tk = S["pending"]
        u = self.uavs[k]
        m = int(m)
        tk.mode, tk.bits, tk.rem_bits = m, self.tab.bits[tk.img, m], self.tab.bits[tk.img, m]
        tk.enc_start = max(self.t, u.enc_free)
        tk.ready = tk.enc_start + self.tab.enc_s[m]
        u.enc_free = tk.ready
        self.decisions.append((self.t, k, tk.wp, tk.tile, m))
        S["pending"] = None

    def advance(self):
        """Run the slot from the current instant until a mode decision is needed (returns ("mode", k, task)) or the
        slot is over (returns ("slot", done))."""
        S, c = self._slot, self.c
        assert S["pending"] is None, "decide() the pending image first"
        t1, tx, gains = S["t1"], S["tx"], S["gains"]
        while self.t < t1 - 1e-12:
            S["n_ev"] += 1
            assert S["n_ev"] < 20000, "event loop did not terminate"
            nd = self._next_decision()
            if nd is not None:
                S["pending"] = nd
                return ("mode",) + nd
            tau = self.t
            sending, nxt = {}, t1
            for k, u in enumerate(self.uavs):          # info arrivals are events for every UAV
                for tk in u.queue:
                    if tk.mode < 0 and tk.info_t > tau + 1e-12:
                        nxt = min(nxt, tk.info_t)
                        break
            for k in tx:
                u = self.uavs[k]
                while u.queue and u.queue[0].mode >= 0 and self.latest_tx_end(u.queue[0]) <= tau + 1e-12:
                    self._r += self._drop(u.queue.pop(0))
                if not u.queue or u.queue[0].mode < 0:
                    continue
                h = u.queue[0]
                if h.ready <= tau + 1e-12:
                    sending[k] = h
                else:
                    nxt = min(nxt, h.ready)
            rates = {}
            for k in sending:
                ch, p = tx[k]
                interf = sum(tx[j][1] * gains[j] for j in sending if j != k and tx[j][0] == ch) \
                    if c["interference"] else 0.0
                rates[k] = (c["B_total"] / c["M"] * math.log2(1 + p * gains[k] / (self.noise_w() + interf)), interf)
                h = sending[k]
                r = rates[k][0]
                nxt = min(nxt, (tau + h.rem_bits / r) if r > 0 else math.inf, self.latest_tx_end(h))
            nxt = max(nxt, tau)
            for k, h in sending.items():
                ch, p = tx[k]
                r, interf = rates[k]
                d = nxt - tau
                fin = r > 0 and tau + h.rem_bits / r <= nxt + 1e-12
                sent = h.rem_bits if fin else r * d
                if math.isnan(h.tx_start):
                    h.tx_start = tau
                h.rem_bits -= sent
                h.sent_bits += sent
                h.airtime += d
                if d > 0:
                    self.segments.append((k, ch, tau, nxt, interf, r))
                e = p * d
                S["e"] += e
                self.e_tx += e
                self.uavs[k].energy += e
                if fin:
                    h.rem_bits = 0.0
                    self._r += self._finish(h, nxt)
                    self.uavs[k].queue.pop(0)
                elif self.latest_tx_end(h) <= nxt + 1e-12:
                    self._r += self._drop(h)
                    self.uavs[k].queue.pop(0)
            if not sending and nxt >= t1 - 1e-12:
                self.t = t1
                break
            self.t = nxt
        return ("slot", self._end_slot())

    def _end_slot(self):
        """Steps (4)-(7) of env v2.1, unchanged: deadline expiry, movement / propulsion / captures, sortie end,
        energy and time costs, horizon."""
        S, c = self._slot, self.c
        W, L = c["n_wp"], self.L
        t1 = S["t1"]
        self.t = t1
        for u in self.uavs:
            keep = []
            for tk in u.queue:
                if self.latest_tx_end(tk) <= t1 + 1e-12:
                    self._r += self._drop(tk)
                else:
                    keep.append(tk)
            u.queue = keep
        bs2 = np.array(c["bs"][:2])
        dt = c["dt"]
        for k, (mv, ch, pl) in enumerate(S["acts"]):
            u = self.uavs[k]
            if not u.active:
                continue
            p_old = u.pos.copy()
            if c["mobility"] and 1 <= mv <= 8:
                a = 2 * math.pi * (mv - 1) / 8
                u.pos = u.pos + c["v_max"] * dt * np.array([math.cos(a), math.sin(a)])
            elif c["mobility"] and mv in (9, 10):
                tgt = (u.wps[u.wp_i] if u.wp_i < W else u.pos) if mv == 9 else bs2
                d = tgt - u.pos
                n = float(np.linalg.norm(d))
                if n > 1e-9:
                    u.pos = u.pos + d / n * min(n, c["v_max"] * dt)
            r = float(np.linalg.norm(u.pos - bs2))
            if r > c["area_radius"]:
                u.pos = bs2 + (u.pos - bs2) * c["area_radius"] / r
            v = float(np.linalg.norm(u.pos - p_old)) / dt
            e = E.propulsion_power(v) * dt
            u.energy += e
            S["e"] += e
            if u.wp_i < W and np.linalg.norm(u.pos - u.wps[u.wp_i]) <= c["capture_radius"] + c["v_max"] * dt / 2:
                w = u.wp_i
                for j in range(L):
                    tk = E.Task(uav=k, wp=w, tile=j, img=int(self.task_img[k, w, j]), t_cap=t1,
                                dmax=float(self.task_dmax[k, w, j]), qmin=float(self.task_qmin[k, w, j]), info_t=t1)
                    if self.probe_s() > 0:                # probe encode queued on the UAV encoder at capture
                        tk.info_t = max(t1, u.enc_free) + self.probe_s()
                        u.enc_free = tk.info_t
                    u.queue.append(tk)
                    self.tasks.append(tk)
                u.wp_i += 1
        for u in self.uavs:
            if not u.active:
                continue
            if u.energy >= c["battery_j"]:
                u.battery_out = True
                for tk in u.queue:
                    self._r += self._drop(tk)
                self._r += -1.0 * (W - u.wp_i) * L
                u.queue, u.active, u.t_done = [], False, t1
            elif u.wp_i >= W and not u.queue:
                u.active, u.t_done = False, t1
        if c["energy"]:
            self._r -= c["w_energy"] * S["e"] / 1000.0
        self._r -= c["w_time"] * dt * S["n_active"]
        self.slot += 1
        self._slot = None
        done = all(not u.active for u in self.uavs) or self.t >= c["T"] - 1e-9
        if done:
            for u in self.uavs:
                if u.active:
                    for tk in u.queue:
                        self._r += self._drop(tk)
                    self._r += -1.0 * (W - u.wp_i) * L
                    u.queue = []
        return done

    def step(self, actions):
        raise NotImplementedError("two-timescale env: use begin_slot / advance / decide (or run_episode_ev)")


# ---------------- heuristic mode policy at a decision event ----------------
def greedy_mode(env, k, task):
    """Greedy mode for ONE image at its decision instant: predicted utility through the UAV's encoder backlog and
    FIFO transmitter backlog at the current expected (channel-shared) rate, with the info known at this instant."""
    import policies as P
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
    return P.greedy_select(env, rate)(task, b_hat, q_hat, max(env.t, u.enc_free), tx_free)


def fixed_mode(m):
    def f(env, k, task):
        return m
    f.__name__ = f"fixed_{m}"
    return f


def run_episode_ev(env, slot_pol, mode_pol, seed):
    """slot_pol: any v2 heuristic (its mode entries are ignored); mode_pol(env, k, task) -> mode."""
    env.reset(seed)
    done = False
    while not done:
        env.begin_slot([a[:3] for a in slot_pol(env)])
        while True:
            ev = env.advance()
            if ev[0] == "mode":
                env.decide(mode_pol(env, ev[1], ev[2]))
            else:
                done = ev[1]
                break
    m = env.metrics()
    m["return"] = env.pop_reward()     # reward accumulated over the whole episode
    return m
