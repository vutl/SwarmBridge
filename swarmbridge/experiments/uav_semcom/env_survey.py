"""A4: photogrammetry survey workload on the two-timescale env (env_ev.py). Topology, channel, encoder / transmitter,
deadlines, utility, reward and constraints are those of v2.1 / v2.2; only the WORKLOAD changes:

  * each UAV flies a lawn-mower survey of `survey_lines` parallel lines of length `line_len`, spaced `line_spacing`
    apart (side overlap), and takes ONE native-resolution frame at every capture point, spaced `frame_spacing` along
    a line (forward overlap). Defaults from a Phantom-4-Pro-class camera at H = 100 m (84 deg diagonal FOV, 3:2;
    footprint ~150 x 100 m, short side along track) with the Pix4D-recommended 75 % forward / 60 % side overlap:
    frame_spacing 25 m, line_spacing 60 m;
  * survey speed = v_max (10 or 20 m/s in A4); the frames come from the native-resolution A4 mode table;
  * move 9 = follow the survey path at v_max for the whole slot, passing (and capturing at) every capture point it
    reaches -- captures happen at their exact instants inside the slot, so probes and mode decisions can happen
    mid-slot; 0 = hover (wait), 1-8 = compass at v_max (no capture), 10 = toward the BS (no capture).
The channel uses the slot-start position (block model, as before).
"""
import math

import numpy as np

import env as E
import env_ev as V

SURVEY = dict(v_max=10.0, tiles_per_wp=1, survey_lines=4, line_len=300.0, frame_spacing=25.0, line_spacing=60.0,
              T=600, sector_radius=(1000.0, 2500.0))


class UavSurveyEnv(V.UavSemcomEnvEv):

    def __init__(self, table, img_pool, cfg=None, seed=0):
        super().__init__(table, img_pool, dict(SURVEY, **(cfg or {})), seed)

    def reset(self, seed=None):
        c = self.c
        ss = np.random.SeedSequence(seed)
        r_layout, r_task, r_fade = [np.random.default_rng(s) for s in ss.spawn(3)]
        K = c["K"]
        self.uavs = []
        n_per_line = int(round(c["line_len"] / c["frame_spacing"])) + 1
        for k in range(K):
            ang = 2 * math.pi * (k + r_layout.random() * 0.5) / K
            r = r_layout.uniform(*c["sector_radius"])
            centre = np.array([r * math.cos(ang), r * math.sin(ang)])
            tang = np.array([-math.sin(ang), math.cos(ang)])
            rad = np.array([math.cos(ang), math.sin(ang)])
            pts = []
            for i in range(c["survey_lines"]):
                off = (i - (c["survey_lines"] - 1) / 2) * c["line_spacing"] * rad
                xs = [(j - (n_per_line - 1) / 2) * c["frame_spacing"] for j in range(n_per_line)]
                if i % 2:
                    xs = xs[::-1]
                pts += [centre + off + x * tang for x in xs]
            pts = np.array(pts)
            start = pts[0] - 2 * c["frame_spacing"] * (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
            self.uavs.append(E.UAV(pos=start.copy(), wps=pts))
        W = len(self.uavs[0].wps)
        c["n_wp"] = W                      # metrics / horizon accounting use n_wp: one frame per capture point
        self.task_img = r_task.choice(self.pool, size=(K, W, 1))
        self.task_dmax = r_task.uniform(*c["deadline"], size=(K, W, 1))
        self.task_qmin = r_task.uniform(*c["qmin"], size=(K, W, 1))
        n_slots = int(math.ceil(c["T"] / c["dt"]))
        self.f_u = r_fade.random((n_slots, K))
        self.f_n = r_fade.normal(size=(n_slots, K, 2))
        self.f_e = r_fade.exponential(size=(n_slots, K))
        self.t, self.slot = 0.0, 0
        self.tasks, self.finished, self.segments = [], [], []
        self.e_tx = 0.0
        self._slot, self._r, self.decisions = None, 0.0, []
        return self.obs()

    # ---------- movement plan of a slot (decided at the slot start) ----------
    def _plan(self, k, mv):
        """End position, travelled distance and the capture instants (time, waypoint index) of UAV k in this slot."""
        c, u = self.c, self.uavs[k]
        dt, v = c["dt"], c["v_max"]
        pos, caps, dist = u.pos.copy(), [], 0.0
        if not u.active or not c["mobility"]:
            return pos, 0.0, caps
        if 1 <= mv <= 8:
            a = 2 * math.pi * (mv - 1) / 8
            return pos + v * dt * np.array([math.cos(a), math.sin(a)]), v * dt, caps
        if mv == 10:
            d = np.array(c["bs"][:2]) - pos
            n = float(np.linalg.norm(d))
            step = min(n, v * dt)
            return (pos + d / n * step if n > 1e-9 else pos), step, caps
        if mv == 9:
            budget, i = v * dt, u.wp_i
            while budget > 1e-9 and i < len(u.wps):
                d = u.wps[i] - pos
                n = float(np.linalg.norm(d))
                if n <= budget + 1e-9:
                    pos, budget, dist = u.wps[i].copy(), budget - n, dist + n
                    caps.append((self.t + dist / v, i))
                    i += 1
                else:
                    pos, dist, budget = pos + d / n * budget, dist + budget, 0.0
        return pos, dist, caps

    def begin_slot(self, slot_actions):
        super().begin_slot(slot_actions)
        S = self._slot
        S["plan"] = [self._plan(k, mv) for k, (mv, ch, pl) in enumerate(S["acts"])]
        S["caps"] = sorted((t, k, w) for k, (_, _, cl) in enumerate(S["plan"]) for (t, w) in cl)

    def _do_captures(self, upto):
        S = self._slot
        while S["caps"] and S["caps"][0][0] <= upto + 1e-12:
            t, k, w = S["caps"].pop(0)
            t = min(t, upto)                  # captured within float tolerance of the current instant
            u = self.uavs[k]
            tk = E.Task(uav=k, wp=w, tile=0, img=int(self.task_img[k, w, 0]), t_cap=t,
                        dmax=float(self.task_dmax[k, w, 0]), qmin=float(self.task_qmin[k, w, 0]), info_t=t)
            if self.probe_s() > 0:
                tk.info_t = max(t, u.enc_free) + self.probe_s()
                u.enc_free = tk.info_t
            u.queue.append(tk)
            self.tasks.append(tk)
            u.wp_i = w + 1

    def advance(self):
        """As env_ev.advance, with the slot's capture instants as extra events."""
        S = self._slot
        t1 = S["t1"]
        while True:
            self._do_captures(self.t)
            nxt_cap = S["caps"][0][0] if S["caps"] else math.inf
            if nxt_cap >= t1 - 1e-12:
                # no capture left inside the slot (except possibly exactly at t1): run the loop to the slot end
                ev = self._advance_until(t1)
                if ev is not None:
                    return ev
                self._do_captures(t1)
                return ("slot", self._end_slot())
            ev = self._advance_until(nxt_cap)
            if ev is not None:
                return ev

    def _advance_until(self, horizon):
        """Run the env_ev event loop up to `horizon` (<= slot end); returns a ("mode", k, task) event or None."""
        S = self._slot
        t1_saved = S["t1"]
        S["t1"] = horizon
        try:
            while self.t < horizon - 1e-12:
                nd = self._next_decision()
                if nd is not None:
                    S["pending"] = nd
                    return ("mode",) + nd
                self._loop_once(horizon)
            nd = self._next_decision()
            if nd is not None:
                S["pending"] = nd
                return ("mode",) + nd
            return None
        finally:
            S["t1"] = t1_saved

    def _loop_once(self, horizon):
        """One iteration of the env_ev transmission/event loop, bounded by `horizon`."""
        S, c = self._slot, self.c
        tx, gains = S["tx"], S["gains"]
        S["n_ev"] += 1
        assert S["n_ev"] < 50000, "event loop did not terminate"
        tau = self.t
        sending, nxt = {}, horizon
        for k, u in enumerate(self.uavs):
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
        self.t = nxt if (sending or nxt < horizon) else horizon

    def _end_slot(self):
        """Deadline expiry, movement (planned at the slot start) + propulsion energy, sortie end, costs, horizon."""
        S, c = self._slot, self.c
        t1, dt = S["t1"], c["dt"]
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
        for k, u in enumerate(self.uavs):
            if not u.active:
                continue
            pos, dist, _ = S["plan"][k]
            r = float(np.linalg.norm(pos - bs2))
            if r > c["area_radius"]:
                pos = bs2 + (pos - bs2) * c["area_radius"] / r
            u.pos = pos
            e = E.propulsion_power(dist / dt) * dt
            u.energy += e
            S["e"] += e
        W = len(self.uavs[0].wps)
        for u in self.uavs:
            if not u.active:
                continue
            if u.energy >= c["battery_j"]:
                u.battery_out = True
                for tk in u.queue:
                    self._r += self._drop(tk)
                self._r += -1.0 * (W - u.wp_i)
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
                    self._r += -1.0 * (W - u.wp_i)
                    u.queue = []
        return done

    def metrics(self):
        m = super().metrics()
        m["frames"] = len(self.uavs) * len(self.uavs[0].wps)
        return m
