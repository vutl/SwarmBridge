"""[FROZEN COPY of env v2 at commit d812c7e7 - reference for v2.1 regression tests only; do not edit]
A2 v2: multi-UAV image delivery environment (UAV camera -> UAV encoder -> air-to-ground link -> ground BS).

The system model and MDP (state, observation, action, transition order, reward, constraints, parameter sources) are
written out in reports/archive/uav-semcom-a/uav-semcom-a.md section 4, with pointers back to the functions here.

v2 changes vs v1 (v1 = commit 1c85cfda; v1 results in report sections 3 / 3.1 are invalid, see 3.2):
  * a slot is no longer capped at one image per UAV: the transmitter serves its FIFO queue continuously inside the
    slot, and the encoder runs in parallel with the transmitter (serial encoder, serial transmitter = tandem queue);
  * ONE mode decision applies to exactly ONE image: the action carries a mode vector, entry i is the mode of the i-th
    image still waiting for a mode decision (capture order); images get their mode at the first decision epoch after
    capture;
  * a UAV's channel gain is drawn once per slot and used both for its own link and as an interferer;
  * policy-independent randomness: the task list (image, deadline, quality floor per UAV x waypoint x tile) and the
    fading variates per slot x UAV are pre-drawn at reset from separate streams, so every policy faces the same
    exogenous scenario under the same seed (common random numbers);
  * deadline = hard expiry: an image that cannot be delivered (incl. decoding) by t_cap + dmax is discarded at that
    moment; no airtime is spent on it afterwards;
  * transmit energy = power x actual airtime; a UAV's sortie ends when its survey is done and its queue is empty,
    after which it consumes no energy and pays no time cost; battery limit is a hard constraint;
  * the controller's knowledge of per-image x per-mode payload / quality is a config choice (oracle | predicted |
    mean), and heuristics read it through the same accessor (mode_info); the observation contains time.
"""
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Parameter sources (paper / measured / assumed) are tabulated in the report, section 4.
DEFAULT = dict(
    # topology and link
    K=3, M=2, B_total=1e6, fc=2.4e9, noise_dbm_hz=-174.0, nf_db=7.0, p_levels=(0.02, 0.08, 0.2),
    bs=(0.0, 0.0, 25.0), rician_k_db=10.0, los_a=9.61, los_b=0.16, eta_los_db=1.0, eta_nlos_db=20.0,
    # UAVs and survey mission
    H=100.0, v_max=20.0, dt=1.0, T=200, sector_radius=(1000.0, 2500.0), n_wp=6, tiles_per_wp=4,
    wp_spacing=120.0, capture_radius=15.0, area_radius=3000.0, battery_j=3.21e5,   # 89.2 Wh (Phantom 4 Pro class, ~2 kg, matches the Zeng model: hover 168 W -> ~32 min)
    # per-image requirements
    deadline=(4.0, 12.0), qmin=(22.0, 30.0),
    # utility and reward weights
    q_span=15.0, xi_q=0.5, xi_d=0.5, w_energy=0.02, w_time=0.0,
    # controller knowledge of per-image x per-mode (bits, PSNR): oracle | predicted | mean
    mode_info="oracle", probe_mode="avif_5",
    # observation: number of queue entries described individually
    q_obs=6,
    # mechanism switches (ablation)
    mobility=True, interference=True, energy=True,
)


def propulsion_power(v):
    """Rotary-wing propulsion power (W) vs horizontal speed, Zeng, Xu, Zhang, IEEE TWC 2019, Table I values."""
    P0, Pi, U_tip, v0, d0, rho, s, A = 79.86, 88.63, 120.0, 4.03, 0.6, 1.225, 0.05, 0.503
    return (P0 * (1 + 3 * v ** 2 / U_tip ** 2)
            + Pi * math.sqrt(math.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2))
            + 0.5 * d0 * rho * s * A * v ** 3)


class ModeTable:
    """Per-image x per-mode payload / quality / latency from the A1 profile files (measured, not modelled)."""

    def __init__(self, profile_files, modes=None, cpu_file=None):
        rows = []
        for f in profile_files:
            rows += [json.loads(l) for l in open(f)]
        by = defaultdict(dict)
        for r in rows:
            by[r["image"]][r["mode"]] = r
        all_modes = sorted({r["mode"] for r in rows})
        self.modes = list(modes) if modes else all_modes
        self.images = sorted(im for im in by if all(m in by[im] for m in self.modes))
        I, Mn = len(self.images), len(self.modes)
        self.bits = np.zeros((I, Mn))
        self.psnr = np.zeros((I, Mn))
        self.lpips = np.zeros((I, Mn))
        for i, im in enumerate(self.images):
            for j, m in enumerate(self.modes):
                r = by[im][m]
                self.bits[i, j], self.psnr[i, j], self.lpips[i, j] = 8 * r["bytes"], r["psnr"], r["lpips"]
        enc = {m: np.mean([by[im][m]["enc_s"] for im in self.images]) for m in self.modes}
        dec = {m: np.mean([by[im][m]["dec_s"] for im in self.images]) for m in self.modes}
        if cpu_file:  # learned codecs on the UAV CPU
            cr = defaultdict(list)
            for l in open(cpu_file):
                r = json.loads(l)
                cr[r["mode"]].append(r["enc_s"])
            for m in self.modes:
                if m in cr:
                    enc[m] = float(np.mean(cr[m]))
        self.enc_s = np.array([enc[m] for m in self.modes])
        self.dec_s = np.array([dec[m] for m in self.modes])
        self.fitted = False

    def split(self, frac=0.7, seed=0, probe_mode="avif_5"):
        """Train / test image split; the controller-side knowledge (mean table, predictor) is fitted on TRAIN only."""
        idx = np.random.default_rng(seed).permutation(len(self.images))
        n = int(frac * len(idx))
        tr, te = np.sort(idx[:n]), np.sort(idx[n:])
        self.fit_info(tr, probe_mode)
        return tr, te

    def fit_info(self, train_idx, probe_mode="avif_5"):
        """'mean': per-mode train means. 'predicted': per-mode linear fit of log(bits) and PSNR on log(bits of a cheap
        probe encode of the same image) -- the UAV runs the probe codec once per image (its time is charged)."""
        self.mean_bits = self.bits[train_idx].mean(0)
        self.mean_psnr = self.psnr[train_idx].mean(0)
        self.probe_j = self.modes.index(probe_mode) if probe_mode in self.modes else None
        if self.probe_j is not None:
            x = np.log(self.bits[train_idx, self.probe_j])
            X = np.stack([np.ones_like(x), x], 1)
            self.coef_bits = np.linalg.lstsq(X, np.log(self.bits[train_idx]), rcond=None)[0]
            self.coef_psnr = np.linalg.lstsq(X, self.psnr[train_idx], rcond=None)[0]
        self._pred_cache = {}
        self.fitted = True

    def predict(self, img):
        if img not in self._pred_cache:
            X = np.array([1.0, math.log(self.bits[img, self.probe_j])])
            b, q = np.exp(X @ self.coef_bits), X @ self.coef_psnr
            b[self.probe_j], q[self.probe_j] = self.bits[img, self.probe_j], self.psnr[img, self.probe_j]  # probe: exact
            self._pred_cache[img] = (b, q)
        return self._pred_cache[img]


@dataclass
class Task:
    uav: int
    wp: int
    tile: int
    img: int
    t_cap: float
    dmax: float
    qmin: float
    mode: int = -1
    enc_start: float = math.nan
    ready: float = math.inf       # time the encoded bitstream is available
    bits: float = 0.0             # payload of the chosen mode
    rem_bits: float = 0.0
    sent_bits: float = 0.0
    airtime: float = 0.0
    tx_start: float = math.nan
    done: bool = False
    dropped: bool = False         # discarded at deadline expiry (or at episode / battery end)
    util: float = 0.0
    ok: bool = False
    delay: float = math.inf


@dataclass
class UAV:
    pos: np.ndarray
    wps: np.ndarray
    wp_i: int = 0
    queue: list = field(default_factory=list)   # captured, not yet delivered / discarded (FIFO, capture order)
    energy: float = 0.0
    enc_free: float = 0.0                         # time the on-board encoder becomes idle
    active: bool = True                           # sortie in progress
    t_done: float = -1.0                          # sortie end time
    battery_out: bool = False


class UavSemcomEnv:
    """Centralised controller for K UAVs, decision epoch = slot of length dt.
    Action per UAV: (move, channel, power level, modes) with move in {0 hover, 1-8 compass at v_max, 9 macro to next
    waypoint, 10 macro to the BS}, channel in {0 silent, 1..M}, power level index, and modes = sequence of
    tiles_per_wp mode indices (entry i -> i-th image awaiting a mode decision; unused entries ignored)."""

    N_MOVE = 11

    def __init__(self, table, img_pool, cfg=None, seed=0):
        self.tab, self.pool = table, np.asarray(img_pool)
        self.c = dict(DEFAULT, **(cfg or {}))
        self.n_modes = len(table.modes)
        self.L = self.c["tiles_per_wp"]
        if self.c["mode_info"] != "oracle":
            assert table.fitted, "mode_info needs ModeTable.split()/fit_info() on the train images"
            if self.c["mode_info"] == "predicted":
                assert table.probe_j is not None, "probe mode not in the mode list"
        # two waypoint captures can never fall in one slot -> at most L images await a mode decision per epoch
        assert self.c["wp_spacing"] > 2 * (self.c["capture_radius"] + self.c["v_max"] * self.c["dt"])
        self.reset(seed)

    # ---------- channel ----------
    def _pathgain(self, pos):
        c = self.c
        bs = np.array(c["bs"])
        d3 = math.dist((pos[0], pos[1], c["H"]), bs)
        dh = math.dist(pos[:2], bs[:2])
        theta = math.degrees(math.atan2(c["H"] - bs[2], max(dh, 1e-3)))
        p_los = 1.0 / (1 + c["los_a"] * math.exp(-c["los_b"] * (theta - c["los_a"])))
        fspl_db = 20 * math.log10(d3) + 20 * math.log10(c["fc"]) - 147.55
        return p_los, 10 ** (-(fspl_db + c["eta_los_db"]) / 10), 10 ** (-(fspl_db + c["eta_nlos_db"]) / 10)

    def expected_gain(self, pos):
        p_los, g_l, g_n = self._pathgain(pos)
        return p_los * g_l + (1 - p_los) * g_n

    def slot_gain(self, k):
        """Channel power gain UAV k -> BS in the current slot (block fading, UAV at its slot-start position), from
        the pre-drawn variates: LoS w.p. p_los with Rician(K) fading, else Rayleigh."""
        p_los, g_l, g_n = self._pathgain(self.uavs[k].pos)
        s = min(self.slot, self.f_u.shape[0] - 1)
        if self.f_u[s, k] < p_los:
            kf = 10 ** (self.c["rician_k_db"] / 10)
            n = self.f_n[s, k]
            h = math.sqrt(kf / (kf + 1)) + math.sqrt(1 / (kf + 1)) * complex(n[0], n[1]) / math.sqrt(2)
            return g_l * abs(h) ** 2
        return g_n * self.f_e[s, k]

    def noise_w(self):
        c = self.c
        return 10 ** ((c["noise_dbm_hz"] + c["nf_db"] - 30) / 10) * c["B_total"] / c["M"]

    # ---------- controller knowledge ----------
    def mode_info(self, task):
        """(bits, PSNR) per mode for this image AS KNOWN TO THE CONTROLLER (policies and heuristics alike)."""
        mi = self.c["mode_info"]
        if mi == "oracle":
            return self.tab.bits[task.img], self.tab.psnr[task.img]
        if mi == "mean":
            return self.tab.mean_bits, self.tab.mean_psnr
        return self.tab.predict(task.img)

    def probe_s(self):
        return self.tab.enc_s[self.tab.probe_j] if self.c["mode_info"] == "predicted" else 0.0

    # ---------- episode ----------
    def reset(self, seed=None):
        c = self.c
        ss = np.random.SeedSequence(seed)
        r_layout, r_task, r_fade = [np.random.default_rng(s) for s in ss.spawn(3)]
        K, W, L = c["K"], c["n_wp"], self.L
        self.uavs = []
        for k in range(K):
            ang = 2 * math.pi * (k + r_layout.random() * 0.5) / K
            r = r_layout.uniform(*c["sector_radius"])
            centre = np.array([r * math.cos(ang), r * math.sin(ang)])
            # lawn-mower survey: n_wp points on two parallel lines across the radial direction
            tang = np.array([-math.sin(ang), math.cos(ang)])
            rad = np.array([math.cos(ang), math.sin(ang)])
            pts = []
            n2 = c["n_wp"] // 2
            for row in range(2):
                cols = range(n2) if row == 0 else reversed(range(n2))
                for j in cols:
                    pts.append(centre + (j - (n2 - 1) / 2) * c["wp_spacing"] * tang + row * c["wp_spacing"] * rad)
            start = centre - 1.5 * c["wp_spacing"] * rad
            self.uavs.append(UAV(pos=start.copy(), wps=np.array(pts)))
        # exogenous scenario, independent of the policy
        self.task_img = r_task.choice(self.pool, size=(K, W, L))
        self.task_dmax = r_task.uniform(*c["deadline"], size=(K, W, L))
        self.task_qmin = r_task.uniform(*c["qmin"], size=(K, W, L))
        n_slots = int(math.ceil(c["T"] / c["dt"]))
        self.f_u = r_fade.random((n_slots, K))
        self.f_n = r_fade.normal(size=(n_slots, K, 2))
        self.f_e = r_fade.exponential(size=(n_slots, K))
        self.t, self.slot = 0.0, 0
        self.tasks, self.finished = [], []
        self.segments = []            # (uav, channel, start, end) transmission intervals, for checks
        self.e_tx = 0.0
        self.last_tx = {}
        return self.obs()

    def utility(self, task, q, delay):
        """SCE utility (Huang et al. / Li et al.) minus 1 if the quality floor or the deadline is violated."""
        c = self.c
        ok = q >= task.qmin and delay <= task.dmax
        u = c["xi_q"] * float(np.clip((q - task.qmin) / c["q_span"], -1, 1)) + c["xi_d"] * max(task.dmax - delay, 0) / task.dmax
        return (u - (0 if ok else 1)), ok

    def _finish(self, task, t_end):
        q = self.tab.psnr[task.img, task.mode]
        delay = t_end + self.tab.dec_s[task.mode] - task.t_cap
        task.util, task.ok = self.utility(task, q, delay)
        task.delay, task.done = delay, True
        self.finished.append(task)
        return task.util

    def _drop(self, task):
        task.util, task.ok, task.done, task.dropped, task.delay = -1.0, False, True, True, math.inf
        self.finished.append(task)
        return -1.0

    def latest_tx_end(self, task):
        """Latest end of transmission that still meets the deadline (decoding at the BS included)."""
        dec = self.tab.dec_s[task.mode] if task.mode >= 0 else 0.0
        return task.t_cap + task.dmax - dec

    @staticmethod
    def _parse(a, L):
        mv, ch, pl, modes = a
        modes = tuple(int(m) for m in modes)
        assert len(modes) == L, "mode vector must have one entry per possible pending image"
        return int(mv), int(ch), int(pl), modes

    def step(self, actions):
        """One slot [t, t+dt). Order: (1) mode decisions + encoder scheduling, (2) slot channel gains, (3) FIFO
        transmission inside the slot, (4) deadline expiry, (5) movement / propulsion / captures at slot end,
        (6) sortie end / battery, (7) energy and time costs. Returns (obs, reward, done)."""
        c, dt = self.c, self.c["dt"]
        K, W, L = c["K"], c["n_wp"], self.L
        t0, t1 = self.t, self.t + dt
        acts = [self._parse(a, L) for a in actions]
        n_active = sum(u.active for u in self.uavs)
        reward, e_slot = 0.0, 0.0
        # (1) every image still without a mode gets the entry of the mode vector at its position (capture order)
        for k, (mv, ch, pl, modes) in enumerate(acts):
            u = self.uavs[k]
            pend = [tk for tk in u.queue if tk.mode < 0]
            assert len(pend) <= L
            for i, tk in enumerate(pend):
                m = modes[i]
                tk.mode, tk.bits, tk.rem_bits = m, self.tab.bits[tk.img, m], self.tab.bits[tk.img, m]
                tk.enc_start = max(t0, u.enc_free)
                tk.ready = tk.enc_start + self.probe_s() + self.tab.enc_s[m]
                u.enc_free = tk.ready
        # (2) one gain per UAV per slot, used for its own link AND when it interferes
        gains = [self.slot_gain(k) for k in range(K)]
        # (3) transmission: a UAV with a channel and an image ready before the slot ends is on air; interference is
        # counted for the whole slot (conservative: co-channel UAVs assumed to overlap fully)
        tx = {}
        for k, (mv, ch, pl, modes) in enumerate(acts):
            u = self.uavs[k]
            if u.active and ch > 0 and u.queue and u.queue[0].ready < t1:
                tx[k] = (ch, c["p_levels"][pl])
        self.last_tx = {}
        for k, (ch, p) in tx.items():
            interf = sum(p2 * gains[k2] for k2, (ch2, p2) in tx.items() if k2 != k and ch2 == ch) \
                if c["interference"] else 0.0
            rate = c["B_total"] / c["M"] * math.log2(1 + p * gains[k] / (self.noise_w() + interf))
            self.last_tx[k] = dict(ch=ch, p=p, gain=gains[k], interf=interf, rate=rate)
            u, cur = self.uavs[k], t0
            while u.queue:
                tk = u.queue[0]
                start = max(cur, tk.ready)
                if start >= t1:
                    break
                lat = self.latest_tx_end(tk)
                if start >= lat:                       # expired before it could be sent: discard, no airtime
                    reward += self._drop(tk)
                    u.queue.pop(0)
                    continue
                need = tk.rem_bits / rate if rate > 0 else math.inf
                delivered = start + need <= min(t1, lat)
                if delivered:                          # delivered inside this slot
                    end, sent = start + need, tk.rem_bits
                elif lat <= t1:                        # deadline hits mid-transmission: stop and discard
                    end, sent = lat, rate * (lat - start)
                else:                                  # slot over, continue next slot
                    end, sent = t1, rate * (t1 - start)
                if math.isnan(tk.tx_start):
                    tk.tx_start = start
                tk.rem_bits -= sent
                tk.sent_bits += sent
                tk.airtime += end - start
                self.segments.append((k, ch, start, end))
                e = p * (end - start)
                e_slot += e
                self.e_tx += e
                u.energy += e
                cur = end
                if delivered:
                    tk.rem_bits = 0.0
                    reward += self._finish(tk, end)
                    u.queue.pop(0)
                elif lat <= t1:
                    reward += self._drop(tk)
                    u.queue.pop(0)
                else:
                    break
        # (4) deadline expiry for every image still waiting (no channel, still encoding, ...)
        for u in self.uavs:
            keep = []
            for tk in u.queue:
                if self.latest_tx_end(tk) <= t1:
                    reward += self._drop(tk)
                else:
                    keep.append(tk)
            u.queue = keep
        # (5) movement (hard limits: v_max, area radius), propulsion energy, captures at slot end
        bs2 = np.array(c["bs"][:2])
        for k, (mv, ch, pl, modes) in enumerate(acts):
            u = self.uavs[k]
            if not u.active:
                continue
            p_old = u.pos.copy()
            if c["mobility"] and 1 <= mv <= 8:
                a = 2 * math.pi * (mv - 1) / 8
                u.pos = u.pos + c["v_max"] * dt * np.array([math.cos(a), math.sin(a)])
            elif c["mobility"] and mv in (9, 10):     # macro moves: exact heading to next waypoint / to the BS
                tgt = (u.wps[u.wp_i] if u.wp_i < W else u.pos) if mv == 9 else bs2
                d = tgt - u.pos
                n = float(np.linalg.norm(d))
                if n > 1e-9:
                    u.pos = u.pos + d / n * min(n, c["v_max"] * dt)
            r = float(np.linalg.norm(u.pos - bs2))
            if r > c["area_radius"]:
                u.pos = bs2 + (u.pos - bs2) * c["area_radius"] / r
            v = float(np.linalg.norm(u.pos - p_old)) / dt
            e = propulsion_power(v) * dt
            u.energy += e
            e_slot += e
            if u.wp_i < W and np.linalg.norm(u.pos - u.wps[u.wp_i]) <= c["capture_radius"] + c["v_max"] * dt / 2:
                w = u.wp_i
                for j in range(L):
                    tk = Task(uav=k, wp=w, tile=j, img=int(self.task_img[k, w, j]), t_cap=t1,
                              dmax=float(self.task_dmax[k, w, j]), qmin=float(self.task_qmin[k, w, j]))
                    u.queue.append(tk)
                    self.tasks.append(tk)
                u.wp_i += 1
        # (6) sortie end: survey done and queue empty; battery exhausted -> sortie aborted, its tasks fail
        for u in self.uavs:
            if not u.active:
                continue
            if u.energy >= c["battery_j"]:
                u.battery_out = True
                for tk in u.queue:
                    reward += self._drop(tk)
                reward += -1.0 * (W - u.wp_i) * L
                u.queue, u.active, u.t_done = [], False, t1
            elif u.wp_i >= W and not u.queue:
                u.active, u.t_done = False, t1
        # (7) costs: energy of the slot, and time per UAV whose sortie was running during the slot
        if c["energy"]:
            reward -= c["w_energy"] * e_slot / 1000.0
        reward -= c["w_time"] * dt * n_active
        self.t, self.slot = t1, self.slot + 1
        done = all(not u.active for u in self.uavs) or self.t >= c["T"] - 1e-9
        if done:   # horizon reached: queued images and never-captured waypoints fail
            for u in self.uavs:
                if u.active:
                    for tk in u.queue:
                        reward += self._drop(tk)
                    reward += -1.0 * (W - u.wp_i) * L
                    u.queue = []
        return self.obs(), reward, done

    # ---------- observation ----------
    def remaining_path(self, u):
        if u.wp_i >= len(u.wps):
            return 0.0
        pts = np.vstack([u.pos[None], u.wps[u.wp_i:]])
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def obs(self):
        """Global: elapsed time. Per UAV: activity, position, next-waypoint offset, survey progress, remaining path,
        expected SNR at the current position (statistical CSI; the slot's fading is not observed), energy used,
        queue length, encoder backlog; per queue entry (first q_obs): slack to deadline, quality floor, decided
        flag, payload (exact once encoded, else as known via mode_info), time until encoded; per image awaiting a
        mode decision (first L): slack, quality floor, and the per-mode (bits, PSNR) as known via mode_info."""
        c = self.c
        n_m = self.n_modes
        out = [self.t / c["T"]]
        for u in self.uavs:
            wp = u.wps[u.wp_i] if u.wp_i < len(u.wps) else u.pos
            snr_db = 10 * math.log10(c["p_levels"][-1] * self.expected_gain(u.pos) / self.noise_w())
            f = [float(u.active), u.pos[0] / 1000, u.pos[1] / 1000, (wp[0] - u.pos[0]) / 1000,
                 (wp[1] - u.pos[1]) / 1000, (len(u.wps) - u.wp_i) / len(u.wps), self.remaining_path(u) / 1000,
                 snr_db / 40, u.energy / 1e5, len(u.queue) / 8, max(u.enc_free - self.t, 0.0)]
            for i in range(c["q_obs"]):
                if i < len(u.queue):
                    tk = u.queue[i]
                    slack = self.latest_tx_end(tk) - self.t
                    if tk.mode >= 0:
                        b = tk.rem_bits if tk.ready <= self.t else self.mode_info(tk)[0][tk.mode] - tk.sent_bits
                        f += [1.0, slack / 12, tk.qmin / 40, 1.0, math.log10(1 + max(b, 0)) / 7,
                              max(tk.ready - self.t, 0.0)]
                    else:
                        f += [1.0, slack / 12, tk.qmin / 40, 0.0, 0.0, 0.0]
                else:
                    f += [0.0] * 6
            pend = [tk for tk in u.queue if tk.mode < 0]
            for i in range(self.L):
                if i < len(pend):
                    tk = pend[i]
                    b, q = self.mode_info(tk)
                    f += [1.0, (tk.t_cap + tk.dmax - self.t) / 12, tk.qmin / 40]
                    f += list(np.log10(b) / 7) + list(np.asarray(q) / 40)
                else:
                    f += [0.0] * (3 + 2 * n_m)
            out += f
        return np.array(out, dtype=np.float32)

    def metrics(self):
        c = self.c
        L, W = self.L, c["n_wp"]
        fin = self.finished
        n_total = len(self.uavs) * W * L
        uncaptured = sum(W - u.wp_i for u in self.uavs) * L
        utils = [tk.util for tk in fin] + [-1.0] * uncaptured
        deliv = [tk for tk in fin if not tk.dropped]
        return dict(
            mean_util=float(np.sum(utils) / n_total),
            ok_rate=float(sum(tk.ok for tk in fin) / n_total),
            drop_rate=float(sum(tk.dropped for tk in fin) / n_total),
            uncaptured_rate=float(uncaptured / n_total),
            q_viol_rate=float(sum(self.tab.psnr[tk.img, tk.mode] < tk.qmin for tk in deliv) / n_total),
            late_rate=float(sum(tk.delay > tk.dmax for tk in deliv) / n_total),
            mean_psnr=float(np.mean([self.tab.psnr[tk.img, tk.mode] for tk in deliv])) if deliv else float("nan"),
            mean_delay=float(np.mean([tk.delay for tk in deliv])) if deliv else float("nan"),
            energy_kj=float(sum(u.energy for u in self.uavs) / 1000),
            tx_energy_j=float(self.e_tx),
            captured=float(sum(u.wp_i for u in self.uavs) * L / n_total),
            mission_s=float(np.mean([u.t_done if u.t_done >= 0 else c["T"] for u in self.uavs])),
            makespan_s=float(max(u.t_done if u.t_done >= 0 else c["T"] for u in self.uavs)),
            airtime_s=float(sum(e - s for _, _, s, e in self.segments)),
            battery_out=int(sum(u.battery_out for u in self.uavs)),
        )
