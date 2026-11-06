"""A2: multi-UAV image delivery environment (UAV camera -> UAV encoder -> air-to-ground link -> ground BS).

Topology chosen to match the data (VisDrone images are taken BY the UAV camera): no satellite, no relay, no NOMA.
Mechanisms that make the decision sequential, each switchable for ablation:
  mobility     : every UAV flies a survey mission (waypoints far from the BS); each waypoint visit captures one image
                 (= one delivery task). Flying toward the BS improves the link but delays the mission; waypoints not
                 visited at episode end count as failed tasks.
  channel      : air-to-ground path loss with elevation-dependent LoS probability (Al-Hourani et al. 2014, urban),
                 per-slot Rician (LoS) / Rayleigh (NLoS) fading, co-channel interference between UAVs on the same
                 channel (M orthogonal channels shared by K UAVs).
  resources    : channel choice (or stay silent) and transmit power per slot; per-UAV FIFO queue of images.
  energy       : propulsion (rotary-wing model, Zeng et al. 2019) + transmit energy, weighted in the reward.
  semantics    : per IMAGE x MODE payload and quality from the real A1 measurements (bytes, PSNR, LPIPS), encoder
                 latency on the UAV, decoder latency at the BS.
Utility per task follows Huang/Li (SCE): xi_q * clip((Q - Qmin)/Q_span) + xi_d * (Dmax - delay)/Dmax, minus 1 if
the quality floor or the deadline is violated; undelivered / uncaptured tasks score -1.
"""
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Link budget regime (NOTE 2026-11-06: NOT payload-limited, best mode needs ~0.2 s of a 1 s slot; see report 3.2): 1 MHz shared by M=2 channels, UAVs surveying
# 1-2.5 km from the BS (urban A2G, mostly NLoS at low elevation), each waypoint captures `tiles_per_wp` 512x512 tiles.
DEFAULT = dict(
    K=3, M=2, B_total=1e6, fc=2.4e9, noise_dbm_hz=-174.0, nf_db=7.0, p_levels=(0.02, 0.08, 0.2),
    H=100.0, v_max=20.0, dt=1.0, T=200, bs=(0.0, 0.0, 25.0), sector_radius=(1000.0, 2500.0), n_wp=6,
    tiles_per_wp=4,
    wp_spacing=120.0, capture_radius=15.0, deadline=(4.0, 12.0), qmin=(22.0, 30.0), q_span=15.0,
    xi_q=0.5, xi_d=0.5, w_energy=0.02, w_time=0.0, rician_k_db=10.0, los_a=9.61, los_b=0.16, eta_los_db=1.0,
    eta_nlos_db=20.0, enc_device="cpu", mobility=True, interference=True, energy=True, per_image_obs=True,
)


def propulsion_power(v):
    """Rotary-wing propulsion power (W) vs horizontal speed, Zeng, Xu, Zhang, IEEE TWC 2019, Table I values."""
    P0, Pi, U_tip, v0, d0, rho, s, A = 79.86, 88.63, 120.0, 4.03, 0.6, 1.225, 0.05, 0.503
    return (P0 * (1 + 3 * v ** 2 / U_tip ** 2)
            + Pi * math.sqrt(math.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2))
            + 0.5 * d0 * rho * s * A * v ** 3)


class ModeTable:
    """Per-image x per-mode payload/quality/latency from A1 profile files."""

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

    def split(self, frac=0.7, seed=0):
        idx = np.random.default_rng(seed).permutation(len(self.images))
        n = int(frac * len(idx))
        return np.sort(idx[:n]), np.sort(idx[n:])


@dataclass
class Task:
    uav: int
    img: int
    t_cap: float
    dmax: float
    qmin: float
    mode: int = -1
    ready: float = 0.0        # time encoding finishes
    rem_bits: float = 0.0
    done: bool = False
    util: float = 0.0
    ok: bool = False
    delay: float = 0.0


@dataclass
class UAV:
    pos: np.ndarray
    wps: np.ndarray
    wp_i: int = 0
    queue: list = field(default_factory=list)
    energy: float = 0.0
    t_done: float = -1.0   # time the last waypoint was captured


class UavSemcomEnv:
    """Centralised controller for K UAVs. Action per UAV: (move in {hover, 8 directions at v_max}, channel in
    {silent, 1..M}, power level, mode for the head-of-queue image if not yet encoded)."""

    N_MOVE = 11  # 0 hover, 1-8 compass directions at v_max, 9 = to next waypoint, 10 = to BS

    def __init__(self, table, img_pool, cfg=None, seed=0):
        self.tab, self.pool = table, np.asarray(img_pool)
        self.c = dict(DEFAULT, **(cfg or {}))
        self.n_modes = len(table.modes)
        self.seed(seed)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

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

    def _gain_draw(self, pos):
        p_los, g_l, g_n = self._pathgain(pos)
        if self.rng.random() < p_los:
            k = 10 ** (self.c["rician_k_db"] / 10)
            h = math.sqrt(k / (k + 1)) + math.sqrt(1 / (k + 1)) * (self.rng.normal() + 1j * self.rng.normal()) / math.sqrt(2)
            return g_l * abs(h) ** 2
        return g_n * self.rng.exponential()

    def noise_w(self):
        c = self.c
        return 10 ** ((c["noise_dbm_hz"] + c["nf_db"] - 30) / 10) * c["B_total"] / c["M"]

    # ---------- episode ----------
    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        c = self.c
        self.t = 0.0
        self.uavs = []
        for k in range(c["K"]):
            ang = 2 * math.pi * (k + self.rng.random() * 0.5) / c["K"]
            r = self.rng.uniform(*c["sector_radius"])
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
        self.tasks = []
        self.finished = []
        return self.obs()

    def _capture(self, k):
        c = self.c
        task = Task(uav=k, img=int(self.rng.choice(self.pool)), t_cap=self.t,
                    dmax=self.rng.uniform(*c["deadline"]), qmin=self.rng.uniform(*c["qmin"]))
        self.uavs[k].queue.append(task)
        self.tasks.append(task)

    def utility(self, task, q, delay):
        c = self.c
        ok = q >= task.qmin and delay <= task.dmax
        u = c["xi_q"] * float(np.clip((q - task.qmin) / c["q_span"], -1, 1)) + c["xi_d"] * max(task.dmax - delay, 0) / task.dmax
        return (u - (0 if ok else 1)), ok

    def _finish(self, task, t_arrive):
        q = self.tab.psnr[task.img, task.mode]
        delay = t_arrive + self.tab.dec_s[task.mode] - task.t_cap
        task.util, task.ok = self.utility(task, q, delay)
        task.delay, task.done = delay, True
        self.finished.append(task)
        return task.util

    def step(self, actions):
        """actions: list of K tuples (move, channel, power_level, mode)."""
        c = self.c
        dt = c["dt"]
        reward, energy_j = 0.0, 0.0
        # 1) mode decisions + encoding start for head tasks without a mode
        for k, (mv, ch, pl, md) in enumerate(actions):
            q = self.uavs[k].queue
            if q and q[0].mode < 0:
                head = q[0]
                head.mode = int(md)
                head.ready = max(self.t, head.t_cap) + self.tab.enc_s[head.mode]
                head.rem_bits = self.tab.bits[head.img, head.mode]
        # 2) transmission with co-channel interference
        tx = {}
        for k, (mv, ch, pl, md) in enumerate(actions):
            q = self.uavs[k].queue
            if ch > 0 and q and q[0].mode >= 0 and q[0].ready < self.t + dt:
                tx[k] = (int(ch), c["p_levels"][int(pl)], self._gain_draw(self.uavs[k].pos))
        for k, (ch, p, g) in tx.items():
            interf = sum(p2 * self._gain_draw(self.uavs[k2].pos) for k2, (ch2, p2, _) in tx.items()
                         if k2 != k and ch2 == ch) if c["interference"] else 0.0
            rate = c["B_total"] / c["M"] * math.log2(1 + p * g / (self.noise_w() + interf))
            head = self.uavs[k].queue[0]
            t0 = max(self.t, head.ready)
            avail = self.t + dt - t0
            sent = rate * avail
            energy_j += p * avail
            self.uavs[k].energy += p * avail
            if sent >= head.rem_bits:
                reward += self._finish(head, t0 + head.rem_bits / rate)
                self.uavs[k].queue.pop(0)
            else:
                head.rem_bits -= sent
        # 3) deadline drops (undelivered past deadline -> -1)
        for u in self.uavs:
            keep = []
            for tk in u.queue:
                if self.t + dt - tk.t_cap > tk.dmax:
                    tk.util, tk.ok, tk.done, tk.delay = -1.0, False, True, float("inf")
                    self.finished.append(tk)
                    reward += -1.0
                else:
                    keep.append(tk)
            u.queue = keep
        # 4) movement, mission captures, propulsion energy
        for k, (mv, ch, pl, md) in enumerate(actions):
            u = self.uavs[k]
            v = 0.0
            if c["mobility"] and 1 <= mv <= 8:
                a = 2 * math.pi * (mv - 1) / 8
                u.pos = u.pos + c["v_max"] * dt * np.array([math.cos(a), math.sin(a)])
                v = c["v_max"]
            elif c["mobility"] and mv in (9, 10):  # macro moves: exact heading to next waypoint / to the BS
                tgt = (u.wps[u.wp_i] if u.wp_i < len(u.wps) else u.pos) if mv == 9 else np.array(c["bs"][:2])
                d = tgt - u.pos
                n = float(np.linalg.norm(d))
                step = min(n, c["v_max"] * dt)
                if n > 1e-9:
                    u.pos = u.pos + d / n * step
                v = step / dt
            e = propulsion_power(v) * dt
            u.energy += e
            energy_j += e
            if u.wp_i < len(u.wps) and np.linalg.norm(u.pos - u.wps[u.wp_i]) <= c["capture_radius"] + c["v_max"] * dt / 2:
                u.wp_i += 1
                for _ in range(c["tiles_per_wp"]):
                    self._capture(k)
        if c["energy"]:
            reward -= c["w_energy"] * energy_j / 1000.0
        # mission-time cost (Si et al. minimise collection time): per UAV per second its survey is unfinished
        for u in self.uavs:
            if u.wp_i < len(u.wps):
                reward -= c["w_time"] * dt
            elif u.t_done < 0:
                u.t_done = self.t + dt
        self.t += dt
        done = self.t >= c["T"]
        if done:  # uncaptured waypoints and still-queued tasks fail
            for u in self.uavs:
                missed = (len(u.wps) - u.wp_i) * c["tiles_per_wp"] + len(u.queue)
                reward += -1.0 * missed
                for tk in u.queue:
                    tk.util, tk.ok, tk.done = -1.0, False, True
                    self.finished.append(tk)
                u.queue = []
        return self.obs(), reward, done

    # ---------- observation ----------
    def obs(self):
        c = self.c
        out = []
        for k, u in enumerate(self.uavs):
            wp = u.wps[u.wp_i] if u.wp_i < len(u.wps) else u.pos
            g = self.expected_gain(u.pos)
            snr_db = 10 * math.log10(c["p_levels"][-1] * g / self.noise_w())
            f = [u.pos[0] / 1000, u.pos[1] / 1000, (wp[0] - u.pos[0]) / 1000, (wp[1] - u.pos[1]) / 1000,
                 (len(u.wps) - u.wp_i) / len(u.wps), len(u.queue) / 5, snr_db / 40, u.energy / 3e4]
            if u.queue:
                h = u.queue[0]
                f += [1.0, (h.dmax - (self.t - h.t_cap)) / 12, h.qmin / 40, float(h.mode >= 0),
                      math.log10(1 + h.rem_bits) / 7 if h.mode >= 0 else 0.0]
                if c["per_image_obs"]:
                    f += list(np.log10(self.tab.bits[h.img]) / 7) + list(self.tab.psnr[h.img] / 40)
                else:
                    f += list(np.log10(self.tab.bits[self.pool].mean(0)) / 7) + list(self.tab.psnr[self.pool].mean(0) / 40)
            else:
                f += [0.0] * 5 + [0.0] * (2 * self.n_modes)
            out.append(f)
        return np.array(out, dtype=np.float32).ravel()

    def metrics(self):
        fin = self.finished
        n_total = sum(len(u.wps) for u in self.uavs) * self.c["tiles_per_wp"]
        uncaptured = sum(len(u.wps) - u.wp_i for u in self.uavs) * self.c["tiles_per_wp"]
        utils = [tk.util for tk in fin] + [-1.0] * uncaptured   # never-captured images count as failed (-1)
        return dict(mean_util=float(np.sum(utils) / n_total), ok_rate=float(sum(tk.ok for tk in fin) / n_total),
                    energy_kj=float(sum(u.energy for u in self.uavs) / 1000),
                    captured=float(sum(u.wp_i for u in self.uavs) * self.c["tiles_per_wp"] / n_total),
                    mission_s=float(np.mean([u.t_done if u.t_done >= 0 else self.c["T"] for u in self.uavs])))
