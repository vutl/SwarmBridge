"""Step 0B: event/reservation simulator of a simplified Huang-inspired SAGIN with temporal coupling.

Coupling sources (each can be switched off for ablation):
  queue      each UAV receives from one LEO at a time and transmits to one user at a time (paper: a UAV connects to
             one LEO per slot); each user decodes one image at a time. Off = infinite parallel ports.
  visibility LEOs move (7.8 km/s, 750 km, overhead passes); a task whose source LEO is not visible goes over an ISL
             (eq. 4) to the highest visible LEO. Off = source LEO always overhead at zenith.
  interference UAV-user rate (eq. 6) counts UAVs transmitting at the same time. Off = no interference.
Decisions are made per task, in arrival order (mode in {M1, M2, M3_1, M3_2, M3_3}, relay UAV in {0..N-1});
ports serve in decision order (FIFO, non-preemptive). A decision's outcome is determined when it is committed, so
a policy's quality is measured by the realised per-task utility (model.utility), averaged over the G tasks.
Causal policies see expected link rates; the simulator realises rates with per-(task, UAV) fading draws that are
fixed by the scenario seed (common random numbers across policies). Hover only (no UAV movement) in this version.
"""
import math

import numpy as np

import model as M


class Scenario:
    def __init__(self, seed, lam, cfg):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.lam = lam
        n, m, k, g = M.N_UAV, M.M_USER, M.K_LEO, M.G_TASKS
        self.uav = np.c_[(np.arange(n) + rng.uniform(0, 1, n)) * M.AREA, rng.uniform(0, M.AREA, n)]
        area = rng.integers(0, n, m)
        self.user_area = area
        self.user = np.c_[(area + rng.uniform(0, 1, m)) * M.AREA, rng.uniform(0, M.AREA, m)]
        phi = [rng.uniform(-0.5, 0.0) * M.PHI_VIS]
        for _ in range(k - 1):
            phi.append(phi[-1] - rng.uniform(250e3, 500e3) / (M.R_E + M.H_LEO))
        self.phi0 = np.array(phi)
        self.arrival = np.cumsum(rng.exponential(1 / lam, g))
        self.src = rng.integers(0, k, g)
        self.dst = rng.integers(0, m, g)
        self.dmax = rng.uniform(*M.DMAX_RANGE, g)
        self.qmin = rng.uniform(*M.QMIN_RANGE, g)
        self.fade_leo = rng.exponential(1.0, (g, n))
        cn = lambda *s: (rng.normal(size=s) + 1j * rng.normal(size=s)) / math.sqrt(2)
        self.nlos = cn(g, n)
        self.nlos_int = cn(g, n, n)
        self.noise = M.noise_power(cfg["noise"])
        self.bits = M.mode_bits(cfg["payload"])
        # expected LEO->UAV rate as a function of slant range: tabulate E[log2(1+snr*g)] on a range grid
        self._dgrid = np.linspace(700e3, 1300e3, 61)
        self._rgrid = np.array([M.expected_log_rate(M.leo_uav_snr_mean(d, self.noise), n=2000) for d in self._dgrid])

    # ---- geometry ----
    def phi(self, leo, t):
        return self.phi0[leo] + M.OMEGA * t

    def visible(self, leo, t):
        return (not self.cfg["visibility"]) or abs(self.phi(leo, t)) <= M.PHI_VIS

    def slant(self, leo, t):
        if not self.cfg["visibility"]:
            return M.H_LEO
        return float(M.leo_elev_dist(self.phi(leo, t))[1])

    def best_leo(self, t):
        ph = np.abs(self.phi0 + M.OMEGA * t)
        j = int(np.argmin(ph))
        return j if ph[j] <= M.PHI_VIS else None

    def leo_rate(self, d, fade):
        if fade is None:
            return float(np.interp(d, self._dgrid, self._rgrid))
        return M.BW * math.log2(1 + M.leo_uav_snr_mean(d, self.noise) * fade)

    def uu_gain(self, uav, user, nlos):
        dx = self.uav[uav] - self.user[user]
        dist = math.sqrt(dx[0] ** 2 + dx[1] ** 2 + M.UAV_ALT ** 2)
        if nlos is None:   # mean power gain
            mu = M.RICIAN_MU
            return mu / (mu + 1) * dist ** -M.KAPPA_L + 1 / (mu + 1) * dist ** -M.KAPPA_N
        return float(M.uav_user_gain(dist, nlos))


class State:
    """Port reservations. Copy is cheap (small lists)."""

    def __init__(self, sc):
        self.rx = [0.0] * M.N_UAV
        self.tx = [0.0] * M.N_UAV
        self.dec = [0.0] * M.M_USER
        self.txint = [[] for _ in range(M.N_UAV)]
        self.total = 0.0
        self.n_ok = 0

    def copy(self):
        s = State.__new__(State)
        s.rx, s.tx, s.dec = list(self.rx), list(self.tx), list(self.dec)
        s.txint = [list(v) for v in self.txint]
        s.total, s.n_ok = self.total, self.n_ok
        return s


def outcome(sc, st, i, k, n, actual):
    """Plan task i with mode k via UAV n on state st (no mutation). Returns (utility, ok, delay, times)."""
    cfg = sc.cfg
    bits = sc.bits[k]
    t0 = sc.arrival[i]
    src = int(sc.src[i])
    holder, t_ready = src, t0
    if not sc.visible(src, t0):
        b = sc.best_leo(t0)
        if b is None:
            return -1.0 + M.sce(M.PSNR[k], 1e3, sc.qmin[i], sc.dmax[i]), False, 1e3, None
        chord = 2 * (M.R_E + M.H_LEO) * math.sin(abs(sc.phi(src, t0) - sc.phi(b, t0)) / 2)
        holder, t_ready = b, t0 + bits / M.isl_rate(max(chord, 1.0))
    q = cfg["queue"]
    t1s = max(t_ready, st.rx[n]) if q else t_ready
    if not sc.visible(holder, t1s):
        b = sc.best_leo(t1s)
        if b is None:
            return -1.0 + M.sce(M.PSNR[k], 1e3, sc.qmin[i], sc.dmax[i]), False, 1e3, None
        chord = 2 * (M.R_E + M.H_LEO) * math.sin(abs(sc.phi(holder, t1s) - sc.phi(b, t1s)) / 2)
        t1s += bits / M.isl_rate(max(chord, 1.0))
        holder = b
    r1 = sc.leo_rate(sc.slant(holder, t1s), sc.fade_leo[i, n] if actual else None)
    t1e = t1s + bits / r1
    t2s = max(t1e, st.tx[n]) if q else t1e
    user = int(sc.dst[i])
    interf = 0.0
    if cfg["interference"]:
        for j in range(M.N_UAV):
            if j != n and any(a <= t2s < b for a, b in st.txint[j][-8:]):
                interf += sc.uu_gain(j, user, sc.nlos_int[i, n, j] if actual else None)
    s = M.P_UAV * sc.uu_gain(n, user, sc.nlos[i, n] if actual else None)
    r2 = M.BW * math.log2(1 + s / (M.P_UAV * interf + sc.noise))
    t2e = t2s + bits / r2
    t3s = max(t2e, st.dec[user]) if q else t2e
    t3e = t3s + M.COMPUTE_S[k]
    d = t3e - t0
    u, ok = M.utility(M.PSNR[k], d, sc.qmin[i], sc.dmax[i])
    return u, ok, d, (t1e, t2s, t2e, t3e, user)


def commit(sc, st, i, k, n, actual=True):
    u, ok, d, tm = outcome(sc, st, i, k, n, actual=actual)
    if tm is not None:
        t1e, t2s, t2e, t3e, user = tm
        st.rx[n] = max(st.rx[n], t1e)
        st.tx[n] = max(st.tx[n], t2e)
        st.dec[user] = max(st.dec[user], t3e)
        st.txint[n].append((t2s, t2e))
    st.total += u
    st.n_ok += ok
    return u, ok, d


def run(sc, policy):
    st = State(sc)
    acts = []
    for i in range(M.G_TASKS):
        k, n = policy(sc, st, i)
        commit(sc, st, i, k, n)
        acts.append((k, n))
    return st.total / M.G_TASKS, st.n_ok / M.G_TASKS, acts


def replay(sc, acts, start_state=None, start=0, snapshots=False, actual=True):
    st = start_state.copy() if start_state is not None else State(sc)
    snaps = []
    for i in range(start, M.G_TASKS):
        if snapshots:
            snaps.append(st.copy())
        commit(sc, st, i, *acts[i], actual=actual)
    return st, snaps
