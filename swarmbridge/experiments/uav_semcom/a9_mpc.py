"""A-E2 / A-E3 (reports/phases/2026-11-15_two-track/swarmbridge-two-track-plan.md, reviews #27-#28): mode-only sampling MPC vs PPO mode-only.

Control authority: the MPC chooses ONLY the compression mode of the image whose mode decision is due; movement,
channel and power come from the SAME heuristic slot policy that drives PPO mode-only (slot part of the A5 T:best
heuristic of the regime).

Information (same as the heuristics / PPO observation, all causal):
  * per-mode (bits, PSNR) of the image only through env.mode_info (A5 'predicted': probe-based regression);
  * queue state of the UAV (modes, remaining bits, encoder / transmitter backlog), time, UAV position;
  * statistical CSI: path gain, LoS probability and the fading DISTRIBUTION at the UAV's current position (the same
    channel model the env uses; this is what "correct model" means). The env's pre-drawn fading variates are never
    read (test: perturbing future fading does not change the decision);
  * future frames: not known; modelled by the train-split mean per-mode (bits, PSNR), the mean deadline / quality
    floor of the regime, and arrival every frame_spacing / v_max s while capture points remain.
Rate model of the planner: 'env' = the env's PHY (link_rate; MCS in phy_mcs_blind, which without co-channel reuse is
exact up to fading) or 'shannon' = B/M log2(1 + SNR) (the heuristics' model; mismatched under MCS). Co-channel
interference is not modelled by the planner (the slot policy of phy_mcs_blind does not reuse channels). The channel
share is the heuristics' M / #busy UAVs.

Planner: for each candidate mode m of the current image, N fading sample paths over the next S slots give per-slot
rates; the FIFO transmitter and serial encoder are simulated per sample: queue ahead, current image with mode m,
then the other images already awaiting a decision and H future frames, whose modes are chosen by a greedy rollout
(per sample, best predicted utility given that sample's backlog). Value = mean over samples of the summed utility
(env.utility; -1 for a predicted deadline miss, the transmitter then being busy until the deadline). N = 0 means
one deterministic path at the expected gain (the heuristics' rate estimate). Time budget (anytime): the regime's
budget-heuristic mode is computed first and evaluated as the incumbent; the other modes follow; once planning exceeds
budget_ms the best mode evaluated so far is returned. The fallback computation and the incumbent's evaluation are
inside the timed decision; timeouts are counted and their consequences are in the episode return.

  run OUT --regimes phy_mcs_blind,ref --configs env:0:0,shannon:8:4,... [--seeds 8000:8020 --procs 12]
      config = MODEL:N:H[:BUDGET_MS]
"""
import argparse
import copy
import json
import math
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import a5_eval as AE
import a55_heur as H
import env as E
import env_ev as V
import env_survey as SV
import mode_heur as MH
import policies as P

S_SLOTS = 40


class Feedback:
    """A-E9 (review #34): causal delayed / lossy throughput feedback for the planner's rate model.

    Every (UAV, slot) throughput report produced at a slot boundary t (the slot that just ended) is lost with
    probability `loss` (Bernoulli, independent across UAVs and slots; the draw depends only on (seed, episode index,
    slot index, UAV), so the calibrator and the error tracker, each holding a Feedback with the same seed, lose the
    same reports) and otherwise becomes usable `delay` slots later (t + delay * dt). Reports still in flight when an
    episode ends are discarded. delay=0, loss=0 reproduces the immediate observed-throughput update exactly.
    Only the rate model (CalRates / RateErrTracker) is affected; the queue state seen by PPO / the slot policy is the
    A5 environment's observation and is unchanged (stated limitation)."""

    def __init__(self, delay=0, loss=0.0, seed=0):
        self.delay, self.loss, self.seed = int(delay), float(loss), int(seed)
        self.ep, self.q = -1, []
        self.n_sent = self.n_lost = self.n_used = 0

    def new_episode(self):
        self.ep += 1
        self.q = []

    def lost(self, t, dt, k):
        if self.loss <= 0:
            return False
        return bool(np.random.default_rng([self.seed, self.ep, int(round(t / dt)), int(k)]).random() < self.loss)

    def push(self, t, dt, k, payload):
        self.n_sent += 1
        if self.lost(t, dt, k):
            self.n_lost += 1
            return
        self.q.append((t + self.delay * dt - 1e-9, payload))

    def pop(self, t):
        out = [p for a, p in self.q if a <= t]
        self.q = [(a, p) for a, p in self.q if a > t]
        self.n_used += len(out)
        return out


class CalRates:
    """Causal, ACK-based empirical rate model (review #29). At each slot boundary the realised link rates of the
    transmission segments of the slot that just ended (known to the sender through ACKs) are stored in bins of the
    UAV's expected SNR at that slot's start (statistical CSI, 2 dB bins). The planner samples per-slot rates from the
    bin of the UAV's current expected SNR (neighbouring bins pooled if needed) once >= MIN_AIR s of air time has been
    observed there; otherwise it falls back to the Shannon model scaled by the pooled ratio rho (observed / Shannon).
    persistent=False resets at every episode (cold); True keeps what earlier episodes of the run observed (warm).
    slot_weight (A-E5): airtime weighting (False, the A-E4 version) is biased low, because a UAV whose queue empties
    early in a good-fading slot logs little airtime at the high rate while a bad-fading slot logs a full slot at the
    low rate. With slot_weight=True each (UAV, slot) with transmission is ONE observation of that slot's rate (its
    airtime-weighted mean within the slot) with weight dt; whether a UAV transmits in a slot is decided before the
    slot's fading is drawn, so this is unbiased for the per-slot rate distribution."""
    BIN_DB, MIN_AIR = 2.0, 20.0

    def __init__(self, persistent=False, slot_weight=False, fb=None, window=None):
        """window (A-E11 recency baseline): keep only the last `window` observations per SNR bin (warm-start and
        online alike, oldest dropped first); the rho fallback then pools the last 10 x window observations."""
        self.persistent, self.slot_weight, self.fb, self.window = persistent, slot_weight, fb, window
        self.bins, self.num, self.den = {}, 0.0, 0.0
        self.rho_hist = []
        self.tasks, self.seg_i, self.prev = None, 0, None
        self.snap = None

    def freeze(self):
        """A-E6 'pre' protocol: keep the current state (from separate calibration episodes) as the start state of
        every following episode (test episodes then do not share what they observe)."""
        self.snap = copy.deepcopy((self.bins, self.num, self.den, self.rho_hist))

    def _snr_db(self, env, k):
        c = env.c
        return 10 * math.log10(max(c["p_levels"][-1] * env.expected_gain(env.uavs[k].pos) / env.noise_w(), 1e-30))

    def update(self, env):
        if self.tasks is not env.tasks:                       # new episode
            self.tasks, self.seg_i, self.prev = env.tasks, 0, None
            if self.fb is not None:
                self.fb.new_episode()
            if self.snap is not None:                       # 'pre': every episode restarts from the warm-start
                self.bins, self.num, self.den, self.rho_hist = copy.deepcopy(self.snap)
            elif not self.persistent:
                self.bins, self.num, self.den, self.rho_hist = {}, 0.0, 0.0, []
        if self.prev is not None and self.prev[0] != env.t:
            t0, snr = self.prev
            for k, r, dur in slot_obs(env, self, env.c["dt"] if self.slot_weight else None):
                key = int(math.floor(snr[k] / self.BIN_DB))
                sh = env.c["B_total"] / env.c["M"] * math.log2(1 + 10 ** (snr[k] / 10))
                if self.fb is None:
                    self._add(key, r, dur, sh)
                else:
                    self.fb.push(env.t, env.c["dt"], k, (key, r, dur, sh))
        if self.fb is not None:
            for pl in self.fb.pop(env.t):
                self._add(*pl)
        self.prev = (env.t, [self._snr_db(env, k) for k in range(len(env.uavs))])

    def _add(self, key, r, dur, sh):
        self.bins.setdefault(key, ([], []))
        self.bins[key][0].append(r); self.bins[key][1].append(dur)
        self.num += r * dur; self.den += sh * dur
        if self.window:
            rs, ws = self.bins[key]
            if len(rs) > self.window:
                del rs[:-self.window]; del ws[:-self.window]
            self.rho_hist.append((r * dur, sh * dur))
            if len(self.rho_hist) > 10 * self.window:
                a, b = self.rho_hist.pop(0)
                self.num -= a; self.den -= b

    def rho(self):
        return (self.num + C_RHO) / (self.den + C_RHO)

    def sample(self, env, k, shape, rng):
        key = int(math.floor(self._snr_db(env, k) / self.BIN_DB))
        rs, ws = [], []
        for d in (0, -1, 1, -2, 2):
            if key + d in self.bins:
                rs += self.bins[key + d][0]; ws += self.bins[key + d][1]
            if sum(ws) >= self.MIN_AIR:
                w = np.asarray(ws) / sum(ws)
                return np.asarray(rs)[rng.choice(len(rs), size=shape, p=w)]
        return None

    def mean(self, env, k):
        """airtime-weighted mean rate the planner would sample from (same bin pooling as sample); None if no data."""
        key = int(math.floor(self._snr_db(env, k) / self.BIN_DB))
        num = air = 0.0
        for d in (0, -1, 1, -2, 2):
            if key + d in self.bins:
                r, w = self.bins[key + d]
                num += float(np.dot(r, w)); air += sum(w)
            if air >= self.MIN_AIR:
                return num / air
        return None


C_RHO = 5e6


class PoolRates(CalRates):
    """Paper A E2b ablation: CalRates WITHOUT SNR conditioning. Same slot-weighted ACKed observations, same sampling
    of the empirical rate distribution, same MIN_AIR rule and rho fallback, but all observations of all UAVs share
    one bin (expected SNR at UAV positions is above -1e6 dB, so the bin key is 0, or -1 for negative-dB SNR, and
    the neighbour pooling merges both). Isolates the value of conditioning on the UAV's expected SNR."""
    BIN_DB = 1e6


class HarmRates(CalRates):
    """Paper A E2 (review of 2026-11-19): RobustMPC-style throughput predictor (Yin et al., SIGCOMM 2015) with the
    same interface as CalRates, so the planner, incumbent, budget and gate are unchanged. Per UAV it keeps the last
    `hist` observed per-slot rates (same slot-weighted ACKed observations as CalRates) and predicts the harmonic mean
    divided by (1 + the max relative prediction error over the last `hist` predictions). It ignores the UAV's
    position / expected SNR (history only), which is the difference to the SNR-binned CalRates. Deterministic:
    every sampled path gets the same rate. Before any observation of a UAV: the rho-scaled Shannon fallback."""

    def __init__(self, hist=5, **kw):
        super().__init__(**kw)
        self.hist = hist
        self.h, self.err, self.last_pred = {}, {}, {}

    def update(self, env):
        new_ep = self.tasks is not env.tasks
        if new_ep and not self.persistent:
            self.h, self.err, self.last_pred = {}, {}, {}
        n0 = self.seg_i if not new_ep else 0
        segs_before = len(env.segments)
        prev = self.prev
        super().update(env)                                  # keeps rho / bins bookkeeping and the segment cursor
        if prev is None or new_ep or prev[0] == env.t:
            return
        acc = {}
        for k, ch, a, b, interf, r in env.segments[n0:segs_before]:
            if b - a > 0:
                x = acc.setdefault(k, [0.0, 0.0])
                x[0] += r * (b - a); x[1] += b - a
        for k, (rd, d) in acc.items():
            r = rd / d
            if k in self.last_pred and r > 0:
                e = self.err.setdefault(k, [])
                e.append(abs(self.last_pred[k] - r) / r)
                del e[:-self.hist]
            hk = self.h.setdefault(k, [])
            hk.append(max(r, 1.0))
            del hk[:-self.hist]
            self.last_pred[k] = self._pred(k)

    def _pred(self, k):
        hk = self.h.get(k)
        if not hk:
            return None
        hm = len(hk) / sum(1.0 / x for x in hk)
        return hm / (1.0 + max(self.err.get(k, [0.0]) or [0.0]))

    def sample(self, env, k, shape, rng):
        p = self._pred(k)
        return None if p is None else np.full(shape, p)

    def mean(self, env, k):
        return self._pred(k)


class AblRates(CalRates):
    """Paper A E2c ablation ladder between HarmRates (RobustMPC-style) and PoolRates (pooled calibrator), one factor
    at a time. Observations are the same slot-weighted ACKed per-UAV slot rates (weight dt). Factors:
      per_uav  True: one history per UAV; False: all UAVs pooled
      hist     None: the whole episode; n: the last n observations of the key
      pred     'hmean' harmonic mean (rates floored at 1 bit/s, as HarmRates) | 'amean' mean | 'sample' empirical
               distribution (independent draws per path and slot, as CalRates)
      pess     divide a point prediction by 1 + max relative error of the last `hist or 5` predictions (HarmRates)
      drop0    leave zero-rate slots (every block of the UAV failed) out of the history
      min_air  observed weight (s) needed before the predictor is used; before that the rho-scaled Shannon fallback
    The SNR-bin-free planner interface is the same as CalRates / HarmRates; persistent=False (cold start)."""

    def __init__(self, per_uav, hist, pred, pess=False, drop0=False, min_air=None, **kw):
        super().__init__(persistent=False, slot_weight=True, **kw)
        self.per_uav, self.hist, self.pred, self.pess, self.drop0 = per_uav, hist, pred, pess, drop0
        self.min_air = min_air
        self.obs, self.err, self.last_pred = {}, {}, {}

    def update(self, env):
        if self.tasks is not env.tasks:
            self.tasks, self.seg_i, self.prev = env.tasks, 0, None
            self.bins, self.num, self.den, self.rho_hist = {}, 0.0, 0.0, []
            self.obs, self.err, self.last_pred = {}, {}, {}
        if self.prev is not None and self.prev[0] != env.t:
            t0, snr = self.prev
            for k, r, dur in slot_obs(env, self, env.c["dt"]):
                sh = env.c["B_total"] / env.c["M"] * math.log2(1 + 10 ** (snr[k] / 10))
                self.num += r * dur; self.den += sh * dur
                if self.drop0 and r <= 0:
                    continue
                key = k if self.per_uav else 0
                if self.pess and key in self.last_pred and self.last_pred[key] is not None and r > 0:
                    e = self.err.setdefault(key, [])
                    e.append(abs(self.last_pred[key] - r) / r)
                    del e[:-(self.hist or 5)]
                lst = self.obs.setdefault(key, [])
                lst.append((r, dur))
                if self.hist:
                    del lst[:-self.hist]
                self.last_pred[key] = self._point(key)
        self.prev = (env.t, [self._snr_db(env, k) for k in range(len(env.uavs))])

    def _ready(self, key):
        lst = self.obs.get(key)
        if not lst:
            return None
        if self.min_air is not None and sum(w for _, w in lst) < self.min_air:
            return None
        return lst

    def _point(self, key):
        lst = self._ready(key)
        if lst is None:
            return None
        r = np.array([x for x, _ in lst]); w = np.array([x for _, x in lst])
        p = (w.sum() / float(np.sum(w / np.maximum(r, 1.0)))) if self.pred == "hmean" else float(np.dot(r, w) / w.sum())
        if self.pess:
            p /= 1.0 + max(self.err.get(key, [0.0]) or [0.0])
        return p

    def sample(self, env, k, shape, rng):
        key = k if self.per_uav else 0
        if self.pred != "sample":
            p = self._point(key)
            return None if p is None else np.full(shape, p)
        lst = self._ready(key)
        if lst is None:
            return None
        r = np.array([x for x, _ in lst]); w = np.array([x for _, x in lst])
        return r[rng.choice(len(r), size=shape, p=w / w.sum())]

    def mean(self, env, k):
        key = k if self.per_uav else 0
        if self.pred != "sample":
            return self._point(key)
        lst = self._ready(key)
        return None if lst is None else float(np.dot([x for x, _ in lst], [x for _, x in lst]) / sum(x for _, x in lst))


# E2c ladder, HarmRates -> PoolRates, one factor per step (rh = HarmRates, pl = PoolRates are the existing ends):
ABL = {
    "ab0": dict(per_uav=True, hist=5, pred="hmean", pess=True, drop0=True),     # rh without zero-rate slots
    "ab1": dict(per_uav=True, hist=5, pred="hmean"),                            # rh without the pessimistic factor
    "ab2": dict(per_uav=True, hist=5, pred="amean"),                            # ab1, mean instead of harmonic mean
    "ab3": dict(per_uav=True, hist=5, pred="sample"),                           # ab2, sample the last 5 instead
    "ab4": dict(per_uav=True, hist=None, pred="sample"),                        # ab3, whole-episode memory
    "ab5": dict(per_uav=False, hist=None, pred="sample"),                       # ab4, all UAVs pooled
    "ab6": dict(per_uav=False, hist=None, pred="amean", min_air=CalRates.MIN_AIR),  # pl, mean instead of sampling
    "ab7": dict(per_uav=False, hist=None, pred="sample", min_air=CalRates.MIN_AIR),  # = pl re-implemented (check)
}


def slot_obs(env, owner, slot_dt=None):
    """consume the new segments of env (owner.seg_i cursor) -> list of (uav, rate, weight). slot_dt None: one entry
    per segment weighted by its duration; else one entry per UAV (the slot's airtime-weighted mean rate), weight
    slot_dt."""
    segs = env.segments
    out, acc = [], {}
    while owner.seg_i < len(segs):
        k, ch, a, b, interf, r = segs[owner.seg_i]
        owner.seg_i += 1
        dur = b - a
        if dur <= 0:
            continue
        if slot_dt is None:
            out.append((k, r, dur))
        else:
            x = acc.setdefault(k, [0.0, 0.0])
            x[0] += r * dur; x[1] += dur
    for k, (rd, d) in acc.items():
        out.append((k, rd / d, slot_dt))
    return out


class RateErrTracker:
    """A-E5 (review #30): prequential residuals of a planner rate model against observed throughput.

    Observation model (stated limitation): the throughput of finished transmission segments is read from the
    simulator at the next slot boundary ('observed-throughput calibration'); this is causal in time, but no ACK /
    feedback protocol, feedback delay or overhead is simulated (A-E9: optional Feedback adds report delay / loss).
    At the start of every slot the tracker stores, for each UAV, the planner model's predicted mean per-channel rate
    at the UAV's position (env / shannon: Monte-Carlo mean over the statistical fading model; cal: the calibrator's
    mean, or its rho-scaled Shannon fallback when the bin has no data). When the slot has ended, every segment of that
    slot adds (observed rate x duration, predicted rate x duration, duration) to the bin of the UAV's expected SNR at
    the slot start (2 dB bins). bias(env, k) pools the current bin with neighbours until >= MIN_AIR s of airtime and
    returns the relative bias sum(r*w)/sum(pred*w) - 1 (None if not enough airtime). The prediction for a slot is
    made before that slot's segments are seen (cal: the calibrator then still lacks them). Weighting w: per slot
    (default; unbiased, see CalRates) or per segment duration (slot_weight=False; biased low by queue emptying)."""
    BIN_DB, MIN_AIR = 2.0, 20.0

    def __init__(self, mpc, persistent=False, n_mc=64, seed=7, slot_weight=True, fb=None, window=None):
        """window (A-E11 recency baseline): the residual statistics of a bin use only its last `window` observations."""
        self.mpc, self.persistent, self.n_mc, self.slot_weight, self.fb = mpc, persistent, n_mc, slot_weight, fb
        self.window = window
        self.rng = np.random.default_rng(seed)
        self.bins = {}
        self.tasks, self.seg_i, self.prev = None, 0, None
        self.snap = None

    def freeze(self):
        self.snap = copy.deepcopy(self.bins)

    def _snr_db(self, env, k):
        return 10 * math.log10(max(env.c["p_levels"][-1] * env.expected_gain(env.uavs[k].pos) / env.noise_w(), 1e-30))

    def predict(self, env, k):
        m = self.mpc
        c = env.c
        p = c["p_levels"][-1]
        if m.model == "cal":
            r = m.cal.mean(env, k)
            if r is not None:
                return r
        p_los, g_l, g_n = env._pathgain(env.uavs[k].pos)
        n = self.n_mc
        los = self.rng.random(n) < p_los
        kf = 10 ** (c["rician_k_db"] / 10)
        z = (self.rng.normal(size=n) + 1j * self.rng.normal(size=n)) / math.sqrt(2)
        h = math.sqrt(kf / (kf + 1)) + math.sqrt(1 / (kf + 1)) * z
        g = np.where(los, g_l * np.abs(h) ** 2, g_n * self.rng.exponential(size=n))
        if m.model == "cal":
            return float(np.mean(c["B_total"] / c["M"] * np.log2(1 + p * g / env.noise_w()))) * m.cal.rho()
        return float(np.mean(m._rate(env, p * g)))

    def update(self, env):
        """call at every slot boundary, after CalRates.update (so the cal prediction includes the slot just ended)."""
        if self.tasks is not env.tasks:
            self.tasks, self.seg_i, self.prev = env.tasks, 0, None
            if self.fb is not None:
                self.fb.new_episode()
            if self.snap is not None:
                self.bins = copy.deepcopy(self.snap)
            elif not self.persistent:
                self.bins = {}
        if self.prev is not None and self.prev[0] != env.t:
            _, snr, pred = self.prev
            for k, r, dur in slot_obs(env, self, env.c["dt"] if self.slot_weight else None):
                pl = (int(math.floor(snr[k] / self.BIN_DB)), r, pred[k], dur)
                if self.fb is None:
                    self._add(*pl)
                else:
                    self.fb.push(env.t, env.c["dt"], k, pl)
        if self.fb is not None:
            for pl in self.fb.pop(env.t):
                self._add(*pl)
        if self.prev is None or self.prev[0] != env.t:
            n = len(env.uavs)
            self.prev = (env.t, [self._snr_db(env, k) for k in range(n)], [self.predict(env, k) for k in range(n)])

    def _add(self, key, r, pred, dur):
        acc = self.bins.setdefault(key, [0.0] * 6 + ([[]] if self.window else []))
        ratio = r / max(pred, 1.0)
        x = (r * dur, pred * dur, dur, ratio, ratio * ratio, 1)
        for i in range(6):
            acc[i] += x[i]
        if self.window:
            acc[6].append(x)
            if len(acc[6]) > self.window:
                old = acc[6].pop(0)
                for i in range(6):
                    acc[i] -= old[i]

    def bias(self, env, k):
        key = int(math.floor(self._snr_db(env, k) / self.BIN_DB))
        num = den = air = 0.0
        for d in (0, -1, 1, -2, 2):
            if key + d in self.bins:
                a = self.bins[key + d]
                num += a[0]; den += a[1]; air += a[2]
            if air >= self.MIN_AIR:
                return num / max(den, 1e-9) - 1.0
        return None

    def stats(self, env, k):
        """(relative bias, standard error of the mean observed/predicted ratio) with the same pooling; None if
        < MIN_AIR s."""
        key = int(math.floor(self._snr_db(env, k) / self.BIN_DB))
        a = np.zeros(6)
        for d in (0, -1, 1, -2, 2):
            if key + d in self.bins:
                a += np.asarray(self.bins[key + d][:6])
            if a[2] >= self.MIN_AIR:
                n = max(a[5], 1.0)
                var = max(a[4] / n - (a[3] / n) ** 2, 0.0) * n / max(n - 1, 1)
                return a[0] / max(a[1], 1e-9) - 1.0, math.sqrt(var / n)
        return None


class OnlineMonitor:
    """A-E11 (review #36): online-only check of a frozen warm-start rate model.

    `ref` is the warm-start calibrator as frozen after the calibration episodes (CalRates; only its bins are read, it
    never receives online data). At the start of every slot the monitor stores, per UAV, ref's mean per-slot rate in
    the bin of the UAV's expected SNR (None if ref has no data there); when the slot has ended, every transmitting
    (UAV, slot) of this episode adds the ratio observed / predicted (slot-weighted, as CalRates) to a FIFO of the last
    `window` ratios (pooled over UAVs and bins). Only observations of the current episode enter, so the historical
    pool never certifies itself. Alarm (sticky for the episode) when >= n_min ratios are available and the mean ratio
    m satisfies |log m| > delta (default log 1.5: a 50 % rate error) AND |m - 1| > z * se (se = sd / sqrt(n)).
    alarm_t = env time of the alarm. (A fixed tolerance is needed: with 10 calibration episodes the warm start covers
    a narrow SNR range, and even a matched warm start extrapolates with |log m| up to ~0.38 on single windows.)"""

    def __init__(self, ref, window=40, n_min=20, delta=math.log(1.5), z=3.0):
        self.ref, self.window, self.n_min, self.delta, self.z = ref, window, n_min, delta, z
        self.tasks, self.seg_i, self.prev = None, 0, None
        self.ratios, self.alarm_t = [], None

    @property
    def n(self):
        return len(self.ratios)

    def update(self, env):
        if self.tasks is not env.tasks:
            self.tasks, self.seg_i, self.prev = env.tasks, 0, None
            self.ratios, self.alarm_t = [], None
        if self.prev is not None and self.prev[0] != env.t:
            for k, r, _ in slot_obs(env, self, env.c["dt"]):
                pred = self.prev[1][k]
                if pred is not None:
                    self.ratios.append(r / max(pred, 1.0))
            del self.ratios[:-self.window]
            if self.alarm_t is None and self.n >= self.n_min:
                x = np.asarray(self.ratios)
                m = float(x.mean())
                se = float(x.std(ddof=1)) / math.sqrt(len(x))
                if abs(math.log(max(m, 1e-3))) > self.delta and abs(m - 1.0) > self.z * se:
                    self.alarm_t = float(env.t)
        if self.prev is None or self.prev[0] != env.t:
            self.prev = (env.t, [self.ref.mean(env, k) for k in range(len(env.uavs))])


class Deadline:
    """A-E8 (review #33): cooperative, proactive decision budget. The decision clock starts BEFORE PPO inference
    (start(t0)); before every unit of planning work (building a rate/context set, one mode evaluation, one robustness
    evaluation) the controller asks ok(stage, units): elapsed + units x (per-unit cost estimate) + margin <= budget.
    Per-unit costs are learnt online from the measured wall time of past work of the same stage (a high quantile of
    the last `window` per-unit times; `prior_ms` until 20 observations), a unit being one rolled-out image (1 + later
    queued images + future frames) of one mode evaluation. Work that has started is never interrupted (Python cannot
    pre-empt a running evaluation), so this is NOT a hard real-time guarantee: a unit that runs longer than estimated
    (OS scheduling, contention) still causes a miss; misses are measured and reported."""

    def __init__(self, budget_ms, margin_ms=3.0, prior_ms=1.0, q=0.95, window=300):
        self.budget, self.margin, self.prior, self.q, self.window = budget_ms, margin_ms, prior_ms, q, window
        self.obs, self.est, self.t0 = {}, {}, None

    def start(self, t0):
        self.t0 = t0

    def elapsed(self):
        return (time.perf_counter() - self.t0) * 1e3

    def unit(self, stage):
        return self.est.get(stage, self.prior)

    def ok(self, stage, units):
        return self.elapsed() + units * self.unit(stage) + self.margin <= self.budget

    def record(self, stage, units, ms):
        o = self.obs.setdefault(stage, [])
        o.append(ms / max(units, 1))
        if len(o) >= 20 and (len(o) < 100 or len(o) % 10 == 0):
            self.est[stage] = float(np.quantile(o[-self.window:], self.q))


class MPCMode:
    def __init__(self, model="env", n=8, horizon=4, budget_ms=None, fallback=None, seed=0, rollout="scenario",
                 cal=None, margin_z=None):
        """rollout: 'scenario' (future modes chosen per sampled fading path: optimistic, the A-E2 version) or
        'causal' (one mode per future frame for all samples: nonanticipative). model 'cal' uses CalRates `cal`.
        margin_z: switch away from the incumbent only if the mean per-sample value gain exceeds margin_z standard
        errors (paired over samples); None = plain argmax."""
        self.model, self.n, self.h, self.budget = model, n, horizon, budget_ms
        self.fallback, self.rollout, self.cal, self.margin_z = fallback, rollout, cal, margin_z
        self.rng = np.random.default_rng(seed)
        self.times, self.timeouts, self.chosen, self.switched, self.cal_fallbacks = [], 0, [], 0, 0
        self.scales, self.robust_checks, self.robust_rejects = None, 0, 0
        self.cands, self.last_rates = None, None
        self.last_cands, self.last_abort, self.last_full = [], None, False
        self.__name__ = f"mpc_{model}_n{n}_h{horizon}" + (f"_b{budget_ms:g}" if budget_ms else "")

    # ---- rates ----
    def _rate(self, env, sig):
        """vectorised env.link_rate without co-channel interference (identical values; checked in the tests)."""
        c = env.c
        noise = env.noise_w()
        bw = c["B_total"] / c["M"]
        snr = sig / noise
        phy = "shannon" if self.model == "shannon" else c["phy"]
        if phy == "shannon":
            return bw * np.log2(1 + snr)
        if phy == "gap":
            return bw * np.log2(1 + snr / 10 ** (c["phy_gap_db"] / 10))
        if phy in ("mcs", "mcs_blind", "mcs_partial"):        # no interference -> all three coincide
            i = np.searchsorted(E.CQI_THR_DB, 10 * np.log10(np.maximum(snr, 1e-30)), side="right")
            eff = np.r_[0.0, E.CQI_EFF][i]
            return bw * eff * (1 - c["phy_bler"])
        if phy == "la":                                        # review #44: env_la.link_rate (genie CQI x (1 - target))
            i = np.searchsorted(E.CQI_THR_DB, 10 * np.log10(np.maximum(snr, 1e-30)), side="right")
            eff = np.r_[0.0, E.CQI_EFF][i]
            return bw * eff * (1 - c["la_bler_target"])
        raise ValueError(phy)

    def _rates(self, env, k):
        """(N', S) effective per-slot rates (bit/s) for UAV k from its current position (block fading per slot)."""
        c = env.c
        p = c["p_levels"][-1]
        share = P.channel_share(env)
        if self.model == "cal":
            r = self.cal.sample(env, k, (max(self.n, 1), S_SLOTS), self.rng)
            if r is not None:
                return np.maximum(r * share, 1.0)
            self.cal_fallbacks += 1                      # not enough observations: rho-scaled Shannon samples
        if self.n == 0:
            g = np.full((1, S_SLOTS), env.expected_gain(env.uavs[k].pos))
        else:
            p_los, g_l, g_n = env._pathgain(env.uavs[k].pos)
            shape = (self.n, S_SLOTS)
            los = self.rng.random(shape) < p_los
            kf = 10 ** (c["rician_k_db"] / 10)
            z = (self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)) / math.sqrt(2)
            h = math.sqrt(kf / (kf + 1)) + math.sqrt(1 / (kf + 1)) * z
            g = np.where(los, g_l * np.abs(h) ** 2, g_n * self.rng.exponential(size=shape))
        if self.model == "cal":
            return np.maximum(c["B_total"] / c["M"] * np.log2(1 + p * g / env.noise_w()) * self.cal.rho() * share, 1.0)
        return np.maximum(self._rate(env, p * g) * share, 1.0)

    # ---- fluid FIFO transmitter on a slot grid ----
    @staticmethod
    def _grid(env, rates):
        dt = env.c["dt"]
        t0 = math.floor(env.t / dt + 1e-9) * dt
        cap = np.concatenate([np.zeros((len(rates), 1)), np.cumsum(rates * dt, 1)], 1)
        return t0, dt, cap, rates

    @staticmethod
    def _cap_at(G, t):
        t0, dt, cap, rates = G
        x = np.clip((np.asarray(t) - t0) / dt, 0, S_SLOTS - 1e-9)
        j = np.floor(x).astype(int)
        rows = np.arange(len(cap))
        extra = np.maximum((np.asarray(t) - t0) / dt - S_SLOTS, 0) * dt * rates[:, -1]
        return cap[rows, j] + (x - j) * dt * rates[rows, j] + extra

    @staticmethod
    def _time_at(G, target):
        t0, dt, cap, rates = G
        rows = np.arange(len(cap))
        j = np.minimum((cap < target[:, None]).sum(1) - 1, S_SLOTS - 1)
        j = np.maximum(j, 0)
        rem = target - cap[rows, j]
        return t0 + j * dt + rem / rates[rows, j]

    def _finish(self, G, tx_free, ready, bits):
        start = np.maximum(tx_free, ready)
        return self._time_at(G, self._cap_at(G, start) + bits)

    def _value(self, env, tab, G, tx_free, ready, bits, q, task, m):
        end = self._finish(G, tx_free, ready, bits)
        latest = task.t_cap + task.dmax - tab.dec_s[m]
        miss = end > latest
        delay = end + tab.dec_s[m] - task.t_cap
        c = env.c
        u = c["xi_q"] * np.clip((q - task.qmin) / c["q_span"], -1, 1) + c["xi_d"] * np.maximum(task.dmax - delay, 0) / task.dmax
        u = u - ((q < task.qmin) | (delay > task.dmax))
        return np.where(miss, -1.0, u), np.where(miss, np.maximum(latest, np.maximum(tx_free, ready)), end)

    def _rollout(self, env, tab, G, tx_free, enc_free, tk_like, b_hat, q_hat, g=1):
        """greedy per sample over modes for one later image; returns (utility, new tx_free, new enc_free).
        g > 1 (A-E8 stacked evaluation): the sample axis holds g independent groups (the same rate paths at g rate
        scales); the causal rollout then picks one mode per group, so every group equals a separate evaluation."""
        best_u = np.full(len(tx_free), -np.inf)
        best_tx = tx_free.copy()
        best_enc = enc_free.copy()
        best_mean = -np.inf if g == 1 else np.full(g, -np.inf)
        for m in range(env.n_modes):
            ready = enc_free + tab.enc_s[m]
            u, txe = self._value(env, tab, G, tx_free, ready, b_hat[m], q_hat[m], tk_like, m)
            if self.rollout == "causal" and g > 1:
                mu = u.reshape(g, -1).mean(1)
                better = mu > best_mean
                if better.any():
                    best_mean = np.where(better, mu, best_mean)
                    bm = np.repeat(better, len(u) // g)
                    best_u = np.where(bm, u, best_u)
                    best_tx = np.where(bm, txe, best_tx)
                    best_enc = np.where(bm, np.broadcast_to(ready, txe.shape), best_enc)
                continue
            if self.rollout == "causal":                  # one mode for every sample path (nonanticipative)
                if u.mean() > best_mean:
                    best_mean, best_u, best_tx, best_enc = u.mean(), u, txe, np.broadcast_to(ready, txe.shape).copy()
                continue
            better = u > best_u
            best_u = np.where(better, u, best_u)
            best_tx = np.where(better, txe, best_tx)
            best_enc = np.where(better, ready, best_enc)
        return best_u, best_tx, best_enc

    def _ctx(self, env, k, task, rates):
        """everything the per-mode evaluation needs for one set of per-slot rate paths."""
        u, c = env.uavs[k], env.c
        G = self._grid(env, rates)
        ns = len(rates)
        # transmitter backlog of the images ahead (already decided)
        tx_free = np.full(ns, env.t)
        later = []
        seen = False
        for tk in u.queue:
            if tk is task:
                seen = True
                continue
            if not seen and tk.mode >= 0:
                b = tk.rem_bits if tk.ready <= env.t else env.mode_info(tk)[0][tk.mode] - tk.sent_bits
                tx_free = self._finish(G, tx_free, np.full(ns, max(tk.ready, env.t) if tk.ready < math.inf else env.t),
                                       max(b, 0.0))
            elif seen and tk.mode < 0:
                later.append(tk)
        b_hat, q_hat = env.mode_info(task)
        # future frames (statistical)
        fut = []
        if self.h:
            dtf = c["frame_spacing"] / c["v_max"]
            left = max(len(u.wps) - u.wp_i, 0) if u.active else 0
            dm = sum(c["deadline"]) / 2
            qm = sum(c["qmin"]) / 2
            for i in range(min(self.h, left)):
                fut.append(E.Task(uav=k, wp=-1, tile=0, img=-1, t_cap=env.t + (i + 1) * dtf, dmax=dm, qmin=qm))
        return dict(G=G, ns=ns, tx_free=tx_free, later=later, enc0=max(env.t, u.enc_free), b_hat=b_hat, q_hat=q_hat,
                    fut=fut)

    def _eval_mode(self, env, ctx, task, m):
        """per-sample value of choosing mode m for `task` (+ rollout of later / future images)."""
        tab, G, ns = env.tab, ctx["G"], ctx["ns"]
        ready = np.full(ns, ctx["enc0"] + tab.enc_s[m])
        v, txf = self._value(env, tab, G, ctx["tx_free"], ready, ctx["b_hat"][m], ctx["q_hat"][m], task, m)
        enc = ready
        for tk in ctx["later"]:     # per-image info only if its probe has finished, else train means
            bh, qh = env.mode_info(tk) if tk.info_t <= env.t + 1e-9 else (tab.mean_bits, tab.mean_psnr)
            enc = np.maximum(enc, tk.info_t)
            uu, txf, enc = self._rollout(env, tab, G, txf, enc, tk, bh, qh, ctx.get("g", 1))
            v = v + uu
        for tk in ctx["fut"]:
            enc = np.maximum(enc, tk.t_cap) + env.probe_s()
            uu, txf, enc = self._rollout(env, tab, G, txf, enc, tk, tab.mean_bits, tab.mean_psnr, ctx.get("g", 1))
            v = v + uu
        return v

    def _resolvable(self, d):
        """paired gain d (per sample) exceeds margin_z standard errors (margin None: mean > 0)."""
        if self.margin_z is None or len(d) < 2:
            return d.mean() > 1e-12
        return d.mean() >= self.margin_z * d.std(ddof=1) / math.sqrt(len(d)) + 1e-12

    def stacked_ctx(self, env, k, task, rates, scales):
        """A-E8: one context holding the same rate paths at several scales (groups along the sample axis). The
        per-mode cost is nearly independent of the number of samples, so evaluating g scales together costs about
        one evaluation instead of g."""
        R = np.concatenate([rates if sc == 1.0 else np.maximum(rates * sc, 1.0) for sc in scales]) if len(scales) > 1 \
            else (rates if scales[0] == 1.0 else np.maximum(rates * scales[0], 1.0))
        ctx = self._ctx(env, k, task, R)
        ctx["g"] = len(scales)
        return ctx

    @staticmethod
    def units(ctx):
        return 1 + len(ctx["later"]) + len(ctx["fut"])

    def screen(self, env, k, task, h1, scales=None, dl=None):
        """A-E7 cheap first pass: mean value of every mode with a SHORT horizon h1 (same N, model and rollout). The
        decision cost is dominated by the per-mode rollout loop (it grows with the horizon and the number of evaluated
        modes) and is nearly independent of N, so the screen shortens the horizon. Returns per-mode mean values.
        A-E8: scales (e.g. [1, lo]) -> array (len(scales), n_modes) from one stacked pass (same samples); dl (Deadline)
        -> None if the budget does not allow the next mode evaluation (screen interrupted before it)."""
        h0 = self.h
        self.h = h1
        try:
            if scales is None:
                ctx = self._ctx(env, k, task, self._rates(env, k))
                return np.array([float(self._eval_mode(env, ctx, task, m).mean()) for m in range(env.n_modes)])
            t1 = time.perf_counter()
            ctx = self.stacked_ctx(env, k, task, self._rates(env, k), scales)
            g, L = ctx["g"], self.units(ctx)
            if dl is not None:
                dl.record("ctx", 1, (time.perf_counter() - t1) * 1e3)
            out = np.zeros((g, env.n_modes))
            for m in range(env.n_modes):
                if dl is not None and not dl.ok(f"screen{g}", L):
                    return None
                t1 = time.perf_counter()
                out[:, m] = self._eval_mode(env, ctx, task, m).reshape(g, -1).mean(1)
                if dl is not None:
                    dl.record(f"screen{g}", L, (time.perf_counter() - t1) * 1e3)
            return out
        finally:
            self.h = h0

    def selective(self, env, k, task, inc, lo, cand="n2", trig="nom", h1=1, n_cand=2, stack=False, dl=None):
        """A-E8 selective controller around the incumbent PPO mode `inc` (self.scales = robustness scales set by the
        caller; lo = low end of the sensitivity interval).
          cand 'all': no screen, full planning over all modes (= dhyb; anytime under a deadline).
          cand 'n2' : screen at the nominal model, candidates = top n_cand screened modes (= A-E7 shyb).
          cand 'np' : one stacked screen at scales [1, lo]; candidates = best nominal mode + best pessimistic mode
                      (the next nominal one if both coincide); trig: full planning if a mode beats PPO in the nominal
                      screen ('nom'), in the pessimistic screen ('pess') or in either ('any').
        Then the full MPC over {inc} + candidates and the endpoint-sensitivity check. last_abort: None | skip_screen |
        int_screen | skip_full | trunc_full | int_check (deadline-driven fallbacks / truncations)."""
        self.last_cands, self.last_abort, self.last_full = [], None, False
        if cand == "all":
            cands = None
        else:
            if dl is not None and not dl.ok("ctx", 1):
                self.last_abort = "skip_screen"
                return inc
            if cand == "n2" and not stack and dl is None:
                sv = self.screen(env, k, task, h1)[None]          # exactly the A-E7 path
            else:
                sv = self.screen(env, k, task, h1, scales=[1.0, lo] if cand == "np" else [1.0], dl=dl)
                if sv is None:
                    self.last_abort = "int_screen"
                    return inc
            nom = sv[0]
            on = [m for m in np.argsort(-nom) if m != inc]
            if cand == "np":
                pes = sv[1]
                op = [m for m in np.argsort(-pes) if m != inc]
                fire_n, fire_p = nom[on[0]] > nom[inc], pes[op[0]] > pes[inc]
                fire = dict(any=fire_n or fire_p, nom=fire_n, pess=fire_p)[trig]
                cands = [int(on[0]), int(op[0] if op[0] != on[0] else on[1])]
            else:
                fire = nom[on[0]] > nom[inc]
                cands = [int(m) for m in on[:n_cand]]
            if not fire:
                return inc
        self.last_full = True
        return self(env, k, task, incumbent=inc, cands=cands, dl=dl, stack=stack)

    def gain_at(self, env, k, task, a, b, rates):
        ctx = self._ctx(env, k, task, rates)
        return self._eval_mode(env, ctx, task, b) - self._eval_mode(env, ctx, task, a)

    def gain(self, env, k, task, a, b):
        """per-sample value gain of mode b over mode a with fresh rate samples of this planner's model."""
        ctx = self._ctx(env, k, task, self._rates(env, k))
        return self._eval_mode(env, ctx, task, b) - self._eval_mode(env, ctx, task, a)

    def __call__(self, env, k, task, incumbent=None, cands=None, dl=None, stack=False):
        """self.scales (set by the caller per decision, A-E6): if not None, a switch away from the incumbent must
        also be resolvable when every sampled rate path is multiplied by each scale (same samples: paired).
        A-E8: cands (modes besides the incumbent; default self.cands or all); stack: evaluate the nominal model and the
        robustness scales in one stacked pass (identical decisions, see stacked_ctx); dl (Deadline): proactive
        budget checks before each evaluation, reserving the robustness evaluations; the incumbent is returned when
        not even it, one candidate and the check fit (skip_full); evaluation stops early otherwise (trunc_full)."""
        t_start = time.perf_counter()
        if incumbent is not None:
            inc = int(incumbent)
        else:
            inc = int(self.fallback(env, k, task)) if self.fallback else 0
        if dl is not None and not dl.ok("ctx", 1):
            self.last_abort = "skip_full"
            return self._done(inc, inc, t_start)
        rates = self._rates(env, k)
        self.last_rates = rates
        stacked = bool(stack and self.scales)
        if stacked:
            ctx = self.stacked_ctx(env, k, task, rates, [1.0] + list(self.scales))
        else:
            ctx = self._ctx(env, k, task, rates)
        g, ns = ctx.get("g", 1), len(rates)
        L = self.units(ctx)
        stage = f"full{g}"
        if dl is not None:
            dl.record("ctx", 1, (time.perf_counter() - t_start) * 1e3)
        chk_units = 0 if (stacked or not self.scales) else 2 * len(self.scales)
        # anytime search: the fallback heuristic's mode is the incumbent and is evaluated first; on timeout the best
        # mode evaluated so far is returned (never an unevaluated or arbitrary one)
        pool = cands if cands is not None else (range(env.n_modes) if self.cands is None else self.cands)
        order = [inc] + [m for m in pool if m != inc]
        self.last_cands = list(order)
        best, best_v, vals = None, -math.inf, {}
        for i, m in enumerate(order):
            if best is not None and self.budget and (time.perf_counter() - t_start) * 1e3 > self.budget:
                self.timeouts += 1
                break
            if dl is not None:
                need = (2 if i == 0 else 1) * L + chk_units * L * dl.unit("full1") / max(dl.unit(stage), 1e-9)
                if not dl.ok(stage, need):
                    if i <= 1:
                        self.last_abort = "skip_full"
                        return self._done(inc, inc, t_start)
                    self.last_abort = "trunc_full"
                    self.last_cands = list(order[:i])
                    break
            t1 = time.perf_counter()
            vals[m] = self._eval_mode(env, ctx, task, m)
            if dl is not None:
                dl.record(stage, L, (time.perf_counter() - t1) * 1e3)
            val = float(vals[m][:ns].mean())
            if val > best_v:
                best, best_v = m, val
        if self.margin_z is not None and best != inc and not self._resolvable(vals[best][:ns] - vals[inc][:ns]):
            best = inc                                       # gain not resolvable from sampling noise: keep incumbent
        self.last_cand, self.last_rejected = best, False     # for diagnostics (candidate before the robustness check)
        if best != inc and self.scales:
            self.robust_checks += 1
            for j, sc in enumerate(self.scales):
                if stacked:
                    d = (vals[best] - vals[inc])[(j + 1) * ns:(j + 2) * ns]
                else:
                    if dl is not None and not dl.ok("full1", 2 * L):
                        best = inc
                        self.last_abort = "int_check"
                        break
                    t1 = time.perf_counter()
                    cs = self._ctx(env, k, task, np.maximum(rates * sc, 1.0))
                    d = self._eval_mode(env, cs, task, best) - self._eval_mode(env, cs, task, inc)
                    if dl is not None:
                        dl.record("full1", 2 * L, (time.perf_counter() - t1) * 1e3)
                if not self._resolvable(d):
                    best = inc                               # ordering not robust to the model's rate uncertainty
                    self.robust_rejects += 1
                    self.last_rejected = True
                    break
        return self._done(best, inc, t_start)

    def _done(self, best, inc, t_start):
        self.switched += int(best != inc)
        self.chosen.append(int(best))
        self.times.append(time.perf_counter() - t_start)
        return int(best)


# ---------------------------------------------------------------- runner
def job(j):
    regime, cfg, seeds = j
    import torch
    torch.set_num_threads(1)
    heur = AE.tuned(regime, "best")
    sp, mp_heur = H.from_name(heur, C.MODES)
    parts = cfg.split(":")
    model, n, h = parts[0], int(parts[1]), int(parts[2])
    budget = float(parts[3]) if len(parts) > 3 else None
    mpc = MPCMode(model, n, h, budget, fallback=mp_heur, seed=12345)
    rows = []
    for s in seeds:
        core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime))
        m = V.run_episode_ev(core, sp, mpc, s)
        rows.append({k: float(m[k]) for k in AE.KEEP})
    t = np.asarray(mpc.times) * 1e6
    return regime, cfg, heur, rows, dict(n=int(len(t)), mean_us=float(t.mean()), p50_us=float(np.percentile(t, 50)),
                                         p95_us=float(np.percentile(t, 95)), p99_us=float(np.percentile(t, 99)),
                                         timeouts=int(mpc.timeouts))


def init():
    AE.init(False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--regimes", default="phy_mcs_blind,ref")
    r.add_argument("--configs", required=True)
    r.add_argument("--seeds", default="8000:8020")
    r.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    jobs = [(rg, cf, seeds) for rg in a.regimes.split(",") for cf in a.configs.split(",")]
    t0 = time.time()
    with Pool(a.procs, initializer=init) as p:
        res = p.map(job, jobs, chunksize=1)
    out = []
    for rg, cf, heur, rows, tm in res:
        keys = ("return", "ok_rate", "drop_rate", "late_rate", "q_viol_rate", "mean_psnr")
        agg = {k: float(np.nanmean([x[k] for x in rows])) for k in keys}
        out.append(dict(regime=rg, config=cf, slot_policy_from=heur, per_seed_return=[x["return"] for x in rows],
                        **agg, timing=tm))
        print(f"{rg:14s} {cf:20s} ret {agg['return']:7.2f} ok {agg['ok_rate']:.3f} drop {agg['drop_rate']:.3f} "
              f"t {tm['mean_us']:.0f}/{tm['p95_us']:.0f}us to {tm['timeouts']}", flush=True)
    json.dump(dict(seeds=seeds, wall_s=time.time() - t0, rows=out), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
