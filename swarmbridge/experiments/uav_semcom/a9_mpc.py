"""A-E2 / A-E3 (reports/swarmbridge-two-track-plan.md, reviews #27-#28): mode-only sampling MPC vs PPO mode-only.

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

    def __init__(self, persistent=False, slot_weight=False):
        self.persistent, self.slot_weight = persistent, slot_weight
        self.bins, self.num, self.den = {}, 0.0, 0.0
        self.tasks, self.seg_i, self.prev = None, 0, None

    def _snr_db(self, env, k):
        c = env.c
        return 10 * math.log10(max(c["p_levels"][-1] * env.expected_gain(env.uavs[k].pos) / env.noise_w(), 1e-30))

    def update(self, env):
        if self.tasks is not env.tasks:                       # new episode
            self.tasks, self.seg_i, self.prev = env.tasks, 0, None
            if not self.persistent:
                self.bins, self.num, self.den = {}, 0.0, 0.0
        if self.prev is not None and self.prev[0] != env.t:
            t0, snr = self.prev
            for k, r, dur in slot_obs(env, self, env.c["dt"] if self.slot_weight else None):
                key = int(math.floor(snr[k] / self.BIN_DB))
                self.bins.setdefault(key, ([], []))
                self.bins[key][0].append(r); self.bins[key][1].append(dur)
                sh = env.c["B_total"] / env.c["M"] * math.log2(1 + 10 ** (snr[k] / 10))
                self.num += r * dur; self.den += sh * dur
        self.prev = (env.t, [self._snr_db(env, k) for k in range(len(env.uavs))])

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
    feedback protocol, feedback delay or overhead is simulated.
    At the start of every slot the tracker stores, for each UAV, the planner model's predicted mean per-channel rate
    at the UAV's position (env / shannon: Monte-Carlo mean over the statistical fading model; cal: the calibrator's
    mean, or its rho-scaled Shannon fallback when the bin has no data). When the slot has ended, every segment of that
    slot adds (observed rate x duration, predicted rate x duration, duration) to the bin of the UAV's expected SNR at
    the slot start (2 dB bins). bias(env, k) pools the current bin with neighbours until >= MIN_AIR s of airtime and
    returns the relative bias sum(r*w)/sum(pred*w) - 1 (None if not enough airtime). The prediction for a slot is
    made before that slot's segments are seen (cal: the calibrator then still lacks them). Weighting w: per slot
    (default; unbiased, see CalRates) or per segment duration (slot_weight=False; biased low by queue emptying)."""
    BIN_DB, MIN_AIR = 2.0, 20.0

    def __init__(self, mpc, persistent=False, n_mc=64, seed=7, slot_weight=True):
        self.mpc, self.persistent, self.n_mc, self.slot_weight = mpc, persistent, n_mc, slot_weight
        self.rng = np.random.default_rng(seed)
        self.bins = {}
        self.tasks, self.seg_i, self.prev = None, 0, None

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
            if not self.persistent:
                self.bins = {}
        if self.prev is not None and self.prev[0] != env.t:
            _, snr, pred = self.prev
            for k, r, dur in slot_obs(env, self, env.c["dt"] if self.slot_weight else None):
                acc = self.bins.setdefault(int(math.floor(snr[k] / self.BIN_DB)), [0.0, 0.0, 0.0])
                acc[0] += r * dur; acc[1] += pred[k] * dur; acc[2] += dur
        if self.prev is None or self.prev[0] != env.t:
            n = len(env.uavs)
            self.prev = (env.t, [self._snr_db(env, k) for k in range(n)], [self.predict(env, k) for k in range(n)])

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

    def _rollout(self, env, tab, G, tx_free, enc_free, tk_like, b_hat, q_hat):
        """greedy per sample over modes for one later image; returns (utility, new tx_free, new enc_free)."""
        best_u = np.full(len(tx_free), -np.inf)
        best_tx = tx_free.copy()
        best_enc = enc_free.copy()
        best_mean = -np.inf
        for m in range(env.n_modes):
            ready = enc_free + tab.enc_s[m]
            u, txe = self._value(env, tab, G, tx_free, ready, b_hat[m], q_hat[m], tk_like, m)
            if self.rollout == "causal":                  # one mode for every sample path (nonanticipative)
                if u.mean() > best_mean:
                    best_mean, best_u, best_tx, best_enc = u.mean(), u, txe, np.broadcast_to(ready, txe.shape).copy()
                continue
            better = u > best_u
            best_u = np.where(better, u, best_u)
            best_tx = np.where(better, txe, best_tx)
            best_enc = np.where(better, ready, best_enc)
        return best_u, best_tx, best_enc

    def __call__(self, env, k, task, incumbent=None):
        t_start = time.perf_counter()
        u, tab, c = env.uavs[k], env.tab, env.c
        rates = self._rates(env, k)
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
        enc0 = max(env.t, u.enc_free)
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
        # anytime search: the fallback heuristic's mode is the incumbent and is evaluated first; on timeout the best
        # mode evaluated so far is returned (never an unevaluated or arbitrary one)
        if incumbent is not None:
            inc = int(incumbent)
        else:
            inc = int(self.fallback(env, k, task)) if self.fallback else 0
        order = [inc] + [m for m in range(env.n_modes) if m != inc]
        best, best_v, vals = None, -math.inf, {}
        for m in order:
            if best is not None and self.budget and (time.perf_counter() - t_start) * 1e3 > self.budget:
                self.timeouts += 1
                break
            ready = np.full(ns, enc0 + tab.enc_s[m])
            v, txf = self._value(env, tab, G, tx_free, ready, b_hat[m], q_hat[m], task, m)
            enc = ready
            for tk in later:     # per-image info only if its probe has finished, else train means
                bh, qh = env.mode_info(tk) if tk.info_t <= env.t + 1e-9 else (tab.mean_bits, tab.mean_psnr)
                enc = np.maximum(enc, tk.info_t)
                uu, txf, enc = self._rollout(env, tab, G, txf, enc, tk, bh, qh)
                v = v + uu
            for tk in fut:
                enc = np.maximum(enc, tk.t_cap) + env.probe_s()
                uu, txf, enc = self._rollout(env, tab, G, txf, enc, tk, tab.mean_bits, tab.mean_psnr)
                v = v + uu
            vals[m] = v
            val = float(v.mean())
            if val > best_v:
                best, best_v = m, val
        if self.margin_z is not None and best != inc and len(vals[inc]) > 1:
            d = vals[best] - vals[inc]
            if d.mean() < self.margin_z * d.std(ddof=1) / math.sqrt(len(d)) + 1e-12:
                best = inc                                   # gain not resolvable from sampling noise: keep incumbent
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
