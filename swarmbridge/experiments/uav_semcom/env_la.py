"""Review #39 step A: block-level link adaptation PHY (phy = "la") on the A4/A5 survey env, no simulator restart.

What it adds to env_survey.UavSurveyEnv (all other mechanics unchanged; selected automatically when cfg phy == "la"):
  * blocks of la_tb seconds (default 0.1 s) on a global time grid; a UAV that transmits in a block uses ONE MCS
    (LTE CQI table, env.CQI_THR_DB / CQI_EFF) for that block;
  * MCS choice (per UAV, per block): the highest CQI whose threshold <= reported SINR - OLLA offset, where the
    reported SINR is the UAV's latest MEASURED SINR that is at least la_fb_delay blocks old (delayed CQI feedback),
    measured at the start of a block it transmitted in, including la_lambda x the co-channel interference it saw,
    plus a bias la_cqi_bias (dB) and Gaussian error la_cqi_sigma (dB); before any measurement exists the
    interference-free SNR at the current position (statistical CSI) is used;
  * block error: logistic waterfall in dB, BLER(SINR, CQI i) = 1 / (1 + (1 - t)/t * exp(la_slope (SINR - thr_i))),
    = t (= la_bler_target, 10 %) exactly at the CQI threshold (ASSUMED shape, slope 2.4 /dB: 10 % -> ~1 % in 1 dB);
    the SINR is the WORST SINR seen in the block (an interferer starting mid-block can still break it); outcome =
    pre-drawn uniform u[block, uav] < 1 - BLER (common random numbers across policies; monotone in SINR);
  * a successful block delivers eff_i x B/M bit/s for the time the UAV sent in it; a failed block delivers nothing
    (airtime and energy spent, the bits stay in the queue: ARQ without soft combining, unbounded within the image
    deadline; bounded HARQ with combining is a separate ablation, not implemented);
  * ACK / NACK of every block -> per-UAV outer-loop link adaptation: NACK: offset += la_olla_step, ACK: offset -=
    la_olla_step x t / (1 - t), clipped to +-la_olla_max dB (drives the long-run BLER to t);
  * episode-level randomisation (la_rand = {key: (lo, hi)} over la_cqi_sigma, la_fb_delay, la_olla_step,
    la_cqi_bias): drawn at reset from the episode seed; HIDDEN (never in the observation; logged in la_metrics);
  * causal link-adaptation observation la_features() (7 per UAV: reported SINR, report age, measured INR, OLLA
    offset, NACK rate of the last 10 / 50 blocks, NACK rate of the last 10 blocks sent with co-channel interference),
    appended by the Gym wrappers only when cfg obs_la is True (older controllers see an unchanged observation prefix);
  * link_rate() for phy "la" (used only by planners / heuristics as the expected rate) = genie CQI rate x (1 - t).

Review #41 additions (A-X6). Defaults: la_atomic True, la_mode "blind", la_interf "bursty", la_fade "slot"; A-X5 is
reproduced exactly with la_atomic False (only allowed with the other three at their defaults).
  * la_atomic: a block is ONE codeword. Bits sent in it are provisional; an image whose last bit is sent in it
    completes only when the block is ACKed (completion time = time of its last bit). When the block becomes a NACK
    (an interferer starting mid-block can break it) every bit sent in it is restored to its image and the images that
    had completed in it go back to the head of the queue (retransmission; dropped there if the deadline passed).
    Block ends are events, so outcomes commit promptly. (A-X5 credited bits sent before a mid-block flip: 0.3-0.5 %.)
  * la_mode, the MCS choice:
      blind  : as above (the UAV's own delayed SINR report, which contains whatever interference that past block saw,
               minus one OLLA offset); schedule-blind.
      sched  : scheduler-aware, CAUSAL: the BS knows its own channel grants and which co-scheduled UAVs radiate now
               (grants + buffer reports), and estimates SINR = p_k g_k / (N + sum_j p_j g_j) from each UAV's latest
               DELAYED SNR-per-watt report (same delay / bias / noise as above; statistical gain before any report).
               Two OLLA offsets per UAV: blocks predicted shared / predicted alone. An interferer that starts after
               the block began is not predicted.
      oracle : UPPER-BOUND DIAGNOSTIC, NOT DEPLOYABLE: the true SINR at the block start (realised fading and current
               interference), MCS maximising expected goodput, no OLLA.
  * la_interf, what radiates on a shared channel (two modelling ASSUMPTIONS, neither is "the" LTE/NR behaviour:
    whether a granted UE with an empty buffer transmits padding or skips depends on MAC procedures and configuration):
      bursty     : a co-channel UAV interferes only while it is actually sending image data;
      persistent : every co-scheduled active UAV with pending traffic (captured, not yet delivered; e.g. still
                   encoding) keeps its grant for the whole slot and fills it with padding when it has no ready data:
                   it interferes and spends transmit energy (logged as pad_air_s).
  * la_fade: "slot" = small-scale fading drawn once per 1 s slot (as the A5 env); "block" = redrawn every block
    (LoS / NLoS state still per slot), i.e. a coherence time of one block, shorter than any CQI delay.

Review #42 additions (A-X7):
  * ch_corr_m > 0: SPATIALLY CONSISTENT LoS / NLoS state. Per UAV a latent Gaussian z follows an AR(1) process in
    travelled distance (Gudmundson): z_s = rho z_{s-1} + sqrt(1 - rho^2) e_s, rho = exp(-d_s / ch_corr_m)
    [x exp(-dt / ch_corr_tau_s) if ch_corr_tau_s > 0, so that a hovering UAV also decorrelates], d_s = distance flown
    since the previous slot; the innovation e_s = Phi^-1(f_u[s, k]) re-uses the pre-drawn uniforms of the env (common
    random numbers across correlation distances). LoS iff Phi(z_s) < p_los(position): the MARGINAL P(LoS) at every
    position is exactly the p_los of the env, and ch_corr_m = 0 is exactly the original i.i.d. draw. Small-scale
    fading stays i.i.d. per slot.
  * ch_shadow_db > 0: log-normal shadowing with that standard deviation (dB), correlated with the same rho (own
    innovations). It CHANGES the marginal channel distribution (sensitivity only).
  * la_srs_period = P > 0: measurement probe (uplink sounding): a UAV without any measurement in the last P slots gets
    one at the start of the slot (its current slot gain at maximum power, same delay / bias / noise as a report).
    It costs no airtime or energy in the model (SRS overhead is a small fraction of a slot; ASSUMPTION).
  * slot_los(k): the realised LoS state of the current slot (used only by the LoS-only oracle and by diagnostics).
"""
import collections
import math

import numpy as np

import env as E
import env_survey as SV

LA_DEFAULT = dict(la_tb=0.1, la_bler_target=0.1, la_slope=2.4, la_olla_step=0.5, la_olla_max=10.0, la_fb_delay=2,
                  la_cqi_sigma=1.0, la_cqi_bias=0.0, la_lambda=1.0, la_rand=None, obs_la=False,
                  la_atomic=True, la_mode="blind", la_interf="bursty", la_fade="slot",
                  ch_corr_m=0.0, ch_corr_tau_s=0.0, ch_shadow_db=0.0, la_srs_period=0)
LA_MODES = ("blind", "sched", "oracle")
LA_INTERF = ("bursty", "persistent")
LA_FADE = ("slot", "block")
RAND_KEYS = ("la_cqi_sigma", "la_fb_delay", "la_olla_step", "la_cqi_bias")
THR = np.asarray(E.CQI_THR_DB, float)
EFF = np.asarray(E.CQI_EFF, float)
N_FEAT = 7


def bler(sinr_db, i, target=0.1, slope=2.4):
    """block error probability of CQI i at `sinr_db` (= target at the CQI threshold)."""
    x = min(max(slope * (sinr_db - THR[i]), -700.0), 700.0)
    return 1.0 / (1.0 + (1.0 - target) / target * math.exp(x))


def cqi_for(est_db):
    """highest CQI index whose threshold <= est_db (the lowest one if none)."""
    return max(int(np.searchsorted(THR, est_db, side="right")) - 1, 0)


class LinkAdapter:
    """per-UAV link adaptation state: delayed SINR reports, OLLA offset, ACK / NACK history."""

    def __init__(self, target, step, delay, olla_max):
        self.target, self.step, self.delay, self.max = target, step, int(delay), olla_max
        self.off = 0.0                                   # OLLA offset (blind / oracle; sched: blocks predicted alone)
        self.off_sh = 0.0                                # sched only: blocks predicted to be shared
        self.meas = collections.deque(maxlen=128)        # (block, SINR dB, INR dB, SNR-per-watt dB), all as measured
        self.hist = collections.deque(maxlen=50)         # (block, nack, shared)

    def report(self, b):
        """latest measurement usable at block b: measured on a block <= b - 1 - delay; None if there is none."""
        for m in reversed(self.meas):
            if m[0] <= b - 1 - self.delay:
                return m
        return None

    def feedback(self, nack, shared=False):
        d = self.step if nack else -self.step * self.target / (1.0 - self.target)
        if shared:
            self.off_sh = min(max(self.off_sh + d, -self.max), self.max)
        else:
            self.off = min(max(self.off + d, -self.max), self.max)

    def nack_rate(self, n, shared_only=False):
        h = [x[1] for x in self.hist if x[2] or not shared_only][-n:]
        return float(np.mean(h)) if h else 0.0


class LaSurveyEnv(SV.UavSurveyEnv):

    def __init__(self, table, img_pool, cfg=None, seed=0):
        super().__init__(table, img_pool, dict(LA_DEFAULT, **(cfg or {})), seed)
        c = self.c
        assert c["phy"] == "la" and c["la_mode"] in LA_MODES and c["la_interf"] in LA_INTERF and c["la_fade"] in LA_FADE
        assert c["la_atomic"] or (c["la_mode"], c["la_interf"], c["la_fade"]) == ("blind", "bursty", "slot"), \
            "the non-atomic accounting is kept only to reproduce A-X5"

    # ---------- episode ----------
    def reset(self, seed=None):
        self._la_init(seed)
        return super().reset(seed)

    def _la_init(self, seed):
        c, K = self.c, self.c["K"]
        rng = np.random.default_rng(np.random.SeedSequence([0 if seed is None else int(seed), 0x1A]))
        p = {k: c[k] for k in RAND_KEYS}
        for k, (lo, hi) in (c["la_rand"] or {}).items():
            p[k] = int(rng.integers(int(lo), int(hi) + 1)) if k == "la_fb_delay" else float(rng.uniform(lo, hi))
        self.la_p = p
        nb = int(math.ceil(c["T"] / c["la_tb"])) + 4
        self.la_u = rng.random((nb, K))
        self.la_z = rng.normal(size=(nb, K))
        rf = np.random.default_rng(np.random.SeedSequence([0 if seed is None else int(seed), 0x1B]))
        self.la_fn = rf.normal(size=(nb, K, 2))                 # per-block small-scale fading (la_fade = "block")
        self.la_fe = rf.exponential(size=(nb, K))
        rs = np.random.default_rng(np.random.SeedSequence([0 if seed is None else int(seed), 0x1C]))
        self.ch_eps_sh = rs.normal(size=(int(math.ceil(c["T"] / c["dt"])) + 2, K))   # shadowing innovations
        self._ch = [None] * K                                    # correlated large-scale state per UAV
        self._pos_hist = [[] for _ in range(K)]                  # UAV position at each slot start (causal history)
        self.la = [LinkAdapter(c["la_bler_target"], p["la_olla_step"], p["la_fb_delay"], c["la_olla_max"])
                   for _ in range(K)]
        self.la_cur = [None] * K
        self.la_st = dict(blocks=0, nack=0, shared_blocks=0, shared_nack=0, bits_ok=0.0, air=0.0, air_fail=0.0,
                          bits_rolled_back=0.0, tasks_reopened=0, pad_air=0.0, pred_shared_blocks=0,
                          pred_shared_nack=0, unpred_shared_blocks=0, unpred_shared_nack=0, srs=0)

    def block_of(self, t):
        return int(math.floor(t / self.c["la_tb"] + 1e-9))

    def _snr_free_db(self, k, p):
        return 10 * math.log10(max(p * self.expected_gain(self.uavs[k].pos) / self.noise_w(), 1e-30))

    # ---------- channel (fading timescale) ----------
    def _gains_now(self, b):
        S = self._slot
        if self.c["la_fade"] == "slot":
            return S["gains"]
        if S.get("gb_b") != b:
            S["gb_b"], S["gb"] = b, [self._block_gain(k, b) for k in range(self.c["K"])]
        return S["gb"]

    def _large_scale(self, k):
        """(uniform driving the LoS draw, shadowing normal) of UAV k in the current slot (cached per slot)."""
        c, s = self.c, min(self.slot, self.f_u.shape[0] - 1)
        if c["ch_corr_m"] <= 0 and c["ch_corr_tau_s"] <= 0:
            return float(self.f_u[s, k]), float(self.ch_eps_sh[s, k])
        st = self._ch[k]
        if st is not None and st["slot"] == self.slot:
            return st["u"], st["zs"]
        from scipy.special import ndtr, ndtri
        e = float(ndtri(min(max(self.f_u[s, k], 1e-12), 1 - 1e-12)))
        es = float(self.ch_eps_sh[s, k])
        pos = np.asarray(self.uavs[k].pos, float).copy()
        if st is None:
            z, zs = e, es
        else:
            d = float(np.linalg.norm(pos - st["pos"]))
            rho = math.exp(-d / c["ch_corr_m"]) if c["ch_corr_m"] > 0 else 1.0
            if c["ch_corr_tau_s"] > 0:
                rho *= math.exp(-(self.slot - st["slot"]) * c["dt"] / c["ch_corr_tau_s"])
            q = math.sqrt(max(1.0 - rho * rho, 0.0))
            z, zs = rho * st["z"] + q * e, rho * st["zs"] + q * es
        u = float(ndtr(z))
        self._ch[k] = dict(slot=self.slot, z=z, zs=zs, u=u, pos=pos)
        return u, zs

    def slot_los(self, k):
        """realised LoS state of UAV k in the current slot (NON-CAUSAL: oracle / diagnostics only)."""
        return self._large_scale(k)[0] < self._pathgain(self.uavs[k].pos)[0]

    def slot_gain(self, k):
        c = self.c
        if c["ch_corr_m"] <= 0 and c["ch_corr_tau_s"] <= 0 and c["ch_shadow_db"] <= 0:
            return super().slot_gain(k)
        p_los, g_l, g_n = self._pathgain(self.uavs[k].pos)
        s = min(self.slot, self.f_u.shape[0] - 1)
        u, zs = self._large_scale(k)
        sh = 10 ** (c["ch_shadow_db"] * zs / 10)
        if u < p_los:
            kf = 10 ** (c["rician_k_db"] / 10)
            n = self.f_n[s, k]
            h = math.sqrt(kf / (kf + 1)) + math.sqrt(1 / (kf + 1)) * complex(n[0], n[1]) / math.sqrt(2)
            return g_l * abs(h) ** 2 * sh
        return g_n * self.f_e[s, k] * sh

    def _block_gain(self, k, b):
        """la_fade = "block": LoS / NLoS state per slot (as slot_gain), small-scale fading redrawn every block."""
        p_los, g_l, g_n = self._pathgain(self.uavs[k].pos)
        bb = min(b, self.la_fe.shape[0] - 1)
        u, zs = self._large_scale(k)
        if u < p_los:
            kf = 10 ** (self.c["rician_k_db"] / 10)
            n = self.la_fn[bb, k]
            h = math.sqrt(kf / (kf + 1)) + math.sqrt(1 / (kf + 1)) * complex(n[0], n[1]) / math.sqrt(2)
            return g_l * abs(h) ** 2 * 10 ** (self.c["ch_shadow_db"] * zs / 10)
        return g_n * self.la_fe[bb, k] * 10 ** (self.c["ch_shadow_db"] * zs / 10)

    def _interferers(self, k, ch, sending, tx):
        """co-channel UAVs that radiate on k's channel now. bursty: only those actually sending data; persistent:
        every co-scheduled active UAV with pending traffic (it keeps its grant and fills it with padding)."""
        if not self.c["interference"]:
            return []
        if self.c["la_interf"] == "bursty":
            return [j for j in sending if j != k and tx[j][0] == ch]
        return [j for j in tx if j != k and tx[j][0] == ch and self.uavs[j].active and self.uavs[j].queue]

    # ---------- BS-side causal estimates (also used by schedulers) ----------
    def la_est_snrpw_db(self, k, b=None):
        """BS estimate of UAV k's received SNR per watt of transmit power (dB): its latest delayed report (with the
        report's bias / noise), else the statistical (expected-gain) value. Causal."""
        b = self.block_of(self.t) if b is None else b
        rep = self.la[k].report(b)
        if rep is not None:
            return rep[3]
        return 10 * math.log10(max(self.expected_gain(self.uavs[k].pos) / self.noise_w(), 1e-30))

    def _sched_est_db(self, k, p, interferers, tx, b):
        s = p * 10 ** (self.la_est_snrpw_db(k, b) / 10)
        i = sum(tx[j][1] * 10 ** (self.la_est_snrpw_db(j, b) / 10) for j in interferers)
        return 10 * math.log10(max(s / (1.0 + i), 1e-30))

    # ---------- link adaptation per block ----------
    def _la_open(self, k, b, p, sig, interf, interferers, tx):
        ad, c, nz = self.la[k], self.c, self.noise_w()
        mode, pred_sh = c["la_mode"], bool(interferers)
        if mode == "blind":
            rep = ad.report(b)
            i = cqi_for((rep[1] if rep is not None else self._snr_free_db(k, p)) - ad.off)
        elif mode == "sched":
            i = cqi_for(self._sched_est_db(k, p, interferers, tx, b) - (ad.off_sh if pred_sh else ad.off))
        else:   # ORACLE (upper-bound diagnostic, not deployable): true SINR at the block start, best expected goodput
            g = 10 * math.log10(max(sig / (nz + interf), 1e-30))
            i = int(np.argmax([EFF[j] * (1 - bler(g, j, c["la_bler_target"], c["la_slope"])) for j in range(len(THR))]))
        return dict(b=b, i=i, g=math.inf, u=float(self.la_u[min(b, len(self.la_u) - 1), k]), ok=True,
                    shared=interf > 0, pred_shared=pred_sh, air=0.0, sig=sig, p=p, interf0=interf,
                    bits=0.0, fin=0, prov=[], awaiting=[])

    def _la_close(self, k, rec):
        if rec["air"] <= 0:
            return
        ad, c, st, nz = self.la[k], self.c, self.la_st, self.noise_w()
        nack = not rec["ok"]
        ad.feedback(nack, shared=rec["pred_shared"] and c["la_mode"] == "sched")
        ad.hist.append((rec["b"], nack, rec["shared"]))
        st["blocks"] += 1
        st["nack"] += nack
        st["shared_blocks"] += rec["shared"]
        st["shared_nack"] += nack and rec["shared"]
        if rec["shared"]:
            key = "pred_shared" if rec["pred_shared"] else "unpred_shared"
            st[key + "_blocks"] += 1
            st[key + "_nack"] += nack
        if c["la_atomic"] and rec["ok"]:          # commit: the block's bits are delivered, its images finish
            st["bits_ok"] += rec["bits"]
            for tk, t_end in rec["awaiting"]:
                self._r += self._finish(tk, t_end)
        lam = c["la_lambda"]
        z = float(self.la_z[min(rec["b"], len(self.la_z) - 1), k])
        err = self.la_p["la_cqi_bias"] + self.la_p["la_cqi_sigma"] * z
        s = 10 * math.log10(max(rec["sig"] / (nz + lam * rec["interf0"]), 1e-30)) + err
        spw = 10 * math.log10(max(rec["sig"] / rec["p"] / nz, 1e-30)) + err
        ad.meas.append((rec["b"], s, 10 * math.log10(1 + lam * rec["interf0"] / nz), spw))

    def _la_rollback(self, k, rec):
        """atomic accounting: the block just became a NACK -> nothing sent in it is delivered. Restore the bits of
        every image sent in it and put the images that had completed in it back at the head of the queue."""
        st, q = self.la_st, self.uavs[k].queue
        for tk, bits in rec["prov"]:
            if not tk.done:
                tk.rem_bits += bits
                tk.sent_bits -= bits
        st["bits_rolled_back"] += rec["bits"]
        st["tasks_reopened"] += len(rec["awaiting"])
        q[0:0] = [tk for tk, _ in rec["awaiting"]]
        rec["prov"], rec["awaiting"], rec["bits"] = [], [], 0.0

    def _la_close_finished(self):
        for k, rec in enumerate(self.la_cur):
            if rec is not None and (rec["b"] + 1) * self.c["la_tb"] <= self.t + 1e-9:
                self._la_close(k, rec)
                self.la_cur[k] = None

    def begin_slot(self, slot_actions):
        self._la_close_finished()
        for k, u in enumerate(self.uavs):
            if len(self._pos_hist[k]) == self.slot:
                self._pos_hist[k].append(np.asarray(u.pos, float).copy())
        super().begin_slot(slot_actions)
        P = self.c["la_srs_period"]
        if P > 0:                                      # sounding probe: refresh stale measurements
            nz, b0 = self.noise_w(), self.block_of(self.t)
            per = int(round(self.c["dt"] / self.c["la_tb"]))
            pmax = self.c["p_levels"][-1]
            for k, u in enumerate(self.uavs):
                ad = self.la[k]
                if u.active and (not ad.meas or ad.meas[-1][0] <= b0 - P * per):
                    z = float(self.la_z[min(b0, len(self.la_z) - 1), k])
                    err = self.la_p["la_cqi_bias"] + self.la_p["la_cqi_sigma"] * z
                    g = self._slot["gains"][k]
                    spw = 10 * math.log10(max(g / nz, 1e-30)) + err
                    ad.meas.append((b0, spw + 10 * math.log10(pmax), 0.0, spw))
                    self.la_st["srs"] += 1

    def _end_slot(self):
        if self.c["la_atomic"]:
            self.t = self._slot["t1"]
            self._la_close_finished()
        return super()._end_slot()

    # ---------- PHY ----------
    def link_rate(self, sig, noise, interf, snr_free):
        if self.c["phy"] != "la":
            return super().link_rate(sig, noise, interf, snr_free)
        sinr_db = 10 * math.log10(max(sig / (noise + interf), 1e-30))
        i = int(np.searchsorted(THR, sinr_db, side="right"))
        return self.c["B_total"] / self.c["M"] * (EFF[i - 1] * (1 - self.c["la_bler_target"]) if i else 0.0)

    def _loop_once(self, horizon):
        """env_survey._loop_once with block-level MCS / BLER / ACK (the transmission part is the only change)."""
        S, c = self._slot, self.c
        atomic = c["la_atomic"]
        if atomic:
            self._la_close_finished()
        tx = S["tx"]
        S["n_ev"] += 1
        assert S["n_ev"] < 200000, "event loop did not terminate"
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
        b = self.block_of(tau)
        if sending:
            nxt = min(nxt, (b + 1) * c["la_tb"])
        if atomic:                               # pending block outcomes are events (prompt commit)
            for rec in self.la_cur:
                if rec is not None:
                    nxt = min(nxt, (rec["b"] + 1) * c["la_tb"])
        gains = self._gains_now(b)
        nz, bw = self.noise_w(), c["B_total"] / c["M"]
        rates = {}
        for k in list(sending):
            ch, p = tx[k]
            itf = self._interferers(k, ch, sending, tx)
            interf = sum(tx[j][1] * gains[j] for j in itf)
            sig = p * gains[k]
            rec = self.la_cur[k]
            if rec is None or rec["b"] != b:
                if rec is not None:
                    self._la_close(k, rec)
                rec = self.la_cur[k] = self._la_open(k, b, p, sig, interf, itf, tx)
            was_ok = rec["ok"]
            rec["g"] = min(rec["g"], 10 * math.log10(max(sig / (nz + interf), 1e-30)))
            rec["shared"] = rec["shared"] or interf > 0
            rec["ok"] = rec["u"] < 1.0 - bler(rec["g"], rec["i"], c["la_bler_target"], c["la_slope"])
            if atomic and was_ok and not rec["ok"]:
                self._la_rollback(k, rec)
                sending[k] = self.uavs[k].queue[0]       # a re-opened image goes first
            r = EFF[rec["i"]] * bw if rec["ok"] else 0.0
            rates[k] = (r, interf)
            h = sending[k]
            nxt = min(nxt, (tau + h.rem_bits / r) if r > 0 else math.inf, self.latest_tx_end(h))
        nxt = max(nxt, tau)
        d = nxt - tau
        for k, h in sending.items():
            ch, p = tx[k]
            r, interf = rates[k]
            fin = r > 0 and tau + h.rem_bits / r <= nxt + 1e-12
            sent = h.rem_bits if fin else r * d
            if math.isnan(h.tx_start):
                h.tx_start = tau
            h.rem_bits -= sent
            h.sent_bits += sent
            h.airtime += d
            rec = self.la_cur[k]
            rec["air"] += d
            rec["bits"] += sent
            rec["fin"] += int(fin)
            self.la_st["air"] += d
            if atomic:
                if rec["prov"] and rec["prov"][-1][0] is h:
                    rec["prov"][-1][1] += sent
                else:
                    rec["prov"].append([h, sent])
            else:
                self.la_st["bits_ok"] += sent
            if not rec["ok"]:
                self.la_st["air_fail"] += d
            if d > 0:
                self.segments.append((k, ch, tau, nxt, interf, r))
            e = p * d
            S["e"] += e
            self.e_tx += e
            self.uavs[k].energy += e
            if fin:
                h.rem_bits = 0.0
                self.uavs[k].queue.pop(0)
                if atomic:
                    rec["awaiting"].append((h, nxt))
                else:
                    self._r += self._finish(h, nxt)
            elif self.latest_tx_end(h) <= nxt + 1e-12:
                self._r += self._drop(h)
                self.uavs[k].queue.pop(0)
        if c["la_interf"] == "persistent" and d > 0:     # granted but idle co-scheduled UAVs transmit padding
            for k, (ch, p) in tx.items():
                u = self.uavs[k]
                if k not in sending and u.active and u.queue:
                    e = p * d
                    S["e"] += e
                    self.e_tx += e
                    u.energy += e
                    self.la_st["pad_air"] += d
        self.t = nxt if (sending or nxt < horizon) else horizon

    # ---------- observation / metrics ----------
    def la_features(self):
        """causal per-UAV link-adaptation features (K x N_FEAT), from reports and ACK / NACK only."""
        self._la_close_finished()
        b = self.block_of(self.t)
        out = np.zeros((self.c["K"], N_FEAT), dtype=np.float32)
        p = self.c["p_levels"][-1]
        for k, ad in enumerate(self.la):
            rep = ad.report(b)
            if rep is None:
                out[k, :3] = [self._snr_free_db(k, p) / 40, 1.0, 0.0]
            else:
                out[k, :3] = [rep[1] / 40, min((b - rep[0]) / 50, 1.0), rep[2] / 40]
            out[k, 3:] = [ad.off / 10, ad.nack_rate(10), ad.nack_rate(50), ad.nack_rate(10, shared_only=True)]
        return out

    def la_metrics(self):
        st = self.la_st

        def f(a, b):
            return st[a] / max(st[b], 1e-9)
        return dict(bler=st["nack"] / max(st["blocks"], 1), shared_bler=st["shared_nack"] / max(st["shared_blocks"], 1),
                    shared_block_frac=st["shared_blocks"] / max(st["blocks"], 1),
                    pred_shared_bler=f("pred_shared_nack", "pred_shared_blocks"),
                    unpred_shared_bler=f("unpred_shared_nack", "unpred_shared_blocks"),
                    unpred_shared_frac=f("unpred_shared_blocks", "shared_blocks"),
                    fail_air_frac=f("air_fail", "air"), goodput_kbps=st["bits_ok"] / max(st["air"], 1e-9) / 1e3,
                    delivered_mbit=st["bits_ok"] / 1e6, blocks=st["blocks"],
                    bits_rolled_back_frac=st["bits_rolled_back"] / max(st["bits_ok"] + st["bits_rolled_back"], 1e-9),
                    tasks_reopened=st["tasks_reopened"], pad_air_s=st["pad_air"],
                    la_hidden={k: float(v) for k, v in self.la_p.items()})
