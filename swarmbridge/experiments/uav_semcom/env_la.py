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
"""
import collections
import math

import numpy as np

import env as E
import env_survey as SV

LA_DEFAULT = dict(la_tb=0.1, la_bler_target=0.1, la_slope=2.4, la_olla_step=0.5, la_olla_max=10.0, la_fb_delay=2,
                  la_cqi_sigma=1.0, la_cqi_bias=0.0, la_lambda=1.0, la_rand=None, obs_la=False)
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
        self.off = 0.0
        self.meas = collections.deque(maxlen=128)        # (block, measured SINR dB, measured INR dB)
        self.hist = collections.deque(maxlen=50)         # (block, nack, shared)

    def report(self, b):
        """latest measurement usable at block b: measured on a block <= b - 1 - delay; None if there is none."""
        for mb, s, inr in reversed(self.meas):
            if mb <= b - 1 - self.delay:
                return mb, s, inr
        return None

    def feedback(self, nack):
        self.off += self.step if nack else -self.step * self.target / (1.0 - self.target)
        self.off = min(max(self.off, -self.max), self.max)

    def nack_rate(self, n, shared_only=False):
        h = [x[1] for x in self.hist if x[2] or not shared_only][-n:]
        return float(np.mean(h)) if h else 0.0


class LaSurveyEnv(SV.UavSurveyEnv):

    def __init__(self, table, img_pool, cfg=None, seed=0):
        super().__init__(table, img_pool, dict(LA_DEFAULT, **(cfg or {})), seed)
        assert self.c["phy"] == "la"

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
        self.la = [LinkAdapter(c["la_bler_target"], p["la_olla_step"], p["la_fb_delay"], c["la_olla_max"])
                   for _ in range(K)]
        self.la_cur = [None] * K
        self.la_st = dict(blocks=0, nack=0, shared_blocks=0, shared_nack=0, bits_ok=0.0, air=0.0, air_fail=0.0,
                         bits_in_nack_blocks=0.0, tasks_done_in_nack_blocks=0)

    def block_of(self, t):
        return int(math.floor(t / self.c["la_tb"] + 1e-9))

    def _snr_free_db(self, k, p):
        return 10 * math.log10(max(p * self.expected_gain(self.uavs[k].pos) / self.noise_w(), 1e-30))

    def _la_open(self, k, b, p, sig, interf):
        ad, c = self.la[k], self.c
        rep = ad.report(b)
        est = (rep[1] if rep is not None else self._snr_free_db(k, p)) - ad.off
        return dict(b=b, i=cqi_for(est), g=math.inf, u=float(self.la_u[min(b, len(self.la_u) - 1), k]), ok=True,
                    shared=interf > 0, air=0.0, sig=sig, interf0=interf, bits=0.0, fin=0)

    def _la_close(self, k, rec):
        if rec["air"] <= 0:
            return
        ad, c, st, nz = self.la[k], self.c, self.la_st, self.noise_w()
        nack = not rec["ok"]
        ad.feedback(nack)
        ad.hist.append((rec["b"], nack, rec["shared"]))
        st["blocks"] += 1
        st["nack"] += nack
        st["shared_blocks"] += rec["shared"]
        st["shared_nack"] += nack and rec["shared"]
        if nack:                       # audit (review #40): bits / task completions credited before the block flipped
            st["bits_in_nack_blocks"] += rec["bits"]
            st["tasks_done_in_nack_blocks"] += rec["fin"]
        lam = c["la_lambda"]
        z = float(self.la_z[min(rec["b"], len(self.la_z) - 1), k])
        s = 10 * math.log10(max(rec["sig"] / (nz + lam * rec["interf0"]), 1e-30)) + self.la_p["la_cqi_bias"] \
            + self.la_p["la_cqi_sigma"] * z
        ad.meas.append((rec["b"], s, 10 * math.log10(1 + lam * rec["interf0"] / nz)))

    def _la_close_finished(self):
        for k, rec in enumerate(self.la_cur):
            if rec is not None and (rec["b"] + 1) * self.c["la_tb"] <= self.t + 1e-9:
                self._la_close(k, rec)
                self.la_cur[k] = None

    def begin_slot(self, slot_actions):
        self._la_close_finished()
        super().begin_slot(slot_actions)

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
        tx, gains = S["tx"], S["gains"]
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
        nz, bw = self.noise_w(), c["B_total"] / c["M"]
        rates = {}
        for k in sending:
            ch, p = tx[k]
            interf = sum(tx[j][1] * gains[j] for j in sending if j != k and tx[j][0] == ch) \
                if c["interference"] else 0.0
            sig = p * gains[k]
            rec = self.la_cur[k]
            if rec is None or rec["b"] != b:
                if rec is not None:
                    self._la_close(k, rec)
                rec = self.la_cur[k] = self._la_open(k, b, p, sig, interf)
            rec["g"] = min(rec["g"], 10 * math.log10(max(sig / (nz + interf), 1e-30)))
            rec["shared"] = rec["shared"] or interf > 0
            rec["ok"] = rec["u"] < 1.0 - bler(rec["g"], rec["i"], c["la_bler_target"], c["la_slope"])
            r = EFF[rec["i"]] * bw if rec["ok"] else 0.0
            rates[k] = (r, interf)
            h = sending[k]
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
            rec = self.la_cur[k]
            rec["air"] += d
            rec["bits"] += sent
            rec["fin"] += int(fin)
            self.la_st["air"] += d
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
                self._r += self._finish(h, nxt)
                self.uavs[k].queue.pop(0)
            elif self.latest_tx_end(h) <= nxt + 1e-12:
                self._r += self._drop(h)
                self.uavs[k].queue.pop(0)
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
        return dict(bler=st["nack"] / max(st["blocks"], 1), shared_bler=st["shared_nack"] / max(st["shared_blocks"], 1),
                    shared_block_frac=st["shared_blocks"] / max(st["blocks"], 1),
                    fail_air_frac=st["air_fail"] / max(st["air"], 1e-9), goodput_kbps=st["bits_ok"] / max(st["air"], 1e-9) / 1e3,
                    delivered_mbit=st["bits_ok"] / 1e6, blocks=st["blocks"],
                    bits_in_nack_frac=st["bits_in_nack_blocks"] / max(st["bits_ok"], 1e-9),
                    tasks_done_in_nack_blocks=st["tasks_done_in_nack_blocks"],
                    la_hidden={k: float(v) for k, v in self.la_p.items()})
