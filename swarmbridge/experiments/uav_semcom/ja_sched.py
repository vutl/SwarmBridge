"""Review #41 (A-X6): joint channel assignment as a strong scheduling heuristic, and its oracle-information variant.

Every slot, over the UAVs with pending traffic (active, non-empty queue), enumerate ALL assignments to {idle, 1..M}
((M+1)^n <= 243 for K = 5, M = 2) and take the one maximising

    V(a) = sum_k  w_k * min(R_k(a) * dt, B_k)

  B_k   : bits the UAV can still send (remaining bits of images whose mode is chosen; images not yet decided count as
          the median payload of decided images so far);
  w_k   : urgency^alpha, urgency = 1 / max(slack of its head image, 0.5 s);
  R_k(a): predicted goodput on its channel = B/M * eff(CQI(SINR_k(a) - margin if shared)) * (1 - BLER target),
          SINR_k(a) = p_k g_k / (N + sum_{j co-channel} f_j p_j g_j), f_j = activity of the co-channel UAV
          (persistent interference: 1; bursty: min(1, B_j / (R_j alone * dt)), a mean-field approximation).

Information (g_k):
  stat   : CAUSAL and deployable: the statistical channel model at the UAVs' current positions (LoS probability, LoS /
           NLoS path gains, Rician / Rayleigh fading), which the BS knows; V is averaged over N_MC = 32 joint draws
           (own rng seeded by the slot index, the env's randomness is untouched).
  report : CAUSAL, the BS's latest delayed SNR-per-watt report of each UAV (env.la_est_snrpw_db). Kept as a documented
           NEGATIVE: in this simulator fading AND the LoS state are redrawn i.i.d. every 1 s slot, so at a slot start a
           report carries no information about the coming slot beyond its large-scale part, and a UAV with one bad
           report is not scheduled, never re-measured, and starves (smoke test: 25-37 % of images dropped).
  los    : LoS-ONLY ORACLE (review #42): the true LoS / NLoS state of the coming slot, small-scale fading unknown
           (sampled). NON-CAUSAL diagnostic: how much of the full-oracle gain is large-scale information.
  filt   : CAUSAL: the statistical model with P(LoS) replaced by ch_filter.predict_los (posterior from the latest
           delayed report, propagated through the spatial correlation of the LoS process). Equals stat when the env
           has no spatial correlation.
  oracle : the REALISED gain of the coming slot (env.slot_gain), i.e. future fading and LoS state. UPPER-BOUND
           DIAGNOSTIC ONLY, not a deployable controller. It bounds what channel knowledge could give THIS policy class;
           it is not the global optimum (one-slot greedy, fixed objective, mean-field interference).
Optional ":nN" member suffix: N Monte Carlo draws (default 32). Each call logs its wall time (env._ja_t) and, for
diagnostics only, per candidate (prior P(LoS), P(LoS) used, realised LoS state) in env._ja_diag.
Rate model: MCS table x (1 - BLER target) except phy "shannon" (B/M log2(1 + SINR)).

reuse=False restricts to at most one UAV per channel: it isolates "better choice of which UAVs transmit" from
"co-channel reuse" (the review's point that A-X2b may have gained from the former).

Movement, power and modes come from the base slot / mode policy (matched workload); only channels are replaced.
Each call logs (value of the best assignment, value of the best no-reuse assignment, reuse used, n candidates) in
env._ja_log, which measures how often the policy's own predictor sees reuse as advantageous.
"""
import itertools
import math
import os
import time

import numpy as np

import ch_filter as CF
import env_la as LA

URG_FLOOR = 0.5


def _assignments(n, M, reuse):
    A = np.array(list(itertools.product(range(M + 1), repeat=n)), dtype=np.int64).reshape(-1, n)
    if not reuse:
        ok = [all(np.sum(row == ch) <= 1 for ch in range(1, M + 1)) for row in A]
        A = A[np.asarray(ok, bool)]
    return A


_CACHE = {}


def assignments(n, M, reuse):
    key = (n, M, reuse)
    if key not in _CACHE:
        _CACHE[key] = _assignments(n, M, reuse)
    return _CACHE[key]


def _rate(sinr_db, bw, target, shannon=False):
    if shannon:
        return bw * np.log2(1.0 + 10 ** (np.asarray(sinr_db) / 10))
    i = np.searchsorted(LA.THR, sinr_db, side="right") - 1
    return np.where(i >= 0, LA.EFF[np.clip(i, 0, None)] * bw * (1 - target), 0.0)


def _realised_los(env, k):
    if hasattr(env, "slot_los"):
        return bool(env.slot_los(k))
    s = min(env.slot, env.f_u.shape[0] - 1)
    return bool(env.f_u[s, k] < env._pathgain(env.uavs[k].pos)[0])


N_MC = 32
LOG = os.environ.get("SB_JA_LOG", "1") != "0"   # per-call diagnostics on the env; SB_JA_LOG=0 for training runs


def _stat_samples(env, cand, n_mc=N_MC, p_los_override=None):
    """n_mc x n draws of the channel power gain from the statistical model (causal: positions + model only);
    p_los_override[i] (a probability, or exactly 0 / 1 for the LoS-only oracle) replaces the prior of candidate i."""
    rng = np.random.default_rng([int(env.slot), 0x5A7])
    kf = 10 ** (env.c["rician_k_db"] / 10)
    G = np.empty((n_mc, len(cand)))
    for i, k in enumerate(cand):
        p_los, g_l, g_n = env._pathgain(env.uavs[k].pos)
        if p_los_override is not None:
            p_los = p_los_override[i]
        los = rng.random(n_mc) < p_los
        n = rng.normal(size=(n_mc, 2))
        h2 = np.abs(math.sqrt(kf / (kf + 1)) + math.sqrt(1 / (kf + 1)) * (n[:, 0] + 1j * n[:, 1]) / math.sqrt(2)) ** 2
        G[:, i] = np.where(los, g_l * h2, g_n * rng.exponential(size=n_mc))
    return G


def joint_assign(base_slot, info="stat", reuse=True, alpha=1.0, margin_db=3.0, n_mc=N_MC):
    assert info in ("stat", "los", "filt", "report", "oracle")

    def pol(env):
        t_start = time.perf_counter()
        acts = [list(a) for a in base_slot(env)]
        c, K, M = env.c, len(env.uavs), env.c["M"]
        cand = [k for k, u in enumerate(env.uavs) if u.active and u.queue]
        for a in acts:
            a[1] = 0
        if not cand:
            return [tuple(a) for a in acts]
        nz = env.noise_w()
        dec = [tk.bits for tk in env.tasks if tk.mode >= 0]
        b_und = float(np.median(dec)) if dec else 2e5
        prior = [env._pathgain(env.uavs[k].pos)[0] for k in cand]
        used = list(prior)
        if info == "stat":
            g = _stat_samples(env, cand, n_mc) / nz                   # (n_mc, n)
        elif info == "los":
            used = [1.0 if _realised_los(env, k) else 0.0 for k in cand]
            g = _stat_samples(env, cand, n_mc, used) / nz
        elif info == "filt":
            used = [CF.predict_los(env, k)[0] for k in cand]
            g = _stat_samples(env, cand, n_mc, used) / nz
        else:
            g = np.empty((1, len(cand)))
            for i, k in enumerate(cand):
                if info == "oracle":
                    g[0, i] = env.slot_gain(k) / nz
                elif hasattr(env, "la_est_snrpw_db"):
                    g[0, i] = 10 ** (env.la_est_snrpw_db(k) / 10)
                else:
                    g[0, i] = env.expected_gain(env.uavs[k].pos) / nz
        p = np.array([c["p_levels"][acts[k][2]] for k in cand])
        B = np.array([sum(tk.rem_bits if tk.mode >= 0 else b_und for tk in env.uavs[k].queue) for k in cand])
        slack = np.array([env.latest_tx_end(env.uavs[k].queue[0]) - env.t for k in cand])
        w = (1.0 / np.maximum(slack, URG_FLOOR)) ** alpha
        bw, dt, tgt = c["B_total"] / M, c["dt"], c.get("la_bler_target", 0.1)
        sh = c.get("phy") == "shannon"
        s = p[None] * g                                               # (S, n) received SNR alone
        r_alone = _rate(10 * np.log10(np.maximum(s, 1e-30)), bw, tgt, sh)
        persistent = c.get("la_interf", "bursty") == "persistent"
        f = np.ones_like(s) if persistent else np.minimum(1.0, B[None] / np.maximum(r_alone * dt, 1.0))

        def value(A):
            on = A > 0
            same = (A[:, :, None] == A[:, None, :]) & on[:, :, None] & on[:, None, :]
            same &= ~np.eye(len(cand), dtype=bool)[None]
            interf = np.einsum("aij,sj->sai", same.astype(float), f * s)           # (S, nA, n)
            shared = same.any(axis=2)[None]
            sinr = 10 * np.log10(np.maximum(s[:, None, :] / (1.0 + interf), 1e-30)) - margin_db * shared
            r = np.where(on[None], _rate(sinr, bw, tgt, sh), 0.0)
            v = (w[None, None] * np.minimum(r * dt, B[None, None])).sum(axis=2).mean(axis=0)
            return v - 1e-9 * on.sum(axis=1)

        A_all = assignments(len(cand), M, reuse)
        v_all = value(A_all)
        best = int(np.argmax(v_all))
        A_nr = assignments(len(cand), M, False)
        v_nr = float(value(A_nr).max())
        row = A_all[best]
        used_reuse = any(np.sum(row == ch) > 1 for ch in range(1, M + 1))
        if LOG:
            env.__dict__.setdefault("_ja_log", []).append((float(v_all[best]), v_nr, bool(used_reuse), len(cand)))
        for i, k in enumerate(cand):
            acts[k][1] = int(row[i])
        if LOG:
            env.__dict__.setdefault("_ja_t", []).append(time.perf_counter() - t_start)
        if LOG and env.__dict__.get("_ja_diag_on", True):
            env.__dict__.setdefault("_ja_diag", []).extend(
                (float(a), float(b), _realised_los(env, k)) for a, b, k in zip(prior, used, cand))
        return [tuple(a) for a in acts]

    pol.__name__ = f"JA:{info}:{'R' if reuse else 'NR'}:a{alpha:g}:m{margin_db:g}" + (f":n{n_mc}" if n_mc != N_MC else "")
    return pol


def family(info):
    """tuning family (train seeds): reuse x urgency exponent x SINR back-off for shared links."""
    out = [(False, a, 0.0) for a in (0.0, 1.0, 2.0)]
    out += [(True, a, m) for a in (0.0, 1.0, 2.0) for m in (0.0, 3.0, 6.0)]
    return [f"JA:{info}:{'R' if r else 'NR'}:a{a:g}:m{m:g}" for r, a, m in out]


def parse(name):
    """'JA:stat:R:a1:m3[:n8]' -> dict(info, reuse, alpha, margin_db[, n_mc])"""
    f = name.split(":")
    d = dict(info=f[1], reuse=f[2] == "R", alpha=float(f[3][1:]), margin_db=float(f[4][1:]))
    if len(f) > 5:
        d["n_mc"] = int(f[5][1:])
    return d


def from_name(name, base_slot):
    return joint_assign(base_slot, **parse(name))
