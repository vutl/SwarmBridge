"""Review #42 (A-X7): causal Bayesian prediction of a UAV's LoS / NLoS state for the coming slot.

Inputs the BS really has: the UAV's latest DELAYED measurement report (env.la[k].report: SNR per watt incl. bias and
noise, from its own transmissions or a sounding probe), the UAV's position history (env._pos_hist; the BS knows where
its UAVs are), and the channel MODEL (LoS probability vs elevation, LoS / NLoS path gains, Rician K, and the spatial
correlation distance of the LoS process, assumed known).

1. Posterior of the LoS state AT THE REPORT, from the measured gain G: likelihoods
     LoS : G = g_L |h|^2, |h|^2 ~ Rician power with factor K   (non-central chi-square, 2 dof)
     NLoS: G = g_N x Exp(1)                                    (Rayleigh)
   (measurement bias / noise and shadowing are ignored in the likelihood; the LoS / NLoS gap is ~19 dB).
2. Prediction to the coming slot through the Gaussian-copula AR(1) of env_la: with a = Phi^-1(p_rep),
   b = Phi^-1(p_now) and rho = exp(-distance flown since the report / ch_corr_m) [x exp(-elapsed / tau)],
     P(LoS now | LoS at report)  = Phi2(a, b; rho) / p_rep
     P(LoS now | NLoS at report) = (p_now - Phi2(a, b; rho)) / (1 - p_rep).
   Without correlation (rho = 0) or without any report the prediction is the prior p_now, so a UAV whose last report
   was bad is NOT locked out forever: its prediction relaxes to the prior as it flies on (no starvation by design).
"""
import math

import numpy as np
from scipy.special import i0e, ndtr, ndtri, owens_t


def bvn_cdf(h, k, rho):
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation rho (Owen's T formula)."""
    if rho >= 1 - 1e-12:
        return float(ndtr(min(h, k)))
    if rho <= -1 + 1e-12:
        return float(max(ndtr(h) + ndtr(k) - 1.0, 0.0))
    h = h if abs(h) > 1e-12 else 1e-12
    k = k if abs(k) > 1e-12 else 1e-12
    q = math.sqrt(1.0 - rho * rho)
    v = 0.5 * ndtr(h) + 0.5 * ndtr(k) - owens_t(h, (k - rho * h) / (h * q)) - owens_t(k, (h - rho * k) / (k * q))
    if h * k < 0:
        v -= 0.5
    return float(min(max(v, 0.0), 1.0))


def los_posterior(G, p_rep, g_l, g_n, k_db):
    """P(LoS | measured linear gain G) at a position with prior p_rep and LoS / NLoS path gains g_l / g_n."""
    kf = 10 ** (k_db / 10)
    x = max(G / g_l, 1e-30)
    arg = 2.0 * math.sqrt(kf * (kf + 1.0) * x)
    log_l = math.log(kf + 1.0) - kf - (kf + 1.0) * x + math.log(max(i0e(arg), 1e-300)) + arg - math.log(g_l)
    log_n = -G / g_n - math.log(g_n)
    p = min(max(p_rep, 1e-9), 1 - 1e-9)
    lo = math.log(p) + log_l - math.log(1 - p) - log_n
    return 1.0 / (1.0 + math.exp(-min(max(lo, -700.0), 700.0)))


def predict_los(env, k):
    """causal P(LoS) of UAV k for the current / coming slot; returns (prediction, prior p_now, report age in slots)."""
    c = env.c
    p_now = env._pathgain(env.uavs[k].pos)[0]
    if c.get("ch_corr_m", 0.0) <= 0 and c.get("ch_corr_tau_s", 0.0) <= 0:
        return p_now, p_now, None
    rep = env.la[k].report(env.block_of(env.t))
    hist = env._pos_hist[k]
    if rep is None or not hist:
        return p_now, p_now, None
    s_r = min(int(math.floor(rep[0] * c["la_tb"] / c["dt"] + 1e-9)), len(hist) - 1)
    p_r, g_l, g_n = env._pathgain(hist[s_r])
    G = 10 ** (rep[3] / 10) * env.noise_w()
    post = los_posterior(G, p_r, g_l, g_n, c["rician_k_db"])
    path = hist[s_r:] + [np.asarray(env.uavs[k].pos, float)]
    d = float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))
    rho = math.exp(-d / c["ch_corr_m"]) if c["ch_corr_m"] > 0 else 1.0
    if c.get("ch_corr_tau_s", 0.0) > 0:
        rho *= math.exp(-(env.slot - s_r) * c["dt"] / c["ch_corr_tau_s"])
    pr, pn = min(max(p_r, 1e-9), 1 - 1e-9), min(max(p_now, 1e-9), 1 - 1e-9)
    j = bvn_cdf(float(ndtri(pr)), float(ndtri(pn)), rho)
    p_ll = min(max(j / pr, 0.0), 1.0)
    p_nl = min(max((pn - j) / (1 - pr), 0.0), 1.0)
    return post * p_ll + (1 - post) * p_nl, p_now, env.slot - s_r
