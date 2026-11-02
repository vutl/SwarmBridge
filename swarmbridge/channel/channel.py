"""Wireless corruption of per-UAV latent messages (Q1, track B).

Everything here is a deliberately simple abstraction, not a 5G stack: the point is to
isolate the bridge-reference question. Messages are assumed to have ~unit power.

Per UAV i and per spatial block:
  staleness : message = sqrt(a2) * fresh + sqrt(1-a2) * independent field, a2 = exp(-age/tau)
  fading    : effective SNR = snr * |h|^2 (|h|^2 ~ Exp(1) when Rayleigh, else 1), per block
  AWGN      : y = message + n / sqrt(snr_eff)
  erasure   : a whole block is dropped with probability p_erase
  FOV       : locations outside the UAV's field of view carry no evidence at all

Treating staleness + noise as noise, UAV i observes y_i = sqrt(a2) z + noise of variance
    s2_i = (1 - a2) + 1 / snr_eff,   so its evidence precision is  g_i = visible * a2 / s2_i.
With a unit-variance prior on z, the receiver combines by per-location LMMSE:
    z_p = sum_i (visible sqrt(a2) / s2_i) y_i / (1 + G),   G = sum_i g_i
    r   = G / (1 + G)  in [0, 1]   -> 1 - r is the residual error variance of z_p.
(A normalised average sum(r y)/sum(r) would instead *amplify* a lone low-SNR message
into a huge outlier; that broke the flow-matching baseline in the first run.)
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ChannelConfig:
    block: int = 4
    snr_db: tuple = (-10.0, 20.0)
    p_erase: tuple = (0.0, 0.6)
    ages: tuple = (0.0, 1.0, 2.0, 4.0)
    rayleigh_prob: float = 0.5
    age_tau: float = 4.0


def sample_params(cfg, B, N, device, gen, fixed=None):
    """Per-(sample, UAV) channel state. `fixed` overrides keys with a constant (for eval)."""
    fixed = fixed or {}

    def uni(lo, hi):
        return lo + (hi - lo) * torch.rand(B, N, device=device, generator=gen)

    def const(v):
        return torch.full((B, N), float(v), device=device)

    ages = torch.tensor(cfg.ages, device=device)
    return {
        "snr_db": const(fixed["snr_db"]) if "snr_db" in fixed else uni(*cfg.snr_db),
        "p_erase": const(fixed["p_erase"]) if "p_erase" in fixed else uni(*cfg.p_erase),
        "age": const(fixed["age"]) if "age" in fixed
        else ages[torch.randint(len(ages), (B, N), device=device, generator=gen)],
        "rayleigh": const(fixed["rayleigh"]).bool() if "rayleigh" in fixed
        else torch.rand(B, N, device=device, generator=gen) < cfg.rayleigh_prob,
    }


def _blocks_to_pixels(x, block):
    return F.interpolate(x, scale_factor=block, mode="nearest")


def transmit(msg, stale_src, fov, params, cfg, gen):
    """msg, stale_src: (B,N,C,H,W); fov: (B,N,1,H,W)
    -> received y, LMMSE combining weight w_i, evidence precision g_i (all zero if not received)."""
    B, N, C, H, W = msg.shape
    dev = msg.device
    a2 = torch.exp(-params["age"] / cfg.age_tau).view(B, N, 1, 1, 1)
    m = a2.sqrt() * msg + (1 - a2).sqrt() * stale_src

    Hb, Wb = H // cfg.block, W // cfg.block
    u = torch.rand(B * N, 1, Hb, Wb, device=dev, generator=gen).clamp_min(1e-6)
    h2 = (-torch.log(u)).view(B, N, 1, Hb, Wb)  # Exp(1) = |h|^2 of Rayleigh fading
    ray = params["rayleigh"].view(B, N, 1, 1, 1)
    h2 = torch.where(ray, h2, torch.ones_like(h2))
    h2 = _blocks_to_pixels(h2.view(B * N, 1, Hb, Wb), cfg.block).view(B, N, 1, H, W)
    snr_eff = (10.0 ** (params["snr_db"] / 10)).view(B, N, 1, 1, 1) * h2

    noise = torch.randn(B, N, C, H, W, device=dev, generator=gen)
    y = m + noise / snr_eff.sqrt()

    keep = torch.rand(B * N, 1, Hb, Wb, device=dev, generator=gen) >= params["p_erase"].view(B * N, 1, 1, 1)
    keep = _blocks_to_pixels(keep.float(), cfg.block).view(B, N, 1, H, W)
    vis = fov * keep

    s2 = (1 - a2) + 1 / snr_eff
    w = vis * a2.sqrt() / s2
    g = vis * a2 / s2
    return y * vis, w, g


def aggregate(y, w, g):
    """Per-location LMMSE over UAVs with a unit prior. z_p = 0, r = 0 where nothing arrived."""
    G = g.sum(1)
    z_p = (w * y).sum(1) / (1 + G)
    return z_p, G / (1 + G)
