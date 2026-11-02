"""Mapping / flow matching / Schrödinger bridge on the same UNet, z_p -> z*.

Every model is conditioned on z_p explicitly (input channel), and optionally on the
reliability map r ("CSI in network"). Conditioning on z_p is what keeps the bridge's
samples tied to p(z* | z_p) rather than a Markovian projection that loses the coupling
(cf. augmented bridge matching) -- required before reading Q1b uncertainty off samples.

All share the interface:
    loss(z_p, r, z_star) -> scalar
    sample(z_p, r, nfe, gen=None) -> z_hat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import UNet


class _Base(nn.Module):
    def __init__(self, C, use_csi, width, extra_in, residual=False):
        super().__init__()
        self.use_csi = use_csi
        self.residual = residual
        self.net = UNet(extra_in * C + int(use_csi), C, width)
        self.last_pred = None  # clean-latent prediction of the last loss() call (for task losses)

    def _inp(self, *xs, r):
        return torch.cat([*xs, r] if self.use_csi else list(xs), dim=1)

    def _out(self, raw, z_p, r):
        """residual=True keeps what was received: z_p is the LMMSE estimate whose error variance
        is 1 - r, so the net only predicts a correction scaled by that std (none where r ~ 1)."""
        return z_p + (1 - r).clamp_min(0).sqrt() * raw if self.residual else raw


class Mapping(_Base):
    """Direct regression. aug: 'none' | 'iso' | 'aniso' (noise shaped by `ref`, no bridge)."""

    def __init__(self, C, use_csi, aug="none", ref=None, aug_std=0.5, width=64, residual=False):
        super().__init__(C, use_csi, width, extra_in=1, residual=residual)
        assert aug in ("none", "iso", "aniso") and (aug != "aniso" or ref is not None)
        self.aug, self.ref, self.aug_std = aug, ref, aug_std

    def loss(self, z_p, r, z_star):
        x = z_p
        if self.aug != "none":
            lam = self.ref(r) if self.aug == "aniso" else torch.ones_like(r)
            s = self.aug_std * torch.rand(z_p.shape[0], 1, 1, 1, device=z_p.device)
            x = z_p + s * lam.sqrt() * torch.randn_like(z_p)
        t = torch.ones(z_p.shape[0], device=z_p.device)
        self.last_pred = self._out(self.net(self._inp(x, r=r), t), z_p, r)
        return F.mse_loss(self.last_pred, z_star)

    @torch.no_grad()
    def sample(self, z_p, r, nfe=1, gen=None):
        t = torch.ones(z_p.shape[0], device=z_p.device)
        return self._out(self.net(self._inp(z_p, r=r), t), z_p, r)


class HeteroMapping(_Base):
    """Mapping + per-element variance head (Q1b baseline: uncertainty without a bridge).

    "Faithful" heteroscedastic training (Stirn et al., AISTATS 2023): the mean gets the plain
    MSE loss exactly like Mapping, the variance head gets the Gaussian NLL with the mean
    detached, so learning the variance cannot degrade the mean.
    """

    def __init__(self, C, use_csi, width=64):
        super().__init__(C, use_csi, width, extra_in=1)
        self.net = type(self.net)(C + int(use_csi), 2 * C, width)
        self.C = C

    def _forward(self, z_p, r):
        t = torch.ones(z_p.shape[0], device=z_p.device)
        mu, logvar = self.net(self._inp(z_p, r=r), t).split(self.C, dim=1)
        return mu, logvar.clamp(-10, 5)

    def loss(self, z_p, r, z_star):
        mu, logvar = self._forward(z_p, r)
        nll = 0.5 * (logvar + (z_star - mu.detach()).pow(2) * torch.exp(-logvar)).mean()
        return F.mse_loss(mu, z_star) + nll

    @torch.no_grad()
    def sample(self, z_p, r, nfe=1, gen=None):
        return self._forward(z_p, r)[0]

    @torch.no_grad()
    def uncertainty(self, z_p, r):
        return self._forward(z_p, r)[1].exp()


class FlowMatching(_Base):
    """Standard (Markovian) OT-CFM path z_p (t=0) -> z* (t=1), velocity target z* - z_p.

    Deliberately NOT fed z_p as a separate input: with a near-straight path, (x_t, z_p, t)
    gives the velocity away as (x_t - z_p) / t, the net learns that shortcut (train loss ~0)
    and is untrained exactly at t=0 where sampling starts. z_p is already the start point.
    """

    def __init__(self, C, use_csi, sigma=0.05, width=64):
        super().__init__(C, use_csi, width, extra_in=1)
        self.sigma = sigma

    def loss(self, z_p, r, z_star):
        t = torch.rand(z_p.shape[0], device=z_p.device)
        tb = t.view(-1, 1, 1, 1)
        x = (1 - tb) * z_p + tb * z_star + self.sigma * torch.randn_like(z_p)
        return F.mse_loss(self.net(self._inp(x, r=r), t), z_star - z_p)

    @torch.no_grad()
    def sample(self, z_p, r, nfe=1, gen=None):
        x = z_p + self.sigma * torch.randn(z_p.shape, device=z_p.device, generator=gen)
        ts = torch.linspace(0, 1, nfe + 1, device=z_p.device)
        for k in range(nfe):
            t = ts[k].expand(z_p.shape[0])
            x = x + (ts[k + 1] - ts[k]) * self.net(self._inp(x, r=r), t)
        return x


class SchrodingerBridge(_Base):
    """Tractable paired bridge (I2SB/SBM style) with reference geometry Lambda(r).

    Forward marginal, t=0 at z*, t=1 at z_p, q = beta0 * Lambda(r):
        Z_t = (1 - t) z* + t z_p + sqrt(t (1 - t) q) eps
    Network predicts z* (data prediction). Sampling steps t -> s < t with the
    Brownian-bridge posterior between z_hat (time 0) and x_t (time t):
        x_s = z_hat + (s / t)(x_t - z_hat) + sqrt(q s (t - s) / t) eps
    nfe=1 collapses to the deterministic one-step predictor z_hat = G(z_p, t=1).
    """

    def __init__(self, C, ref, use_csi, beta0=1.0, width=64, t1_frac=0.0, residual=False):
        super().__init__(C, use_csi, width, extra_in=2, residual=residual)
        self.ref, self.beta0, self.t1_frac = ref, beta0, t1_frac

    def q(self, r):
        return self.beta0 * self.ref(r)

    def loss(self, z_p, r, z_star):
        t = torch.rand(z_p.shape[0], device=z_p.device)
        # uniform t rarely visits t = 1, the only point one-step inference uses
        t = torch.where(torch.rand_like(t) < self.t1_frac, torch.ones_like(t), t)
        tb = t.view(-1, 1, 1, 1)
        std = (tb * (1 - tb) * self.q(r)).sqrt()
        x = (1 - tb) * z_star + tb * z_p + std * torch.randn_like(z_p)
        self.last_pred = self._out(self.net(self._inp(x, z_p, r=r), t), z_p, r)
        return F.mse_loss(self.last_pred, z_star)

    @torch.no_grad()
    def sample(self, z_p, r, nfe=1, gen=None, stochastic=True):
        """stochastic=False drops the posterior noise (deterministic, ODE-like path)."""
        q = self.q(r)
        ts = torch.linspace(1, 0, nfe + 1, device=z_p.device)
        x = z_p
        for k in range(nfe):
            t, s = ts[k], ts[k + 1]
            z_hat = self._out(self.net(self._inp(x, z_p, r=r), t.expand(z_p.shape[0])), z_p, r)
            if s <= 0:
                return z_hat
            x = z_hat + (s / t) * (x - z_hat)
            if stochastic:
                eps = torch.randn(z_p.shape, device=z_p.device, generator=gen)
                x = x + (q * s * (t - s) / t).sqrt() * eps
        return x
