"""Paired bridge recipes used only by the isolated SB implementation audit.

VE cumulative variance follows Jukic et al. (Interspeech 2024), Table 1.
This is a BEV/CIFAR adaptation, not a reproduction of the speech benchmark.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn

from .backbone import UNet


@dataclass(frozen=True)
class VarianceClock:
    kind: str = "ve"
    c: float = 0.4
    k: float = 2.6
    q: float = 1.0

    def __post_init__(self):
        if self.kind not in ("ve", "brownian") or self.c <= 0 or self.k <= 1 or self.q <= 0:
            raise ValueError("Invalid variance-clock parameters")

    def cumulative(self, t):
        if self.kind == "brownian":
            return self.q * t
        a = 2 * math.log(self.k)
        return self.c * torch.expm1(a * t) / a

    def coefficients(self, t):
        eta = self.cumulative(t)
        total = self.cumulative(torch.ones_like(t))
        w_source = eta / total
        variance = eta * (total - eta).clamp_min(0) / total
        return 1 - w_source, w_source, variance

    def marginal(self, clean, source, t, noise):
        u, w, variance = (v.view(-1, 1, 1, 1) for v in self.coefficients(t))
        return u * clean + w * source + variance.sqrt() * noise

    def reverse(self, state, clean_prediction, t, s, *, noise=None):
        """Exact pinned reverse kernel in variance-clock time.

        Replacing the unknown clean endpoint by its estimate is a finite-step
        approximation. noise=None is a mean-path heuristic, not a PF-ODE.
        """
        eta_t, eta_s = self.cumulative(t), self.cumulative(s)
        ratio = eta_s / eta_t
        variance = eta_s * (1 - ratio).clamp_min(0)
        result = ratio * state + (1 - ratio) * clean_prediction
        if noise is not None:
            result = result + variance.sqrt() * noise
        return result


class AuditDenoiser(nn.Module):
    def __init__(self, channels, *, bridge, condition_source=False, csi=False,
                 residual=False, endpoint_fraction=0.1, clock=None, width=64,
                 prediction="clean"):
        super().__init__()
        if not 0 <= endpoint_fraction <= 1:
            raise ValueError("endpoint_fraction must lie in [0, 1]")
        if prediction not in ("clean", "noise"):
            raise ValueError(prediction)
        if residual and prediction != "clean":
            raise ValueError("Residual output is defined only for clean prediction")
        self.bridge = bridge
        self.condition_source, self.csi, self.residual = condition_source, csi, residual
        self.endpoint_fraction, self.clock = endpoint_fraction, clock or VarianceClock()
        self.prediction = prediction
        inputs = channels * (1 + int(condition_source)) + int(csi)
        self.net = UNet(inputs, channels, width=width)
        self.last_pred = None

    def forward(self, state, source, reliability, t):
        inputs = [state]
        if self.condition_source:
            inputs.append(source)
        if self.csi:
            inputs.append(reliability)
        raw = self.net(torch.cat(inputs, dim=1), t)
        if self.prediction == "noise":
            scale = self.clock.cumulative(t).sqrt().view(-1, 1, 1, 1)
            predicted = state - scale * raw
        elif self.residual:
            predicted = source + (1 - reliability).clamp_min(0).sqrt() * raw
        else:
            predicted = raw
        return predicted, raw

    def loss(self, source, reliability, clean, *, generator, loss_mask=None):
        batch = source.shape[0]
        if self.bridge:
            # Match Jukic's positive training-time cutoff; mix exact endpoint examples.
            t = 1e-4 + (1 - 1e-4) * torch.rand(batch, device=source.device, generator=generator)
            endpoint = torch.rand(batch, device=source.device, generator=generator) < self.endpoint_fraction
            t = torch.where(endpoint, torch.ones_like(t), t)
            noise = torch.randn(clean.shape, device=clean.device, generator=generator)
            state = self.clock.marginal(clean, source, t, noise)
        else:
            t = torch.ones(batch, device=source.device)
            state = source
        self.last_pred, raw = self(state, source, reliability, t)
        if self.prediction == "noise":
            scale = self.clock.cumulative(t).sqrt().view(-1, 1, 1, 1)
            target = (state - clean) / scale
            error = (raw.float() - target).square()
        else:
            error = (self.last_pred.float() - clean).square()
        if loss_mask is not None:
            return (error * loss_mask).sum() / (loss_mask.sum() * clean.shape[1]).clamp_min(1)
        return error.mean()

    @torch.no_grad()
    def sample(self, source, reliability, *, nfe=1, stochastic=True, generator=None,
               known=None, clip=False):
        if nfe < 1:
            raise ValueError("nfe must be positive")
        if known is not None:
            known = known.to(dtype=source.dtype)
        state = source
        times = torch.linspace(1, 0, nfe + 1, device=source.device)
        for index in range(nfe if self.bridge else 1):
            t, s = times[index], times[index + 1]
            prediction, _ = self(state, source, reliability, t.expand(len(source)))
            if clip:
                prediction = prediction.clamp(-1, 1)
            if s == 0 or not self.bridge:
                state = prediction
            else:
                noise = torch.randn(state.shape, device=state.device, generator=generator) if stochastic else None
                state = self.clock.reverse(state, prediction, t, s, noise=noise)
            if known is not None:
                state = known * source + (1 - known) * state
        return state
