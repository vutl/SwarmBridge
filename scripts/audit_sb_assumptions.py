r"""CPU-only research diagnostics; no training or checkpoint changes.

Run from the repository root:
    .venv\Scripts\python.exe -m scripts.audit_sb_assumptions

The Gaussian experiment uses an exact conditional-mean estimator, NOT a learned
network. It separates finite-step sampling error from optimization error.
"""

import json
from pathlib import Path

import torch
from torch import nn

from swarmbridge.bridges import Reference, SchrodingerBridge
from swarmbridge.bridges.models import HeteroMapping


class GaussianConditionalMean(nn.Module):
    """E[X0 | Xt, y] for X0|y ~ N(y/2, P), P=1/2.

    Xt=(1-t)X0+t*y+sqrt(q*t*(1-t))*epsilon, with q constant.
    The experiment samples at y=0, so its target conditional variance is P.
    """

    def __init__(self, q, posterior_variance=0.5):
        super().__init__()
        self.q, self.p = q, posterior_variance

    def forward(self, inputs, t):
        xt, y = inputs[:, :1], inputs[:, 1:2]
        t = t.view(-1, 1, 1, 1)
        mean = 0.5 * y
        path_mean = (1 - t) * mean + t * y
        gain = self.p / ((1 - t) * self.p + self.q * t)
        return mean + gain * (xt - path_mean)


def exact_terminal_variance(q, nfe, p=0.5):
    """Independent scalar variance recursion for the implemented plug-in sampler."""
    variance = 0.0
    for k in range(nfe):
        t, s = 1 - k / nfe, 1 - (k + 1) / nfe
        gain = p / ((1 - t) * p + q * t)
        coefficient = s / t + (1 - s / t) * gain
        variance = coefficient**2 * variance + q * s * (t - s) / t
    return variance


def diagnose():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = HeteroMapping(2, True, width=8)
    zp = torch.randn(2, 2, 8, 8)
    reliability = torch.rand(2, 1, 8, 8)
    target = torch.randn_like(zp)
    mean, logvar = model._forward(zp, reliability)
    variance_nll = 0.5 * (
        logvar + (target - mean.detach()).square() * torch.exp(-logvar)
    ).mean()
    gradient = torch.autograd.grad(variance_nll, model.net.inc.weight)[0]

    n = 32768
    y = torch.zeros(n, 1, 1, 1)
    r = torch.full_like(y, 0.5)
    rows = []
    for q in (1.0, 10.0):
        bridge = SchrodingerBridge(1, Reference("iso", 1), True, beta0=q, width=8)
        bridge.net = GaussianConditionalMean(q)
        for nfe in (1, 4, 16, 64, 256):
            samples = bridge.sample(y, r, nfe, torch.Generator().manual_seed(123))
            rows.append({
                "q": q,
                "nfe": nfe,
                "target_posterior_variance": 0.5,
                "exact_implemented_sampler_variance": exact_terminal_variance(q, nfe),
                "monte_carlo_mean": samples.mean().item(),
                "monte_carlo_variance": samples.var(unbiased=False).item(),
            })

    return {
        "device": "cpu",
        "trained_any_model": False,
        "torch_version": torch.__version__,
        "heteroscedastic_variance_gradient_into_shared_trunk_norm": gradient.norm().item(),
        "faithful_shared_trunk_gradient_should_be_zero": True,
        "deterministic_pinned_bridge_counterexample": {
            "q": 1.0, "t": 0.75, "s": 0.5,
            "required_forward_variance": 0.25,
            "posterior_mean_update_variance": (0.5 / 0.75)**2 * 0.75 * 0.25,
            "scope": "Turning off reverse-kernel noise does not preserve pinned marginals; this is not proof that every deterministic heuristic is useless.",
        },
        "gaussian_conditional_sampling": rows,
        "scope": "Exact toy estimator and scalar moments only; these diagnostics do not establish Town3 performance or that learned uncertainty is calibrated.",
    }


if __name__ == "__main__":
    result = diagnose()
    output = Path("reports/metadata/sb-audit-checks-2026-10-30.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
