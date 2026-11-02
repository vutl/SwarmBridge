"""Sanity checks for the parts of Q1 that are easy to get silently wrong.

    python -m pytest tests -q        (or: python tests/test_q1_core.py)
"""
import torch

from swarmbridge.bridges import Reference, SchrodingerBridge
from swarmbridge.data import SyntheticSwarm


def test_bridge_sampler_matches_forward_marginals():
    """With an oracle z_hat = z*, every state the sampler visits at time s must follow the
    forward marginal N((1-s) z* + s z_p, q s (1-s)), and the last step must land on z*."""
    C, n, beta0 = 2, 200_000, 1.0
    z_star = torch.randn(n, C, 1, 1)
    z_p = torch.randn(n, C, 1, 1)
    r = torch.zeros(n, 1, 1, 1)
    sb = SchrodingerBridge(C, Reference("iso", C), use_csi=False, beta0=beta0, width=8)
    seen = {}

    class Oracle(torch.nn.Module):
        def forward(self, x, t):
            seen[round(t[0].item(), 4)] = x[:, :C].clone()
            return z_star

    sb.net = Oracle()
    out = sb.sample(z_p, r, nfe=4)
    assert torch.allclose(out, z_star)
    for s in (0.75, 0.5, 0.25):
        resid = seen[s] - ((1 - s) * z_star + s * z_p)
        assert abs(resid.mean().item()) < 0.01, s
        assert abs(resid.var().item() - beta0 * s * (1 - s)) < 0.01, (s, resid.var().item())


def test_reference_same_noise_budget():
    ds = SyntheticSwarm(device="cpu")
    g = ds.generator(0)
    batches = [ds.batch(32, g) for _ in range(4)]
    r = torch.cat([b["r"] for b in batches])
    for kind in ("iso", "reliability", "empirical"):
        ref = Reference(kind, ds.C)
        ref.fit(batches)
        assert abs(ref(r).mean().item() - 1.0) < 0.02, kind


def test_empirical_reference_tracks_residual():
    """R2 must put more noise where evidence is missing (low r) than where it is good."""
    ds = SyntheticSwarm(device="cpu")
    g = ds.generator(0)
    ref = Reference("empirical", ds.C)
    ref.fit([ds.batch(64, g) for _ in range(4)])
    lo, hi = ref(torch.zeros(1, 1, 1, 1)).mean(), ref(torch.full((1, 1, 1, 1), 0.95)).mean()
    assert lo > 3 * hi, (lo.item(), hi.item())


def test_channel_reliability_ordering():
    ds = SyntheticSwarm(device="cpu")
    easy = ds.batch(64, ds.generator(1), dict(snr_db=20, p_erase=0.0, age=0, rayleigh=False))
    hard = ds.batch(64, ds.generator(1), dict(snr_db=-5, p_erase=0.5, age=2, rayleigh=True))
    for b in (easy, hard):
        assert 0 <= b["r"].min() and b["r"].max() <= 1
    assert easy["r"].mean() > hard["r"].mean() + 0.3
    assert (hard["z_p"] - hard["z_star"]).pow(2).mean() > (easy["z_p"] - easy["z_star"]).pow(2).mean()


def test_lmmse_reliability_is_calibrated():
    """1 - r must equal the actual residual variance of z_p (what R2/PRISM-style refs assume),
    and low-SNR evidence must be shrunk, not amplified into outliers."""
    ds = SyntheticSwarm(device="cpu")
    b = ds.batch(256, ds.generator(3))
    err = (b["z_star"] - b["z_p"]).pow(2).mean(1)
    r = b["r"].squeeze(1)
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        m = (r >= lo) & (r < lo + 0.2)
        if m.sum() > 1000:
            assert abs(err[m].mean() - (1 - r[m]).mean()) < 0.05, (lo, err[m].mean(), 1 - r[m].mean())
    assert b["z_p"].abs().max() < 10


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
