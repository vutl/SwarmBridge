"""Synthetic stand-in for the cached CoPerception-UAV latents (pipeline check only).

z* (oracle) is a smooth multi-channel field. Each of N UAVs sees z* only inside a random
rectangular field of view, so evidence is complementary: a location can be covered by
several UAVs, one, or none. Replace this module with a loader over the real latent cache
once the teacher exists; the dict returned by `batch` is the contract the bridges rely on.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..channel import ChannelConfig, aggregate, sample_params, transmit


class SyntheticSwarm:
    def __init__(self, C=16, H=32, W=32, N=4, cfg=None, device="cpu"):
        self.C, self.H, self.W, self.N = C, H, W, N
        self.cfg = cfg or ChannelConfig()
        self.device = device
        g = torch.Generator().manual_seed(0)
        self.mix = torch.randn(C, C, generator=g).to(device) / C**0.5
        probe = torch.Generator(device=device).manual_seed(1)
        self.scale = self._raw_field(256, probe).std().item()  # fixed -> unit power

    def _raw_field(self, B, gen):
        # Upsample a padded grid and crop the centre: bicubic edges otherwise get up to 2x
        # the variance, which breaks the stationary unit-prior the LMMSE aggregator assumes.
        d, H, W = self.device, self.H, self.W
        ph, pw = H // 4, W // 4
        Hp, Wp = H + 2 * ph, W + 2 * pw
        lo = torch.randn(B, self.C, Hp // 4, Wp // 4, device=d, generator=gen)
        mid = torch.randn(B, self.C, Hp // 2, Wp // 2, device=d, generator=gen)
        up = lambda x: F.interpolate(x, size=(Hp, Wp), mode="bicubic", align_corners=False)
        x = (up(lo) + 0.3 * up(mid))[:, :, ph:ph + H, pw:pw + W]
        return torch.einsum("dc,bchw->bdhw", self.mix, x)

    def field(self, B, gen):
        return self._raw_field(B, gen) / self.scale

    def _fov(self, B, gen):
        d, H, W, N = self.device, self.H, self.W, self.N
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, N, 1, 1, device=d, generator=gen)
        h, w = u(0.5, 0.9) * H, u(0.5, 0.9) * W
        y0, x0 = u(0, 1) * (H - h), u(0, 1) * (W - w)
        ys = torch.arange(H, device=d).view(1, 1, H, 1)
        xs = torch.arange(W, device=d).view(1, 1, 1, W)
        m = (ys >= y0) & (ys < y0 + h) & (xs >= x0) & (xs < x0 + w)
        return m.float().unsqueeze(2)  # (B,N,1,H,W)

    def batch(self, B, gen=None, fixed=None):
        """-> dict(z_star, z_p, r, r_i). `fixed` pins channel params, e.g. {'snr_db': 0}."""
        z = self.field(B * (1 + self.N), gen).view(B, 1 + self.N, self.C, self.H, self.W)
        z_star, stale = z[:, 0], z[:, 1:]
        msg = z_star.unsqueeze(1).expand(-1, self.N, -1, -1, -1)
        params = sample_params(self.cfg, B, self.N, self.device, gen, fixed)
        y, w, g = transmit(msg, stale, self._fov(B, gen), params, self.cfg, gen)
        z_p, r = aggregate(y, w, g)
        return {"z_star": z_star, "z_p": z_p, "r": r, "r_i": g / (1 + g)}

    def generator(self, seed):
        return torch.Generator(device=self.device).manual_seed(seed)

    # task interface used by the Q1 runner (labels = frozen random head on z*)
    conditions = {
        "easy": dict(snr_db=20, p_erase=0.0, age=0, rayleigh=False),
        "mid": dict(snr_db=5, p_erase=0.2, age=1, rayleigh=True),
        "severe": dict(snr_db=-5, p_erase=0.5, age=2, rayleigh=True),
    }

    @property
    def head(self):
        if not hasattr(self, "_head"):
            self._head = TaskHead(self.C).to(self.device)
        return self._head

    def eval_batch(self, n, fixed):
        return self.batch(n, self.generator(1234), fixed)

    @torch.no_grad()
    def predict(self, z):
        return self.head(z)

    def labels(self, b):
        return self.head(b["z_star"])

    def score(self, pred, gt):
        return miou(pred, gt, self.head.K)

    def error_map(self, pred, gt):
        return pred != gt, torch.ones_like(gt, dtype=torch.bool)

    def to_label_res(self, x):
        return x


class TaskHead(nn.Module):
    """Frozen 1x1 conv standing in for the frozen BEV head. Labels = head(z*) argmax."""

    def __init__(self, C, K=4):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(C, K, 1)
        g = torch.Generator().manual_seed(2)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(K, C, 1, 1, generator=g))
            self.conv.bias.zero_()
        self.requires_grad_(False)

    def forward(self, z):
        return self.conv(z).argmax(1)


def miou(pred, gt, K):
    ious = []
    for k in range(K):
        inter = ((pred == k) & (gt == k)).sum().float()
        union = ((pred == k) | (gt == k)).sum().float()
        if union > 0:
            ious.append(inter / union)
    return torch.stack(ious).mean().item()
