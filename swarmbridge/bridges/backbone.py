"""The single backbone shared by every model in the Q1 grid (same capacity, only the
training/transport paradigm changes). Small 2-level U-Net with FiLM time conditioning."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = 1000 * t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.film = nn.Linear(tdim, 2 * cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(te)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, width=64, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        w = width
        self.inc = nn.Conv2d(in_ch, w, 3, padding=1)
        self.d1 = ResBlock(w, w, tdim)
        self.down = nn.Conv2d(w, 2 * w, 3, stride=2, padding=1)
        self.d2 = ResBlock(2 * w, 2 * w, tdim)
        self.mid = ResBlock(2 * w, 2 * w, tdim)
        self.up = nn.Conv2d(2 * w, w, 3, padding=1)
        self.u1 = ResBlock(2 * w, w, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, w), nn.SiLU(), nn.Conv2d(w, out_ch, 3, padding=1))

    def forward(self, x, t):
        te = self.temb(timestep_embedding(t, self.tdim))
        h1 = self.d1(self.inc(x), te)
        h2 = self.mid(self.d2(self.down(h1), te), te)
        u = self.up(F.interpolate(h2, scale_factor=2, mode="nearest"))
        return self.out(self.u1(torch.cat([u, h1], dim=1), te))
