"""A0b: re-creation of the setting of Chen et al., "Diffusion Schroedinger Bridge-Based Channel Estimation for MIMO ISAC
Systems" (IEEE WCL 2025) to test its central claim: a bridge-trained network (VE-DSB, one-step ODE from the LS estimate)
beats the same network trained as a plain regressor.

Setting (paper p.2, p.4; details not stated in the paper are marked):
  sensing channel H = sum_{l=1}^{L} alpha_l a(theta_r,l) a(theta_t,l)^H, N = 32 ULA (half wavelength), L = 3,
  alpha_l ~ CN(0, 1) (not stated), angles uniform in (-pi/2, pi/2) (paper: (0, 2pi) for the TCCN version), normalised so
  E||H||_F^2 = N^2; LS estimate H_ls = H + Z with orthogonal pilots, Z ~ CN(0, 1/SNR) per entry; SNR -10..10 dB;
  7 500 train / 2 500 test samples (paper); one model per SNR (not stated).
Networks: the same small CNN (4 conv layers, 64 ch) on [Re, Im] of the 2-D DFT (angular domain) for every learned method;
the paper uses a UNet with attention (not reproduced).
  reg      plain regression H_ls -> H (MSE)                      (paper's "UNet without bridge" baseline)
  dsb1     bridge-trained data predictor (x_t = (1-t) H + t H_ls + sigma sqrt(t(1-t)) eps, predict H), ONE step from t=1
  bb10     same bridge, stochastic 10-step sampler (Brownian-bridge style, as the TCCN paper)
  ls, lmmse (sample covariance of vec(H) from the train set, 1024 x 1024; complex)
Metric: NMSE (dB) on the test set; 3 seeds. Usage: python a0_sbisac.py OUT [--device cuda] [--steps 6000]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N, L = 32, 3
SNRS = [-10, -5, 0, 5, 10]
SIGMA = 1.0


def steer(theta):
    n = np.arange(N)
    return np.exp(1j * np.pi * np.outer(np.sin(theta), n)) / np.sqrt(N)       # (L, N)


def gen(rng, n):
    H = np.zeros((n, N, N), complex)
    for i in range(n):
        tr, tt = rng.uniform(-np.pi / 2, np.pi / 2, L), rng.uniform(-np.pi / 2, np.pi / 2, L)
        a = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
        H[i] = np.einsum("l,li,lj->ij", a, steer(tr), steer(tt).conj())
    return H * (N / np.sqrt(np.mean(np.sum(np.abs(H) ** 2, (1, 2)))))           # E||H||^2 = N^2


def to_t(H, dev):
    X = np.fft.fft2(H, norm="ortho")                                            # angular domain
    return torch.tensor(np.stack([X.real, X.imag], 1), dtype=torch.float32, device=dev)


def from_t(X):
    X = X.detach().cpu().numpy()
    return np.fft.ifft2(X[:, 0] + 1j * X[:, 1], norm="ortho")


class CNN(nn.Module):
    def __init__(self, cin=2, ch=64, temb=False):
        super().__init__()
        self.temb = temb
        c0 = cin + (1 if temb else 0)
        self.f = nn.Sequential(nn.Conv2d(c0, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
                               nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, 2, 3, padding=1))

    def forward(self, x, t=None):
        if self.temb:
            x = torch.cat([x, t[:, None, None, None].expand(-1, 1, *x.shape[2:])], 1)
        return x[:, :2] + self.f(x)                                              # residual on the input estimate


def nmse_db(Hh, H):
    return float(10 * np.log10(np.sum(np.abs(Hh - H) ** 2) / np.sum(np.abs(H) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--device", default="cpu"); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    dev = args.device
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        Htr, Hte = gen(rng, 7500), gen(rng, 2500)
        v = Htr.reshape(len(Htr), -1)
        mu, C = v.mean(0), np.cov(v.T)
        for snr_db in SNRS:
            nv = 10 ** (-snr_db / 10)
            g = np.random.default_rng([seed, snr_db + 50])
            Ztr = np.sqrt(nv / 2) * (g.standard_normal(Htr.shape) + 1j * g.standard_normal(Htr.shape))
            Zte = np.sqrt(nv / 2) * (g.standard_normal(Hte.shape) + 1j * g.standard_normal(Hte.shape))
            Ltr, Lte = Htr + Ztr, Hte + Zte
            res = {"ls": nmse_db(Lte, Hte)}
            K = C @ np.linalg.inv(C + nv * np.eye(N * N))
            Hl = (mu[None] + (Lte.reshape(len(Lte), -1) - mu[None]) @ K.T).reshape(Hte.shape)
            res["lmmse"] = nmse_db(Hl, Hte)
            Xtr, X1tr, X1te = to_t(Htr, dev), to_t(Ltr, dev), to_t(Lte, dev)
            for kind in ("reg", "bridge"):
                torch.manual_seed(seed)
                net = CNN(temb=(kind == "bridge")).to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
                t0 = time.time()
                for it in range(args.steps):
                    idx = torch.randint(0, len(Xtr), (64,), device=dev)
                    x0, x1 = Xtr[idx], X1tr[idx]
                    if kind == "reg":
                        loss = ((net(x1) - x0) ** 2).mean()
                    else:
                        t = torch.rand(len(idx), device=dev)
                        tt = t[:, None, None, None]
                        xt = (1 - tt) * x0 + tt * x1 + SIGMA * (tt * (1 - tt)).sqrt() * torch.randn_like(x0)
                        loss = ((net(xt, t) - x0) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
                net.eval()
                with torch.no_grad():
                    if kind == "reg":
                        res["reg"] = nmse_db(from_t(torch.cat([net(X1te[i:i + 500]) for i in range(0, len(X1te), 500)])), Hte)
                    else:
                        one = torch.cat([net(X1te[i:i + 500], torch.ones(len(X1te[i:i + 500]), device=dev))
                                         for i in range(0, len(X1te), 500)])
                        res["dsb1"] = nmse_db(from_t(one), Hte)
                        gtor = torch.Generator(device=dev).manual_seed(seed)
                        outs = []
                        for i in range(0, len(X1te), 500):
                            x = X1te[i:i + 500].clone()
                            ts = np.linspace(1, 0, 11)
                            for a, b in zip(ts[:-1], ts[1:]):
                                x0 = net(x, torch.full((len(x),), float(a), device=dev))
                                if b == 0:
                                    x = x0
                                    break
                                x = x0 + (b / a) * (x - x0) + np.sqrt(SIGMA ** 2 * b * (a - b) / a) * torch.randn(x.shape, generator=gtor, device=dev)
                            outs.append(x)
                        res["bb10"] = nmse_db(from_t(torch.cat(outs)), Hte)
                res[f"{kind}_train_s"] = round(time.time() - t0, 1)
            rows.append(dict(seed=seed, snr_db=snr_db, **res))
            print(seed, snr_db, {k: round(v, 2) for k, v in res.items()}, flush=True)
            (out / "a0_sbisac.json").write_text(json.dumps(dict(steps=args.steps, rows=rows), indent=1))


if __name__ == "__main__":
    main()
