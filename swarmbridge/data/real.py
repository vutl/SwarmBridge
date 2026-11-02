"""Cached teacher latents of CoPerception-UAV (leader setup) with the same batch contract as
SyntheticSwarm: dict(z_star, z_p, r, y).

Each grid is one UAV's region and has a single source (that UAV), so N = 1: the message is
z* itself, its FOV is the camera footprint, and the channel (erasure / AWGN / Rayleigh) acts
on it. Staleness is off: the previous latent of a moving UAV covers a different region and
would need warping first.
"""
import torch
import torch.nn.functional as F

from ..channel import ChannelConfig, aggregate, sample_params, transmit
from ..teacher.model import IGNORE, K, Decoder, miou

CONDITIONS = {
    "easy": dict(snr_db=20, p_erase=0.0, age=0, rayleigh=False),
    "mid": dict(snr_db=5, p_erase=0.2, age=0, rayleigh=True),
    "severe": dict(snr_db=-5, p_erase=0.5, age=0, rayleigh=True),
}


class CachedLatents:
    conditions = CONDITIONS

    def __init__(self, teacher_dir, device="cuda", eval_split="val"):
        self.device = device
        self.cfg = ChannelConfig(ages=(0.0,))
        load = lambda s: torch.load(f"{teacher_dir}/latents_{s}.pt")
        self.train, self.eval = load("train"), load(eval_split)
        self.C = self.train["z"].shape[1]
        self.N = 1
        ck = torch.load(f"{teacher_dir}/teacher.pt")
        st = torch.load(f"{teacher_dir}/latent_stats.pt")
        self.mu, self.sd = st["mu"].to(device), st["sd"].to(device)
        self.dec = Decoder(ck["C"]).to(device).eval()
        self.dec.load_state_dict(ck["dec"])
        self.dec.requires_grad_(False)

    def generator(self, seed):
        return torch.Generator(device=self.device).manual_seed(seed)

    def _make(self, data, idx, gen, fixed):
        z = data["z"][idx].to(self.device).float()
        fov = data["fov48"][idx].to(self.device).float()
        B = z.shape[0]
        params = sample_params(self.cfg, B, 1, self.device, gen, fixed)
        y_rx, w, g = transmit(z[:, None], torch.zeros_like(z[:, None]), fov[:, None], params, self.cfg, gen)
        z_p, r = aggregate(y_rx, w, g)
        return {"z_star": z, "z_p": z_p, "r": r, "y": data["y"][idx].to(self.device).long()}

    def batch(self, B, gen=None, fixed=None):
        idx = torch.randint(len(self.train["z"]), (B,), generator=torch.Generator().manual_seed(
            int(torch.randint(2**31, (1,), device=self.device, generator=gen).item())))
        return self._make(self.train, idx, gen, fixed)

    def eval_batch(self, n, fixed):
        idx = torch.arange(min(n, len(self.eval["z"])))
        return self._make(self.eval, idx, self.generator(1234), fixed)

    # task interface used by the Q1 runner
    @torch.no_grad()
    def predict(self, z, bs=32):
        # chunked: a full 200-grid batch through the decoder needs several GB of activations
        return torch.cat([self.dec(z[i:i + bs] * self.sd + self.mu).argmax(1) for i in range(0, len(z), bs)])

    @torch.no_grad()
    def probs(self, z, bs=32):
        """decoder class probabilities (for averaging over stochastic samples in task space)"""
        return torch.cat([self.dec(z[i:i + bs] * self.sd + self.mu).softmax(1) for i in range(0, len(z), bs)])

    def labels(self, b):
        return b["y"]

    def score(self, pred, gt):
        return miou(pred, gt)[0]

    def error_map(self, pred, gt):
        """per-pixel task error at label resolution + validity mask."""
        return pred != gt, gt != IGNORE

    def to_label_res(self, x):
        return F.interpolate(x, size=self.eval["y"].shape[-2:], mode="nearest")


__all__ = ["CachedLatents", "CONDITIONS", "K"]
