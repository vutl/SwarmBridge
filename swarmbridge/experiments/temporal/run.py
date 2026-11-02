"""Realistic temporal baseline (step 1 of reports/sb-huong-moi-2026-11-02.md).

python -m swarmbridge.experiments.temporal.run --seeds 0          # train + rollout eval
python -m swarmbridge.experiments.temporal.run --eval-only --seeds 0

The receiver keeps its own estimate of each UAV's previous latent, warps it into the current
grid with the known poses, and restores frame t from [received z_p, r, warped previous, valid].
Training never sees the clean previous latent: the previous frame is itself received at a random
budget and restored by the frozen per-frame mapping (map_c of the matched pilot, same seed).
Evaluation rolls out whole 20-frame sequences, feeding back the model's own estimates.

Compared at every frame on identical channel draws:
  map_c       per-frame mapping, no memory
  fill        map_c, but blocks with nothing received take the warped previous estimate
  temporal    the model below, rolled out on its own estimates
  temporal_oracle_prev  same model fed the warped CLEAN previous latent (upper bound only)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...bridges.backbone import UNet
from ...channel import aggregate, sample_params, transmit
from ...teacher.model import IGNORE
from ..q2_progressive.run import EVAL_CFG, random_subset
from ..q2_retx.run import to_blocks, to_cells
from ..sb_audit import run as audit
from ..sb_pilot.run import SEND_LEVELS, MatchedBevData, make_model, select_blocks

VERSION = "temporal_v1"
RES, CELL = 0.25, 8  # BEV metres per pixel; pixels per latent cell (384 -> 48)


class TemporalMapping(nn.Module):
    """U-Net mapping with the warped previous estimate as extra input; predicts the clean latent."""

    def __init__(self, channels, width=64):
        super().__init__()
        self.net = UNet(2 * channels + 2, channels, width=width)
        self.last_pred = None

    def forward(self, z_p, r, prev, valid):
        t = torch.ones(len(z_p), device=z_p.device)
        return self.net(torch.cat([z_p, r, prev, valid], 1), t)


class Sequences:
    """Latents of one split with pose-based warps from each frame's predecessor."""

    def __init__(self, split, teacher, data_root, device):
        d = torch.load(f"{teacher}/latents_{split}.pt")
        self.names = d["names"]
        self.z = d["z"].to(device).float()
        self.y = d["y"].to(device).long()
        self.fov = d["fov48"].to(device).float()
        samples = {x["token"]: x for x in json.load(open(
            f"{data_root}/town3/airsim_camera_seg/v1.0-mini/sample.json"))}
        index = {n: i for i, n in enumerate(self.names)}
        poses = [np.load(f"{data_root}/bev_town3/{split}/{n}.npz") for n in self.names]
        prev = []
        for n in self.names:
            tok, u = n.rsplit("_", 1)
            p = samples[tok]["prev"]
            prev.append(index.get(f"{p}_{u}", -1) if p else -1)
        self.prev = torch.tensor(prev, device=device)
        self.grid = torch.zeros(len(self.names), 48, 48, 2, device=device)
        G = 48
        centre = torch.arange(G, dtype=torch.float64) * CELL + (CELL - 1) / 2  # pixel coords of cell centres
        col, row = torch.meshgrid(centre, centre, indexing="xy")
        for i, j in enumerate(prev):
            if j < 0:
                continue
            o, a = (torch.from_numpy(poses[i][k]) for k in ("origin", "axes"))
            op, ap = (torch.from_numpy(poses[j][k]) for k in ("origin", "axes"))
            world = o + RES * (col[..., None] * a[0] + row[..., None] * a[1])
            rel = world - op
            pc, pr = rel @ ap[0] / RES, rel @ ap[1] / RES  # pixel coords in the previous grid
            cc, cr = (pc - (CELL - 1) / 2) / CELL, (pr - (CELL - 1) / 2) / CELL
            self.grid[i] = torch.stack([cc / (G - 1) * 2 - 1, cr / (G - 1) * 2 - 1], -1).float().to(device)
        # sequences: follow next-links from each start (scene-level splits keep them complete)
        nxt = {p: i for i, p in enumerate(prev) if p >= 0}
        self.sequences = []
        for i, p in enumerate(prev):
            if p < 0:
                seq = [i]
                while seq[-1] in nxt:
                    seq.append(nxt[seq[-1]])
                self.sequences.append(seq)

    def warp(self, previous, idx):
        """Warp previous-frame latents (aligned with idx) into the grids of frames idx."""
        has = (self.prev[idx] >= 0).float()[:, None, None, None]
        grid = self.grid[idx]
        src_fov = self.fov[self.prev[idx].clamp_min(0)]
        warped = F.grid_sample(previous, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        valid = F.grid_sample(src_fov, grid, mode="bilinear", padding_mode="zeros", align_corners=True) > 0.5
        valid = valid.float() * has * self.fov[idx]
        return warped * valid, valid


def receive(z, fov, fractions, generator):
    selected = select_blocks(to_blocks(fov) > 0, fractions, generator)
    params = sample_params(EVAL_CFG, len(z), 1, z.device, generator, None)
    return aggregate(*transmit(z[:, None], torch.zeros_like(z[:, None]), (fov * to_cells(selected))[:, None],
                               params, EVAL_CFG, generator))


class TemporalTrainData:
    def __init__(self, seq, base, device, p_no_prev=0.1):
        self.seq, self.base, self.device, self.p_no_prev = seq, base, device, p_no_prev
        self.with_prev = torch.nonzero(seq.prev >= 0).squeeze(1)
        self.levels = torch.tensor(SEND_LEVELS, device=device)

    def batch(self, size, generator):
        seq = self.seq
        pick = torch.randint(len(seq.prev), (size,), device=self.device, generator=generator)
        # a fraction of examples have no usable memory (first frame of a stream)
        forget = torch.rand(size, device=self.device, generator=generator) < self.p_no_prev
        idx = torch.where(forget, pick, self.with_prev[torch.randint(len(self.with_prev), (size,),
                                                                     device=self.device, generator=generator)])
        level = lambda: self.levels[torch.randint(len(self.levels), (size,), device=self.device, generator=generator)]
        z_p, r = receive(seq.z[idx], seq.fov[idx], level(), generator)
        p = seq.prev[idx].clamp_min(0)
        with torch.no_grad():
            pz_p, pr = receive(seq.z[p], seq.fov[p], level(), generator)
            estimate = self.base.sample(pz_p, pr, nfe=1)
        prev, valid = seq.warp(estimate, idx)
        keep = (~forget).float()[:, None, None, None]
        prev, valid = prev * keep, valid * keep
        return dict(z_p=z_p, r=r, prev=prev, valid=valid, z_star=seq.z[idx], y=seq.y[idx])


def train(seed, args, seq, decoder):
    out = Path(args.out) / f"temporal_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"ema_{args.steps}.pt"
    if final.exists():
        return
    device = args.device
    base = load_base(seed, device)
    data = TemporalTrainData(seq, base, device)
    torch.manual_seed(seed)
    model = TemporalMapping(seq.z.shape[1], args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    generator = torch.Generator(device=device).manual_seed(100 + seed)
    step, checkpoint = 0, out / "last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        ema = {k: v.to(device) for k, v in saved["ema"].items()}
        step = saved["step"]; generator.set_state(saved["generator"].cpu())
        print(f"RESUME temporal_s{seed} at {step}", flush=True)
    begin = time.perf_counter()
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(2):  # effective batch 32 as in the pilot
            b = data.batch(16, generator)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = model(b["z_p"], b["r"], b["prev"], b["valid"])
                mse = (pred.float() - b["z_star"]).square().mean()
                ce = F.cross_entropy(decoder(pred, logits=True).float(), b["y"], ignore_index=IGNORE)
                loss = (mse + ce) / 2
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema[k].lerp_(v, 1 - .999)
        step += 1
        if step % 100 == 0:
            print(f"temporal_s{seed} {step}/{args.steps} mse={mse.item():.4f} ce={ce.item():.4f} "
                  f"s/update={(time.perf_counter() - begin) / 100:.3f}", flush=True)
            begin = time.perf_counter()
        if step % 500 == 0 or step == args.steps:
            audit.atomic_checkpoint(checkpoint, dict(model=audit.cpu_state(model), optimizer=optimizer.state_dict(),
                                                     ema={k: v.cpu() for k, v in ema.items()}, step=step,
                                                     generator=generator.get_state().cpu()))
    audit.atomic_checkpoint(final, {k: v.cpu() for k, v in ema.items()})


def load_base(seed, device):
    base = make_model("map_c", 16, 64).to(device)
    base.load_state_dict(torch.load(f"runs/sb_pilot_matched_v1/map_c_s{seed}/ema_6000.pt"))
    return base.eval().requires_grad_(False)


@torch.no_grad()
def rollout(seed, args, seq, decoder):
    device = args.device
    base = load_base(seed, device)
    model = TemporalMapping(seq.z.shape[1], args.width).to(device)
    model.load_state_dict(torch.load(Path(args.out) / f"temporal_s{seed}" / f"ema_{args.steps}.pt"))
    model.eval()
    order = torch.tensor(seq.sequences, device=device)  # (S, T)
    methods = ("received", "map_c", "fill", "temporal", "temporal_oracle_prev")
    results = {}
    for fraction in SEND_LEVELS:
        conf = {m: torch.zeros(3, 3, dtype=torch.long, device=device) for m in methods}
        by_frame = {m: [] for m in ("map_c", "temporal")}
        state = {m: None for m in ("fill", "temporal")}
        for t in range(order.shape[1]):
            idx = order[:, t]
            gen = torch.Generator(device=device).manual_seed(10_000 * seed + 100 * t + int(fraction * 100))
            z_p, r = receive(seq.z[idx], seq.fov[idx], torch.full((len(idx),), fraction, device=device), gen)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                own = base.sample(z_p, r, nfe=1).float()
                outputs = {"received": z_p, "map_c": own}
                if t == 0:
                    zeros = torch.zeros_like(z_p)
                    prev_f = prev_t = prev_o = zeros
                    v_f = v_t = v_o = torch.zeros_like(r)
                else:
                    prev_f, v_f = seq.warp(state["fill"], idx)
                    prev_t, v_t = seq.warp(state["temporal"], idx)
                    prev_o, v_o = seq.warp(seq.z[order[:, t - 1]], idx)
                outputs["fill"] = torch.where((r == 0) & (v_f > 0), prev_f, own)
                outputs["temporal"] = model(z_p, r, prev_t, v_t).float()
                outputs["temporal_oracle_prev"] = model(z_p, r, prev_o, v_o).float()
            state["fill"], state["temporal"] = outputs["fill"], outputs["temporal"]
            for m in methods:
                c = audit.confusion(decoder(outputs[m]).argmax(1), seq.y[idx])
                conf[m] += c
                if m in by_frame:
                    by_frame[m].append(audit.confusion_metrics(c)["miou"])
        results[str(fraction)] = {m: audit.confusion_metrics(conf[m]) for m in methods}
        results[str(fraction)]["per_frame_miou"] = by_frame
        print(f"seed {seed} sent {fraction:.0%}: " + " | ".join(
            f"{m} {results[str(fraction)][m]['miou']:.4f}/{results[str(fraction)][m]['vehicle_iou']:.4f}"
            for m in methods), flush=True)
    return results


class Decoder:
    def __init__(self, teacher, device):
        ds = MatchedBevData(teacher, device).ds
        self.dec, self.mu, self.sd = ds.dec, ds.mu, ds.sd

    def __call__(self, z, logits=True, bs=32):
        if z.requires_grad:
            return self.dec(z * self.sd + self.mu)
        return torch.cat([self.dec(z[i:i + bs] * self.sd + self.mu) for i in range(0, len(z), bs)])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=f"runs/{VERSION}")
    p.add_argument("--teacher", default="runs/teacher")
    p.add_argument("--data-root", default="data/coperception_uav")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--eval-splits", nargs="+", default=["val", "test"])
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    decoder = Decoder(args.teacher, args.device)
    with audit.gpu_lock():
        train_seq = Sequences("train", args.teacher, args.data_root, args.device)
        evals = {s: Sequences(s, args.teacher, args.data_root, args.device) for s in args.eval_splits}
        for seed in args.seeds:
            if not args.eval_only:
                train(seed, args, train_seq, decoder)
            for split, seq in evals.items():
                path = Path(args.out) / f"temporal_s{seed}" / f"rollout_{split}.json"
                audit.atomic_json(path, rollout(seed, args, seq, decoder))
    print("TEMPORAL DONE", flush=True)


if __name__ == "__main__":
    main()
