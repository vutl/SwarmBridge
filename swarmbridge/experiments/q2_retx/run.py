"""Q2 MVP: uncertainty-guided selective retransmission (leader setup, CoPerception-UAV latents).

1. Every UAV sends its latent once over its own channel; the leader restores it with H1
   (mapping + variance head) and gets an uncertainty map U.
2. With a retransmission budget (a fraction of the in-footprint blocks of the 5 UAVs of a
   sample, pooled), the leader asks for blocks again, chosen by one of the policies below.
3. Retransmitted blocks go through a fresh channel use (same SNR / fading type / erasure
   rate, new noise and erasures) and are LMMSE-combined with the first copy; H1 restores again.

Policies (budget pooled over the 5 UAVs of a sample unless noted):
  random | channel (lowest r) | uncertainty (highest H1 variance) |
  uncertainty_per_uav (same budget share for every UAV, top-U within each UAV) |
  oracle (highest true restoration error; upper bound, not deployable).
"uncertainty" vs "uncertainty_per_uav" measures the value of moving bits between UAVs.
Conditions: mid / severe (all UAVs share one channel state) and hetero (every UAV draws its
own SNR / erasure rate / fading from the training ranges).
Output: task mIoU vs extra bits for each policy and channel condition.

    python -m swarmbridge.experiments.q2_retx.run --teacher runs/teacher
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from ...bridges import HeteroMapping
from ...channel import aggregate, sample_params, transmit
from ...data.real import CachedLatents
from ..q1_reference.run import train

POLICIES = ("random", "channel", "uncertainty", "uncertainty_per_uav", "oracle")
CONDITIONS = ("mid", "severe", "hetero")
BLOCK = 4  # latent cells per block side = the channel's packet size


def to_blocks(x):
    """(B,1|C,H,W) -> per-block mean (B,H/BLOCK,W/BLOCK)"""
    return F.avg_pool2d(x.mean(1, keepdim=True), BLOCK).squeeze(1)


def to_cells(sel):
    """(B,h,w) block mask -> (B,1,H,W)"""
    return F.interpolate(sel[:, None].float(), scale_factor=BLOCK, mode="nearest")


@torch.no_grad()
def restore(model, z_p, r, bs=64):
    out, unc = [], []
    for i in range(0, len(z_p), bs):
        out.append(model.sample(z_p[i:i + bs], r[i:i + bs]))
        unc.append(model.uncertainty(z_p[i:i + bs], r[i:i + bs]))
    return torch.cat(out), torch.cat(unc)


def select(scores, cand, groups, frac, gen):
    """Pick the top-`frac` candidate blocks per group (sample = its 5 UAV grids)."""
    sel = torch.zeros_like(cand)
    for idx in groups:
        s, c = scores[idx].flatten(), cand[idx].flatten()
        k = int(round(frac * c.sum().item()))
        if k == 0:
            continue
        s = torch.where(c, s, torch.full_like(s, -float("inf")))
        top = torch.topk(s, k).indices
        flat = torch.zeros_like(c)
        flat[top] = True
        sel[idx] = flat.view(sel[idx].shape)
    return sel


@torch.no_grad()
def run_condition(model, ds, fixed, fracs, seed):
    data = ds.eval
    z = data["z"].to(ds.device).float()
    fov = data["fov48"].to(ds.device).float()
    y_lab = data["y"].to(ds.device).long()
    B = len(z)
    groups = defaultdict(list)
    for i, n in enumerate(data["names"]):
        groups[n.rsplit("_", 1)[0]].append(i)
    groups = [torch.tensor(v, device=ds.device) for v in groups.values()]

    gen = ds.generator(1000 + seed)
    params = sample_params(ds.cfg, B, 1, ds.device, gen, fixed)
    msg, zero, fv = z[:, None], torch.zeros_like(z[:, None]), fov[:, None]
    y1, w1, g1 = transmit(msg, zero, fv, params, ds.cfg, gen)
    z_p, r = aggregate(y1, w1, g1)
    z_hat, U = restore(model, z_p, r)
    base = ds.score(ds.predict(z_hat), y_lab)

    cand = to_blocks(fov) > 0  # blocks with any data in the footprint
    scores = {
        "random": torch.rand(cand.shape, device=ds.device, generator=gen),
        "channel": 1 - to_blocks(r),
        "uncertainty": to_blocks(U),
        "uncertainty_per_uav": to_blocks(U),
        "oracle": to_blocks((z_hat - z).pow(2)),
    }
    per_uav = [torch.tensor([i], device=ds.device) for i in range(B)]  # each grid its own group
    grid_r = (r.flatten(1).sum(1) / fov.flatten(1).sum(1).clamp_min(1))  # mean reliability in footprint
    # second channel use: same channel state, fresh noise / fading / erasures
    y2, w2, g2 = transmit(msg, zero, fv, params, ds.cfg, gen)
    bits_per_block = BLOCK * BLOCK * z.shape[1] * 8  # 8-bit quantized latent values
    res = {"base_miou": base, "n_candidate_blocks": int(cand.sum()), "curves": {}}
    lo, hi = grid_r <= grid_r.quantile(1 / 3), grid_r >= grid_r.quantile(2 / 3)
    for pol in POLICIES:
        curve = []
        for frac in fracs:
            sel = select(scores[pol], cand, per_uav if pol == "uncertainty_per_uav" else groups,
                         frac, ds.generator(7 + seed))
            share = sel.flatten(1).sum(1) / cand.flatten(1).sum(1).clamp_min(1)  # per-UAV budget share
            m = to_cells(sel)[:, None]  # (B,1,1,H,W)
            z_p2, r2 = aggregate(torch.cat([y1, y2], 1), torch.cat([w1, w2 * m], 1), torch.cat([g1, g2 * m], 1))
            z_hat2, _ = restore(model, z_p2, r2)
            curve.append({"frac": frac, "extra_kbit_per_uav": sel.sum().item() * bits_per_block / B / 1e3,
                          "miou": ds.score(ds.predict(z_hat2), y_lab),
                          # how the pooled budget lands on worst- vs best-channel UAVs (bottom/top third of r)
                          "share_worst_third": share[lo].mean().item(), "share_best_third": share[hi].mean().item()})
        res["curves"][pol] = curve
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or f"runs/q2_retx_s{a.seed}.json"
    device = "cuda"
    ds = CachedLatents(a.teacher, device)

    ckpt = Path(f"runs/q2/h1_s{a.seed}_{a.steps}.pt")
    torch.manual_seed(a.seed)
    model = HeteroMapping(ds.C, use_csi=True).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        print(f"loaded {ckpt}")
    else:
        train(model, ds, a.steps, 32, 3e-4, device, a.seed)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)

    results = {}
    for cname in CONDITIONS:
        fixed = ds.conditions.get(cname)  # hetero -> None -> per-UAV random channel states
        res = run_condition(model, ds, fixed, a.fracs, a.seed)
        results[cname] = res
        print(f"\n[{cname}] mIoU after first transmission: {res['base_miou']:.3f} "
              f"({res['n_candidate_blocks']} candidate blocks)")
        print(f"{'extra budget':>14s}" + "".join(f"{p[:13]:>14s}" for p in POLICIES))
        for k, frac in enumerate(a.fracs):
            kb = res["curves"]["random"][k]["extra_kbit_per_uav"]
            print(f"{frac:5.0%} {kb:6.1f} kb" + "".join(f"{res['curves'][p][k]['miou']:14.3f}" for p in POLICIES))
        if cname == "hetero":
            print("budget share of worst- vs best-channel third of UAVs (pooled uncertainty / channel):")
            for k, frac in enumerate(a.fracs[1:], 1):
                u, c = res["curves"]["uncertainty"][k], res["curves"]["channel"][k]
                print(f"  {frac:4.0%}: uncertainty worst {u['share_worst_third']:.2f} best {u['share_best_third']:.2f}"
                      f" | channel worst {c['share_worst_third']:.2f} best {c['share_best_third']:.2f}")
    Path(a.out).write_text(json.dumps({"args": vars(a), "results": results}, indent=2))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
