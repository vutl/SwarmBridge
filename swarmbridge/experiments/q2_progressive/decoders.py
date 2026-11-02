"""Does a bridge keep small objects (vehicles) better than regression when both get the same
upgrades? Mapping (M1) vs bridge (SB, ODE 4 steps), each in two versions:
  base     : latent MSE only (as before)
  upgraded : + lambda * segmentation cross-entropy through the frozen BEV decoder
             (task loss, as the time-domain / mel losses of SB speech enhancement)
             + residual output that keeps received evidence (z_p + sqrt(1-r) * net)
Trained on erasure up to 90% (covers partial sending); evaluated on hetero channels with
each UAV sending a random 15..100% of its blocks. Reports mIoU and vehicle IoU.

    python -m swarmbridge.experiments.q2_progressive.decoders --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...bridges import Mapping, Reference, SchrodingerBridge
from ...channel import aggregate, sample_params, transmit
from ...data.real import CachedLatents
from ...teacher.model import IGNORE, miou
from ..q2_retx.run import to_blocks, to_cells
from .run import EVAL_CFG, TRAIN_CFG, random_subset

CONFIGS = {
    "map_base": dict(kind="map", upgraded=False),
    "map_up": dict(kind="map", upgraded=True),
    "sb_base": dict(kind="sb", upgraded=False),
    "sb_up": dict(kind="sb", upgraded=True),
}


def build(cfg, C):
    if cfg["kind"] == "map":
        return Mapping(C, use_csi=True, residual=cfg["upgraded"])
    return SchrodingerBridge(C, Reference("iso", C), use_csi=True, beta0=10.0, residual=cfg["upgraded"])


def train(model, ds, steps, seed, lam, batch=32, lr=3e-4, ema=0.999):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = ds.generator(100 + seed)
    avg = [p.detach().clone() for p in model.parameters()]
    model.train()
    for step in range(steps):
        b = ds.batch(batch, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(b["z_p"], b["r"], b["z_star"])
            if lam > 0:
                logits = ds.dec(model.last_pred * ds.sd + ds.mu)
                loss = loss + lam * F.cross_entropy(logits.float(), b["y"], ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for a, p in zip(avg, model.parameters()):
                a.lerp_(p, 1 - ema)
        if step % (steps // 5) == 0 or step == steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.4f}", flush=True)
    with torch.no_grad():
        for a, p in zip(avg, model.parameters()):
            p.copy_(a)
    model.eval()


def get(name, ds, seed, steps, lam):
    ck = Path(f"runs/dec/{name}_s{seed}_{steps}.pt")
    torch.manual_seed(seed)
    model = build(CONFIGS[name], ds.C).cuda()
    if ck.exists():
        model.load_state_dict(torch.load(ck))
        model.eval()
        return model
    print(f"training {name} seed {seed}", flush=True)
    train(model, ds, steps, seed, lam if CONFIGS[name]["upgraded"] else 0.0)
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck)
    return model


@torch.no_grad()
def restore(model, z_p, r, nfe=4, bs=64):
    """bridges: deterministic nfe-step path (nfe=1 is the SBM-style one-step predictor)"""
    if isinstance(model, SchrodingerBridge):
        f = lambda a, b: model.sample(a, b, nfe, stochastic=False)
    else:
        f = lambda a, b: model.sample(a, b)
    return torch.cat([f(z_p[i:i + bs], r[i:i + bs]) for i in range(0, len(z_p), bs)])


def variants(models):
    """(label, model, nfe): mappings once, bridges with 4-step ODE and one step"""
    for n, m in models.items():
        if isinstance(m, SchrodingerBridge):
            yield n, m, 4
            yield n + "@1", m, 1
        else:
            yield n, m, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the segmentation loss (upgraded)")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.15, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--out", default="runs/decoders.json")
    a = ap.parse_args()
    ds = CachedLatents(a.teacher, "cuda")
    ds.cfg = TRAIN_CFG
    z, fov, y = ds.eval["z"].cuda().float(), ds.eval["fov48"].cuda().float(), ds.eval["y"].cuda().long()
    cand = to_blocks(fov) > 0
    res = {"miou": {}, "vehicle": {}}
    for seed in a.seeds:
        models = {n: get(n, ds, seed, a.steps, a.lam) for n in CONFIGS}
        params = sample_params(EVAL_CFG, len(z), 1, "cuda", ds.generator(3000 + seed), None)
        for frac in a.fracs:
            sel = random_subset(cand, frac, ds.generator(21 + seed))
            rx = transmit(z[:, None], torch.zeros_like(z[:, None]), (fov * to_cells(sel))[:, None],
                          params, EVAL_CFG, ds.generator(31 + seed))
            z_p, r = aggregate(*rx)
            for n, m, nfe in variants(models):
                mi, per = miou(ds.predict(restore(m, z_p, r, nfe)), y)
                res["miou"].setdefault(str(frac), {}).setdefault(n, []).append(mi)
                res["vehicle"].setdefault(str(frac), {}).setdefault(n, []).append(per[2])
        print(f"seed {seed} evaluated", flush=True)

    for title, table in (("mIoU", res["miou"]), ("vehicle IoU", res["vehicle"])):
        names = list(next(iter(table.values())))
        print(f"\n{title} (mean±std over {len(a.seeds)} seeds; '@1' = bridge one step, else bridge = 4-step ODE)")
        print(f"{'sent':>6s}" + "".join(f"{n:>15s}" for n in names))
        for frac, d in table.items():
            print(f"{float(frac):6.0%}" + "".join(f"{np.mean(d[n]):9.3f}±{np.std(d[n], ddof=1) if len(d[n]) > 1 else 0:.3f}" for n in names))
        print("paired differences, mean (std) over seeds:")
        for frac, d in table.items():
            def diff(x, yk):
                v = np.array(d[x]) - np.array(d[yk])
                return f"{v.mean():+.4f} ({v.std(ddof=1) if len(v) > 1 else 0:.4f})"
            print(f"{float(frac):6.0%}  sb-map base {diff('sb_base', 'map_base')}  sb-map up {diff('sb_up', 'map_up')}  "
                  f"sb@1-map up {diff('sb_up@1', 'map_up')}  map up-base {diff('map_up', 'map_base')}  "
                  f"sb up-base {diff('sb_up', 'sb_base')}")
    Path(a.out).write_text(json.dumps({"args": vars(a), **res}, indent=2))


if __name__ == "__main__":
    main()
