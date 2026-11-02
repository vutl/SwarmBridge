"""Train and evaluate the Q1 grid.

    python -m swarmbridge.experiments.q1_reference.run --data real --teacher runs/teacher
    python -m swarmbridge.experiments.q1_reference.run --data synthetic --steps 2000
    python -m swarmbridge.experiments.q1_reference.run --models M1 S1 S3 --steps 500

--data synthetic is a PIPELINE CHECK only. --data real uses cached CoPerception-UAV teacher
latents (leader setup: each UAV's region has one source, corrupted by its own channel).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from ...bridges import SchrodingerBridge
from ...uncertainty import auroc, binned_auroc
from .grid import GRID, build


def train(model, ds, steps, batch, lr, device, seed=0, ema=0.0):
    """ema > 0: keep an exponential moving average of the weights and end with it loaded."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = ds.generator(100 + seed)
    avg = [p.detach().clone() for p in model.parameters()] if ema > 0 else None
    model.train()
    for step in range(steps):
        b = ds.batch(batch, gen)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = model.loss(b["z_p"], b["r"], b["z_star"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if avg is not None:
            with torch.no_grad():
                for a, p in zip(avg, model.parameters()):
                    a.lerp_(p, 1 - ema)
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.4f}", flush=True)
    if avg is not None:
        with torch.no_grad():
            for a, p in zip(avg, model.parameters()):
                p.copy_(a)
    model.eval()


def sample_batched(model, b, nfe, gen, bs=64):
    return torch.cat([model.sample(b["z_p"][i:i + bs], b["r"][i:i + bs], nfe, gen)
                      for i in range(0, len(b["z_p"]), bs)])


def uncertainty_map(model, b, nfes, n_samples, ds):
    """Per-element uncertainty and the prediction it belongs to: sample spread for bridges,
    the variance head for HeteroMapping. None for models without uncertainty."""
    if hasattr(model, "uncertainty"):
        U = torch.cat([model.uncertainty(b["z_p"][i:i + 64], b["r"][i:i + 64]) for i in range(0, len(b["z_p"]), 64)])
        return U, sample_batched(model, b, 1, None)
    if isinstance(model, SchrodingerBridge):
        g = ds.generator(99)
        zs = torch.stack([sample_batched(model, b, max(nfes), g) for _ in range(n_samples)])
        return zs.var(0), zs.mean(0)
    return None


@torch.no_grad()
def evaluate(model, ds, n_eval, nfes, n_samples):
    out = {}
    for cname, fixed in ds.conditions.items():
        b = ds.eval_batch(n_eval, fixed)
        gt = ds.labels(b)
        res = {"r_mean": b["r"].mean().item(),
               "miou_oracle": ds.score(ds.predict(b["z_star"]), gt),
               "miou_received": ds.score(ds.predict(b["z_p"]), gt)}
        for nfe in nfes:
            z_hat = sample_batched(model, b, nfe, ds.generator(7))
            res[f"mse@{nfe}"] = (z_hat - b["z_star"]).pow(2).mean().item()
            res[f"miou@{nfe}"] = ds.score(ds.predict(z_hat), gt)
        if isinstance(model, SchrodingerBridge) and cname != "easy":
            # deterministic multi-step path, closer to the posterior mean than one sample
            z_ode = torch.cat([model.sample(b["z_p"][i:i + 64], b["r"][i:i + 64], max(nfes), stochastic=False)
                               for i in range(0, len(b["z_p"]), 64)])
            res[f"miou_ode@{max(nfes)}"] = ds.score(ds.predict(z_ode), gt)
        um = uncertainty_map(model, b, nfes, n_samples, ds) if cname != "easy" else None
        if um is not None and isinstance(model, SchrodingerBridge):
            res[f"miou_postmean@{max(nfes)}x{n_samples}"] = ds.score(ds.predict(um[1]), gt)
        if um is not None:
            U = ds.to_label_res(um[0].mean(1, keepdim=True)).squeeze(1)
            r = ds.to_label_res(b["r"]).squeeze(1)
            err, valid = ds.error_map(ds.predict(um[1]), gt)
            U, r, err = U[valid], r[valid], err[valid]
            res["auroc_U"] = auroc(U, err)
            res["auroc_1-r"] = auroc(1 - r, err)
            res["auroc_U|r"] = binned_auroc(U, err, r)
        out[cname] = res
    return out


def make_data(a, device):
    if a.data == "real":
        from ...data.real import CachedLatents
        return CachedLatents(a.teacher, device)
    from ...data import SyntheticSwarm
    return SyntheticSwarm(device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--teacher", default="runs/teacher")
    p.add_argument("--models", nargs="+", default=list(GRID))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--nfes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--n-eval", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta0", type=float, default=1.0, help="bridge noise scale (SB models only)")
    p.add_argument("--t1-frac", type=float, default=0.0, help="share of SB training samples at exactly t=1")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay for all models (0 = off)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    a.out = a.out or f"runs/q1_{a.data}_s{a.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    ds = make_data(a, device)
    fit_gen = ds.generator(5)
    fit_batches = [ds.batch(64, fit_gen) for _ in range(8)]

    results = {}
    for name in a.models:
        print(f"[{name}] {GRID[name]}", flush=True)
        torch.manual_seed(a.seed)
        model = build(name, ds.C, a.width, fit_batches, a.beta0, a.t1_frac).to(device)
        t0 = time.time()
        train(model, ds, a.steps, a.batch, a.lr, device, a.seed, a.ema)
        results[name] = evaluate(model, ds, a.n_eval, a.nfes, a.n_samples)
        results[name]["train_s"] = round(time.time() - t0, 1)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "results": results}, indent=2))
    report(results, a.nfes, list(ds.conditions))
    print(f"\nsaved {a.out}" + ("  (synthetic data: pipeline check, not evidence)" if a.data == "synthetic" else ""))


def report(results, nfes, conds):
    first = next(iter(results.values()))
    print("\nmIoU per channel condition (oracle = clean z*, received = z_p without restoration)")
    print(f"{'':6s}" + "".join(f"{c:>13s}" for c in conds))
    print(f"{'oracle':6s}" + "".join(f"{first[c]['miou_oracle']:13.3f}" for c in conds))
    print(f"{'rx':6s}" + "".join(f"{first[c]['miou_received']:13.3f}" for c in conds))
    cols = [(c, n) for c in conds for n in nfes]
    print(f"\n{'model':6s}" + "".join(f"{f'{c}@{n}':>13s}" for c, n in cols))
    for name, res in results.items():
        print(f"{name:6s}" + "".join(f"{res[c][f'miou@{n}']:13.3f}" for c, n in cols))
    sb = {k: v for k, v in results.items() if "auroc_U" in v.get(conds[-1], {})}
    if sb:
        print(f"\nQ1b error detection ({conds[-1]}): AUROC of U, of 1-r, and of U within r-bins")
        for name, res in sb.items():
            s = res[conds[-1]]
            print(f"{name:6s}  U {s['auroc_U']:.3f}   1-r {s['auroc_1-r']:.3f}   U|r {s['auroc_U|r']:.3f}")


if __name__ == "__main__":
    main()
