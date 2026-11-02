"""Train the teacher (encoder + BEV head) on clean per-UAV BEV grids, then cache latents.

    python -m swarmbridge.teacher.train data/coperception_uav/bev_town3 runs/teacher

Labels: gt restricted to this UAV's own footprint (leader setup: each region has one source),
taken at 0.5 m (every 2nd cell). Cache (<out>/latents_<split>.pt): standardized z (fp16),
labels (192x192), footprint masks at 48 and 192, so Q1 runs never touch images again.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import IGNORE, K, Decoder, Encoder, miou


def load_split(root, split):
    rgb, lab, fov = [], [], []
    files = sorted((Path(root) / split).glob("*.npz"))
    for f in files:
        d = np.load(f)
        i = int(f.stem.rsplit("_", 1)[1])
        own = d["seen"][i]
        y = np.where(own, d["gt"], IGNORE)
        rgb.append(d["rgb"])
        lab.append(y[1::2, 1::2])
        fov.append(own)
    return (torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(np.stack(lab)).long(), torch.from_numpy(np.stack(fov)), [f.stem for f in files])


def augment(x, y):
    """Random 90-degree rotation + flip, applied identically to image and labels (BEV)."""
    k = int(torch.randint(4, (1,)))
    x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    if torch.rand(1) < 0.5:
        x, y = x.flip(3), y.flip(2)
    return x, y


@torch.no_grad()
def evaluate(enc, dec, rgb, lab, device, bs=16):
    preds = []
    for i in range(0, len(rgb), bs):
        with torch.autocast(device, dtype=torch.bfloat16):
            preds.append(dec(enc(rgb[i:i + bs].to(device))).argmax(1).cpu())
    return miou(torch.cat(preds), lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--veh-weight", type=float, default=5.0)
    a = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr_rgb, tr_lab, _, _ = load_split(a.data, "train")
    va_rgb, va_lab, _, _ = load_split(a.data, "val")
    freq = torch.bincount(tr_lab[tr_lab != IGNORE], minlength=K).float()
    print(f"train {len(tr_rgb)}  val {len(va_rgb)}  class freq {(freq / freq.sum()).numpy().round(4)}")

    enc, dec = Encoder(a.C).to(device), Decoder(a.C).to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr_rgb) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    weight = torch.tensor([1.0, 1.0, a.veh_weight], device=device)

    best, step, t0 = -1, 0, time.time()
    for ep in range(a.epochs):
        enc.train(), dec.train()
        perm = torch.randperm(len(tr_rgb))
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            x, y = augment(tr_rgb[perm[i:i + a.bs]], tr_lab[perm[i:i + a.bs]])
            x, y = x.to(device), y.to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = F.cross_entropy(dec(enc(x)).float(), y, weight=weight, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        enc.eval(), dec.eval()
        m, per = evaluate(enc, dec, va_rgb, va_lab, device)
        print(f"ep {ep:2d}  loss {loss.item():.3f}  val mIoU {m:.3f}  "
              f"(other {per[0]:.3f} road {per[1]:.3f} vehicle {per[2]:.3f})  {time.time() - t0:.0f}s", flush=True)
        if m > best:
            best = m
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "C": a.C}, out / "teacher.pt")

    ck = torch.load(out / "teacher.pt")
    enc.load_state_dict(ck["enc"]), dec.load_state_dict(ck["dec"])
    enc.eval(), dec.eval()
    print(f"best val mIoU {best:.3f}; caching latents")

    # standardize z per channel with train statistics (channel model assumes unit power)
    with torch.no_grad():
        zs = torch.cat([enc(tr_rgb[i:i + 32].to(device)).float().cpu() for i in range(0, len(tr_rgb), 32)])
    mu, sd = zs.mean((0, 2, 3), keepdim=True), zs.std((0, 2, 3), keepdim=True)
    torch.save({"mu": mu, "sd": sd}, out / "latent_stats.pt")

    summary = {"best_val_miou": best}
    for split in ("train", "val", "test"):
        rgb, lab, fov, names = load_split(a.data, split)
        with torch.no_grad():
            z = torch.cat([enc(rgb[i:i + 32].to(device)).float().cpu() for i in range(0, len(rgb), 32)])
        z = (z - mu) / sd
        fov48 = F.avg_pool2d(fov[:, None].float(), 8) > 0.5
        torch.save({"z": z.half(), "y": lab.byte(), "fov48": fov48, "fov192": fov[:, 1::2, 1::2],
                    "names": names}, out / f"latents_{split}.pt")
        # sanity: decoding the cached (de-standardized) latent reproduces the teacher's mIoU
        with torch.no_grad():
            pred = torch.cat([dec((z[i:i + 32] * sd + mu).to(device)).argmax(1).cpu() for i in range(0, len(z), 32)])
        m, per = miou(pred, lab)
        summary[split] = {"n": len(z), "miou": m, "per_class": per}
        print(f"  {split}: {len(z)} latents  clean mIoU {m:.3f}  per class {np.round(per, 3)}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
