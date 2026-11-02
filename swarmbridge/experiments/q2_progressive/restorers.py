"""Does a bridge restore better than regression when information is scarce?

Every UAV sends only a fixed random fraction of its blocks (one shot, hetero channels);
the leader restores with H1 (regression mean) or with the bridge (1 step, 4-step ODE, mean of
8 stochastic 4-step samples). Uses the models trained by q2_progressive.run (erasure up to 90%).

    python -m swarmbridge.experiments.q2_progressive.restorers --seeds 0 1 2
"""
import argparse
import json

import numpy as np
import torch

from ...channel import aggregate, sample_params, transmit
from ...data.real import CachedLatents
from ...teacher.model import miou
from ..q2_retx.run import to_blocks, to_cells
from .run import EVAL_CFG, TRAIN_CFG, get_model, random_subset


@torch.no_grad()
def batched(fn, z_p, r, bs=64):
    return torch.cat([fn(z_p[i:i + bs], r[i:i + bs]) for i in range(0, len(z_p), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.15, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--out", default="runs/q2p_restorers.json")
    a = ap.parse_args()
    ds = CachedLatents(a.teacher, "cuda")
    ds.cfg = TRAIN_CFG
    z = ds.eval["z"].cuda().float()
    fov = ds.eval["fov48"].cuda().float()
    y = ds.eval["y"].cuda().long()
    cand = to_blocks(fov) > 0
    out, vehicle = {}, {}
    for seed in a.seeds:
        h1 = get_model("h1", ds, seed, 6000, "cuda")
        sb = get_model("sb", ds, seed, 6000, "cuda")
        gen = ds.generator(3000 + seed)
        params = sample_params(EVAL_CFG, len(z), 1, "cuda", gen, None)
        for frac in a.fracs:
            sel = random_subset(cand, frac, ds.generator(21 + seed))
            rx = transmit(z[:, None], torch.zeros_like(z[:, None]), (fov * to_cells(sel))[:, None],
                          params, EVAL_CFG, ds.generator(31 + seed))
            z_p, r = aggregate(*rx)
            g = ds.generator(41 + seed)
            samples = [batched(lambda a_, b_: sb.sample(a_, b_, 4, g), z_p, r) for _ in range(8)]
            labels = {
                "received": ds.predict(z_p),
                "h1": ds.predict(batched(lambda a_, b_: h1.sample(a_, b_), z_p, r)),
                "sb_1step": ds.predict(batched(lambda a_, b_: sb.sample(a_, b_, 1), z_p, r)),
                "sb_ode4": ds.predict(batched(lambda a_, b_: sb.sample(a_, b_, 4, stochastic=False), z_p, r)),
                "sb_mean8": ds.predict(torch.stack(samples).mean(0)),
                # average the decoder's class probabilities over samples (task-space averaging)
                "sb_softmax8": sum(ds.probs(s) for s in samples).argmax(1),
            }
            for k, pred in labels.items():
                m, per = miou(pred, y)
                out.setdefault(str(frac), {}).setdefault(k, []).append(m)
                vehicle.setdefault(str(frac), {}).setdefault(k, []).append(per[2])
        print(f"seed {seed} done", flush=True)
    for title, table in (("mIoU", out), ("vehicle IoU", vehicle)):
        keys = list(next(iter(table.values())))
        print(f"\n{title}\n{'sent':>6s}" + "".join(f"{k:>15s}" for k in keys))
        for frac, d in table.items():
            print(f"{float(frac):6.0%}" + "".join(
                f"{np.mean(d[k]):9.3f}±{np.std(d[k], ddof=1) if len(d[k]) > 1 else 0:.3f}" for k in keys))
        print(f"paired differences vs h1 ({title}, mean over seeds):")
        for frac, d in table.items():
            h = np.array(d["h1"])
            print(f"{float(frac):6.0%}  " + "  ".join(f"{k} {np.mean(np.array(d[k]) - h):+.4f}" for k in keys if k.startswith("sb")))
    json.dump({"miou": out, "vehicle_iou": vehicle}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
