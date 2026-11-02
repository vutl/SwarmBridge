"""Q2 progressive transmission: is "send part, then ask for what is uncertain" cheaper than
"send everything at once" for the same task quality? (leader setup, CoPerception-UAV latents)

At the same total budget B (fraction of in-footprint blocks per UAV):
  one-shot   : every UAV sends a fixed random B of its blocks, no second round.
  two-round  : round 1 sends a fixed random B/2; the leader restores with H1, scores blocks,
               and asks for B/2 more (budget pooled over the 5 UAVs of a sample) chosen by
               random | channel (lowest r) | unc_h1 (H1 variance head) |
               unc_sb (spread of K stochastic bridge samples) | oracle (true error, bound).
Every variant is restored with the SAME H1 model, so only the choice of blocks differs.
Unsent blocks look exactly like erased ones to the receiver (r = 0), so models are trained
with erasure up to 90%. Bits: 8 bit per latent value; request/feedback cost not counted.

    python -m swarmbridge.experiments.q2_progressive.run --seed 0
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from ...bridges import HeteroMapping, Reference, SchrodingerBridge
from ...channel import ChannelConfig, aggregate, sample_params, transmit
from ...data.real import CachedLatents
from ..q1_reference.run import train
from ..q2_retx.run import BLOCK, restore, select, to_blocks, to_cells

POLICIES = ("random", "channel", "unc_h1", "unc_sb", "oracle")
EVAL_CFG = ChannelConfig(ages=(0.0,))                     # hetero channels: usual ranges
TRAIN_CFG = ChannelConfig(ages=(0.0,), p_erase=(0.0, 0.9))  # also covers "not sent yet"


def get_model(kind, ds, seed, steps, device):
    ck = Path(f"runs/q2p/{kind}_s{seed}_{steps}.pt")
    torch.manual_seed(seed)
    if kind == "h1":
        model = HeteroMapping(ds.C, use_csi=True)
    else:
        model = SchrodingerBridge(ds.C, Reference("iso", ds.C), use_csi=True, beta0=10.0)
    model = model.to(device)
    if ck.exists():
        model.load_state_dict(torch.load(ck))
        model.eval()
        print(f"loaded {ck}")
    else:
        print(f"training {kind} (seed {seed}, {steps} steps)")
        train(model, ds, steps, 32, 3e-4, device, seed, ema=0.999)
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ck)
    return model


@torch.no_grad()
def sb_spread(sb, z_p, r, gen, k=8, nfe=4, bs=64):
    out = []
    for i in range(0, len(z_p), bs):
        zs = torch.stack([sb.sample(z_p[i:i + bs], r[i:i + bs], nfe, gen) for _ in range(k)])
        out.append(zs.var(0))
    return torch.cat(out)


def random_subset(cand, frac, gen):
    """a fixed random `frac` of each grid's candidate blocks (what a UAV sends unprompted)"""
    s = torch.rand(cand.shape, device=cand.device, generator=gen)
    s = torch.where(cand, s, torch.full_like(s, -1.0))
    sel = torch.zeros_like(cand)
    for i in range(len(cand)):
        k = int(round(frac * cand[i].sum().item()))
        if k:
            sel[i].view(-1)[torch.topk(s[i].flatten(), k).indices] = True
    return sel


@torch.no_grad()
def run_condition(h1, sb, ds, fixed, budgets, seed, k_sb):
    data = ds.eval
    z = data["z"].to(ds.device).float()
    fov = data["fov48"].to(ds.device).float()
    y_lab = data["y"].to(ds.device).long()
    B = len(z)
    groups = defaultdict(list)
    for i, n in enumerate(data["names"]):
        groups[n.rsplit("_", 1)[0]].append(i)
    groups = [torch.tensor(v, device=ds.device) for v in groups.values()]
    cand = to_blocks(fov) > 0
    n_cand = cand.flatten(1).sum(1).float()
    bits_per_block = BLOCK * BLOCK * z.shape[1] * 8

    gen = ds.generator(2000 + seed)
    params = sample_params(EVAL_CFG, B, 1, ds.device, gen, fixed)
    msg, zero = z[:, None], torch.zeros_like(z[:, None])
    # two independent channel uses per block are available; which blocks use them is decided later
    order = torch.rand(cand.shape, device=ds.device, generator=gen)

    def send(sel, g):
        return transmit(msg, zero, (fov * to_cells(sel))[:, None], params, EVAL_CFG, g)

    def score(rx):
        z_p, r = aggregate(*rx)
        return ds.score(ds.predict(restore(h1, z_p, r)[0]), y_lab)

    def kbit(*sels):
        return sum(s.sum().item() for s in sels) * bits_per_block / B / 1e3

    full = cand
    res = {"full_send": {"kbit": kbit(full), "miou": score(send(full, ds.generator(11 + seed)))}, "budgets": []}
    for bud in budgets:
        row = {"budget": bud}
        one = random_subset(cand, bud, ds.generator(21 + seed))
        row["one_shot"] = {"kbit": kbit(one), "miou": score(send(one, ds.generator(31 + seed)))}

        first = random_subset(cand, bud / 2, ds.generator(21 + seed))
        y1, w1, g1 = send(first, ds.generator(31 + seed))
        z_p, r = aggregate(y1, w1, g1)
        z_hat, U = restore(h1, z_p, r)
        scores = {"random": order, "channel": 1 - to_blocks(r), "unc_h1": to_blocks(U),
                  "unc_sb": to_blocks(sb_spread(sb, z_p, r, ds.generator(41 + seed), k=k_sb)),
                  "oracle": to_blocks((z_hat - z).pow(2))}
        for pol in POLICIES:
            second = select(scores[pol], cand, groups, bud / 2, None)
            y2, w2, g2 = send(second, ds.generator(51 + seed))
            rx = (torch.cat([y1, y2], 1), torch.cat([w1, w2], 1), torch.cat([g1, g2], 1))
            row[pol] = {"kbit": kbit(first, second), "miou": score(rx)}
        res["budgets"].append(row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--k-sb", type=int, default=8, help="bridge samples for the SB spread")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or f"runs/q2p_s{a.seed}.json"
    device = "cuda"
    ds = CachedLatents(a.teacher, device)
    ds.cfg = TRAIN_CFG
    h1 = get_model("h1", ds, a.seed, a.steps, device)
    sb = get_model("sb", ds, a.seed, a.steps, device)

    results = {}
    for cname, fixed in (("hetero", None), ("mid", ds.conditions["mid"])):
        res = run_condition(h1, sb, ds, fixed, a.budgets, a.seed, a.k_sb)
        results[cname] = res
        f = res["full_send"]
        print(f"\n[{cname}] full send: {f['miou']:.3f} at {f['kbit']:.0f} kbit/UAV")
        print(f"{'budget':>15s}{'one-shot':>10s}" + "".join(f"{p:>10s}" for p in POLICIES))
        for row in res["budgets"]:
            print(f"{row['budget']:4.0%} {row['one_shot']['kbit']:6.0f} kb{row['one_shot']['miou']:10.3f}"
                  + "".join(f"{row[p]['miou']:10.3f}" for p in POLICIES))
    Path(a.out).write_text(json.dumps({"args": vars(a), "results": results}, indent=2))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
