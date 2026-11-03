"""Segmentation-map baseline with exact bit accounting.

python -m swarmbridge.experiments.segmap_baseline.run --split val

The UAV sends blocks of the teacher's BEV label map (argmax of decode(z), 3 classes, 192x192) instead
of latent blocks. Each 16x16-pixel block (= one 4x4-cell latent block) is coded on its own with a static
context model p(label | left, up, up-left) fitted on the teacher maps of the train split; its cost is the
ideal code length sum(-log2 p). The link is digital: a sent block arrives intact or is erased with the
grid's p_erase (EVAL_CFG range, same draws for every policy). Budgets are in kbit per frame per UAV.

Policies (every one rolls out over the 20-frame sequences on its own receiver map):
  nomem_random  receiver keeps nothing; random blocks; unknown pixels -> class 0 ("other")
  mem_random    receiver warps its previous map with the poses; random blocks fill it in
  mem_replica   as mem_random, but the UAV knows the receiver map exactly (digital link + ACK) and sends
                the blocks with the most wrong pixels per bit, until the budget is spent

Also reported: the bits of the whole map and the latent bits per grid (uniform quantisation step 0.5,
per-channel static entropy model), for the comparison with latent transmission.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ...channel import sample_params
from ..q2_progressive.run import EVAL_CFG
from ..sb_audit import run as audit
from ..temporal.run import Decoder, Sequences

VERSION = "segmap_baseline_v1"
BUDGETS_KBIT = (0.1, 0.2, 0.4, 0.7, 1.0, 1.5, float("inf"))
POLICIES = ("nomem_random", "mem_random", "mem_replica")
PX, BLOCK_PX = 192, 16
IGNORE = 255


def contexts(labels):
    """Context index (left, up, up-left; 3 = outside the block) for independently coded 16x16 blocks."""
    y = labels.long()
    none = torch.full_like(y, 3)
    left, up, ul = none.clone(), none.clone(), none.clone()
    left[:, :, 1:], up[:, 1:, :], ul[:, 1:, 1:] = y[:, :, :-1], y[:, :-1, :], y[:, :-1, :-1]
    first = torch.arange(PX, device=y.device) % BLOCK_PX == 0
    left[:, :, first] = 3; ul[:, :, first] = 3; up[:, first, :] = 3; ul[:, first, :] = 3
    return left * 16 + up * 4 + ul


def fit_code(labels, fov):
    counts = torch.ones(64, 3, device=labels.device)
    counts.index_put_((contexts(labels)[fov], labels.long()[fov]), torch.ones(int(fov.sum()), device=labels.device),
                      accumulate=True)
    return -(counts / counts.sum(1, keepdim=True)).log2()


def block_bits(labels, fov, code):
    """(B, 12, 12) code length of each block's FOV pixels."""
    cost = code[contexts(labels), labels.long()] * fov
    return F.avg_pool2d(cost[:, None], BLOCK_PX).squeeze(1) * BLOCK_PX ** 2


def latent_bits(train_z, train_fov, z, fov, step=0.5):
    total = 0.0
    for c in range(z.shape[1]):
        q_tr = (train_z[:, c] / step).round().clamp(-40, 40).long() + 40
        q = (z[:, c] / step).round().clamp(-40, 40).long() + 40
        h = torch.bincount(q_tr[train_fov[:, 0]], minlength=81).float() + 1
        total += -(h / h.sum()).log2()[q[fov[:, 0]]].sum().item()
    return total / len(z)


def pixel_grid(seq):
    """Upsample the 48x48 cell-centre warp grid to 192x192 pixel centres (the warp is affine)."""
    g = seq.grid.permute(0, 3, 1, 2)
    g = F.interpolate(g, size=(PX, PX), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    cell = (g + 1) / 2 * 47
    return (4 * cell + 1.5) / (PX - 1) * 2 - 1


def take(scores, bits, budget, cand):
    """Blocks in descending score order while the cumulative bits stay within the budget."""
    B = len(scores)
    s = torch.where(cand, scores, torch.full_like(scores, -float("inf"))).view(B, -1)
    order = s.argsort(1, descending=True)
    b = bits.view(B, -1).gather(1, order)
    ok = (b.cumsum(1) <= budget) & torch.isfinite(s.gather(1, order))
    sel = torch.zeros_like(s, dtype=torch.bool).scatter(1, order, ok)
    return sel.view_as(scores)


@torch.no_grad()
def run(args, seq, labels, fov192, bits, seed):
    device = args.device
    grid = pixel_grid(seq)
    order = torch.tensor(seq.sequences, device=device)
    S, T = order.shape
    results = {}
    for budget_kbit in BUDGETS_KBIT:
        budget = budget_kbit * 1e3
        conf = {p: torch.zeros(3, 3, dtype=torch.long, device=device) for p in POLICIES}
        state = {p: None for p in POLICIES}
        spent = {p: 0.0 for p in POLICIES}
        for t in range(T):
            idx = order[:, t]
            lab, fov, cost = labels[idx], fov192[idx], bits[idx]
            cand = F.max_pool2d(fov[:, None].float(), BLOCK_PX).squeeze(1) > 0
            gen = torch.Generator(device=device).manual_seed(10 * (10_000 * seed + 100 * t) + 7)
            p_erase = sample_params(EVAL_CFG, S, 1, device, gen, None)["p_erase"].view(S, 1, 1)
            keep = torch.rand(cand.shape, device=device, generator=gen) >= p_erase
            random_scores = torch.rand(cand.shape, device=device, generator=gen)
            for p in POLICIES:
                if p == "nomem_random" or t == 0:
                    pred = torch.zeros_like(lab)
                else:
                    has = (seq.prev[idx] >= 0)[:, None, None]
                    warped = F.grid_sample(state[p][:, None].float(), grid[idx], mode="nearest",
                                           padding_mode="zeros", align_corners=True)[:, 0].long()
                    pred = torch.where(has, warped, torch.zeros_like(warped))
                if p == "mem_replica":
                    wrong = F.avg_pool2d(((pred != lab) & fov).float()[:, None], BLOCK_PX).squeeze(1)
                    scores = torch.where(wrong > 0, wrong / cost.clamp_min(1.0), torch.full_like(wrong, -1.0))
                    sel = take(scores, cost, budget, cand & (wrong > 0))
                else:
                    sel = take(random_scores, cost, budget, cand)
                got = F.interpolate((sel & keep)[:, None].float(), scale_factor=BLOCK_PX)[:, 0] > 0
                state[p] = torch.where(got, lab, pred)
                spent[p] += (cost * sel).sum().item()
                conf[p] += audit.confusion(state[p], seq.y[idx])
        results[str(budget_kbit)] = {p: {**audit.confusion_metrics(conf[p]), "kbit_per_frame": spent[p] / (S * T) / 1e3}
                                     for p in POLICIES}
        print(f"budget {budget_kbit} kbit: " + " | ".join(
            f"{p} {results[str(budget_kbit)][p]['miou']:.4f}/{results[str(budget_kbit)][p]['vehicle_iou']:.4f}"
            f" @{results[str(budget_kbit)][p]['kbit_per_frame']:.2f}" for p in POLICIES), flush=True)
    return results


def teacher_labels(decoder, z, bs=64):
    return torch.cat([decoder(z[i:i + bs].float()).argmax(1) for i in range(0, len(z), bs)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="channel draws (no model is trained)")
    p.add_argument("--teacher", default="runs/teacher")
    p.add_argument("--data-root", default="data/coperception_uav")
    p.add_argument("--out", default=f"runs/{VERSION}")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    decoder = Decoder(args.teacher, args.device)
    with audit.gpu_lock():
        train = torch.load(f"{args.teacher}/latents_train.pt")
        split = torch.load(f"{args.teacher}/latents_{args.split}.pt")
        fov_tr = train["fov192"].to(args.device)
        lab_tr = teacher_labels(decoder, train["z"].to(args.device))
        code = fit_code(lab_tr, fov_tr)
        seq = Sequences(args.split, args.teacher, args.data_root, args.device)
        fov192 = split["fov192"].to(args.device)
        labels = teacher_labels(decoder, seq.z)
        bits = block_bits(labels, fov192, code)
        teacher = audit.confusion_metrics(audit.confusion(labels, seq.y))
        summary = {
            "teacher_full_map": teacher,
            "map_kbit_per_grid": bits.sum().item() / len(bits) / 1e3,
            "latent_kbit_per_grid_step0.5": latent_bits(train["z"].float(), train["fov48"], split["z"].float(),
                                                         split["fov48"]) / 1e3,
        }
        print(json.dumps(summary, indent=1), flush=True)
        runs = {s: run(args, seq, labels, fov192, bits, s) for s in args.seeds}
    audit.atomic_json(Path(args.out) / f"{args.split}.json", {"summary": summary, "seeds": runs})
    print("SEGMAP DONE", flush=True)


if __name__ == "__main__":
    main()
