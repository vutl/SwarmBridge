"""Temporal block-selection pilot (protocol: reports/archive/sb-town3/review-block-selection-proposal-2026-11-03.md).

python -m swarmbridge.experiments.temporal_select.run --split val
python -m swarmbridge.experiments.temporal_select.run --split test

Each frame of a 20-frame rollout gets a total budget b of the UAV's footprint blocks in two rounds.
Round 1 sends b/2 random blocks: same blocks and same channel draw for every policy. Round 2 sends
b/2 further NEW blocks chosen by the policy, again on a channel draw shared by all policies. The
receiver is always the temporal_v1 model of the same seed, and every policy rolls out on its own
state, so only the round-2 selection differs. Evaluation only: no training, no checkpoint changes.

Policies and the information they use (the policy set was fixed before any run):
  random          nothing
  entropy         receiver: entropy of the decoded round-1 estimate
  ens_latent      receiver: variance of the round-1 estimates of the three temporal_v1 seeds
  ens_task        receiver: 1 - max of their mean decoded class probabilities
  replica_latent  sender: |pred - z|^2 of a receiver REPLICA it maintains itself; the replica knows the
                  per-block ACKs and the mean SNR, not the fading or noise draws, so it drifts
  replica_task    sender: same replica, 1 - P(class of decode(z)) under the decoded replica estimate
  ideal_latent    DIAGNOSTIC: as replica_latent but with the receiver's TRUE estimate and state;
  ideal_task      not deployable with ACKs alone, and not an upper bound on task gain

--ack-loss p drops the whole ACK feedback of a frame with probability p (shared draw across policies).
The replica then assumes each sent block arrived with probability 1 - p_erase (the channel's mean).
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ...channel import aggregate, sample_params, transmit
from ..q2_progressive.run import EVAL_CFG
from ..q2_retx.run import to_blocks, to_cells
from ..sb_audit import run as audit
from ..temporal.run import Decoder, Sequences, TemporalMapping

VERSION = "temporal_select_v1"
BUDGETS = (0.15, 0.3, 0.5)
POLICIES = ("random", "entropy", "ens_latent", "ens_task", "replica_latent", "replica_task",
            "ideal_latent", "ideal_task")
MEMBERS = (0, 1, 2)


def load_temporal(seed, args):
    model = TemporalMapping(16, args.width).to(args.device)
    model.load_state_dict(torch.load(Path(args.source) / f"temporal_s{seed}" / f"ema_{args.steps}.pt"))
    return model.eval().requires_grad_(False)


def top_k(scores, cand, k):
    """Per grid i, the k[i] highest-scoring candidate blocks."""
    scores = torch.where(cand, scores, torch.full_like(scores, -float("inf")))
    sel = torch.zeros_like(cand)
    for i in range(len(cand)):
        n = min(int(k[i]), int(cand[i].sum()))
        if n:
            sel[i].view(-1)[torch.topk(scores[i].flatten(), n).indices] = True
    return sel


def block_mean(x, shape):
    """(B,C,H,W) per-cell or per-pixel map -> (B,h,w) block means."""
    return F.adaptive_avg_pool2d(x.float().mean(1, keepdim=True), shape).squeeze(1)


def replica_rx(z, acks, snr):
    """What the sender can reconstruct of the reception: unit-power latent, mean SNR, no fading/noise."""
    g = sum(acks) * snr
    r = g / (1 + g)
    return r * z, r


@torch.no_grad()
def run(seed, args, seq, decoder, members):
    device = args.device
    model = members[seed]

    def net(m, *x):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            return m(*x).float()

    def probs(z):
        return decoder(z).float().softmax(1)

    order = torch.tensor(seq.sequences, device=device)  # (S, T)
    S, T = order.shape
    results = {}
    for budget in BUDGETS:
        conf = {p: torch.zeros(S, 3, 3, dtype=torch.long, device=device) for p in POLICIES}
        state = {p: None for p in POLICIES}
        replica = {p: None for p in POLICIES}
        sent2 = {p: 0 for p in POLICIES}
        arrived2 = {p: 0 for p in POLICIES}
        seconds = {p: 0.0 for p in POLICIES}
        for t in range(T):
            idx = order[:, t]
            z, fov, y = seq.z[idx], seq.fov[idx], seq.y[idx]
            cand = to_blocks(fov) > 0
            shape = cand.shape[-2:]
            k = torch.round(budget / 2 * cand.flatten(1).sum(1).float())
            base = 10 * (10_000 * seed + 100 * t + int(round(budget * 100)))

            def gen(offset):
                return torch.Generator(device=device).manual_seed(base + offset)

            params = sample_params(EVAL_CFG, S, 1, device, gen(0), None)
            snr = (10 ** (params["snr_db"] / 10)).view(S, 1, 1, 1)

            def send(sel, offset):
                return transmit(z[:, None], torch.zeros_like(z[:, None]), (fov * to_cells(sel))[:, None],
                                params, EVAL_CFG, gen(offset))

            first = top_k(torch.rand(cand.shape, device=device, generator=gen(1)), cand, k)
            rx1 = send(first, 2)
            z_p1, r1 = aggregate(*rx1)
            lost = (torch.rand(S, device=device, generator=gen(5)) < args.ack_loss).view(S, 1, 1, 1)
            arrive = (1 - params["p_erase"]).view(S, 1, 1, 1)

            def acked(rx, sel):
                """Per-cell arrival as the sender sees it: the ACK, or its expectation if the ACK was lost."""
                return torch.where(lost, fov * to_cells(sel) * arrive, (rx[2][:, 0] > 0).float())

            ack1 = acked(rx1, first)
            random_scores = torch.rand(cand.shape, device=device, generator=gen(3))
            teacher_label = decoder(z).argmax(1, keepdim=True)

            def memory(previous):
                if t == 0:
                    return torch.zeros_like(z), torch.zeros_like(fov)
                return seq.warp(previous, idx)

            for p in POLICIES:
                tick = time.perf_counter()
                prev, valid = memory(state[p])
                pred1 = net(model, z_p1, r1, prev, valid)
                if p == "random":
                    score = random_scores
                elif p == "entropy":
                    q = probs(pred1)
                    score = block_mean(-(q * q.clamp_min(1e-8).log()).sum(1, keepdim=True), shape)
                elif p.startswith("ens"):
                    preds = torch.stack([pred1 if m is model else net(m, z_p1, r1, prev, valid)
                                         for m in members.values()])
                    if p == "ens_latent":
                        score = block_mean(preds.var(0), shape)
                    else:
                        q = torch.stack([probs(x) for x in preds]).mean(0)
                        score = block_mean(1 - q.max(1, keepdim=True).values, shape)
                else:
                    if p.startswith("replica"):
                        r_prev, r_valid = memory(replica[p])
                        estimate = net(model, *replica_rx(z, [ack1], snr), r_prev, r_valid)
                    else:
                        estimate = pred1
                    if p.endswith("latent"):
                        score = block_mean((estimate - z).square(), shape)
                    else:
                        score = block_mean(1 - probs(estimate).gather(1, teacher_label), shape)
                second = top_k(score, cand & ~first, k)
                rx2 = send(second, 4)
                z_p, r = aggregate(*(torch.cat([a, b], 1) for a, b in zip(rx1, rx2)))
                state[p] = net(model, z_p, r, prev, valid)
                if p.startswith("replica"):
                    replica[p] = net(model, *replica_rx(z, [ack1, acked(rx2, second)], snr), r_prev, r_valid)
                sent2[p] += int(second.sum())
                arrived2[p] += int(((block_mean(rx2[2][:, 0], shape) > 0) & second).sum())
                label = decoder(state[p]).argmax(1)
                for i in range(S):
                    conf[p][i] += audit.confusion(label[i], y[i])
                seconds[p] += time.perf_counter() - tick
        row = {}
        for p in POLICIES:
            row[p] = audit.confusion_metrics(conf[p].sum(0))
            row[p]["per_sequence_miou"] = [audit.confusion_metrics(c)["miou"] for c in conf[p]]
            row[p]["round2_sent"], row[p]["round2_arrived"] = sent2[p], arrived2[p]
            row[p]["seconds"] = seconds[p]
        results[str(budget)] = row
        print(f"seed {seed} budget {budget:.0%}: " + " | ".join(
            f"{p} {row[p]['miou']:.4f}/{row[p]['vehicle_iou']:.4f}" for p in POLICIES), flush=True)
    return results


def summarize(out, split, seeds, ack_loss):
    runs = {s: json.loads((out / f"s{s}_{split}.json").read_text()) for s in seeds}
    lines = [f"# {VERSION} — {split}, seeds {list(seeds)}, ACK loss {ack_loss}", "",
             "mIoU / vehicle IoU, mean over seeds; delta vs random in mIoU points with the per-seed sign count.", ""]
    for budget in map(str, BUDGETS):
        lines += [f"## budget {float(budget):.0%} (half random, half chosen)", "",
                  "| policy | mIoU | vehicle IoU | Δ mIoU vs random | seeds > random | s/rollout |",
                  "|---|---:|---:|---:|---:|---:|"]
        for p in POLICIES:
            m = [runs[s][budget][p] for s in seeds]
            d = [100 * (runs[s][budget][p]["miou"] - runs[s][budget]["random"]["miou"]) for s in seeds]
            lines.append(f"| {p} | {sum(x['miou'] for x in m) / len(m):.4f} | "
                         f"{sum(x['vehicle_iou'] for x in m) / len(m):.4f} | {sum(d) / len(d):+.2f} | "
                         f"{sum(x > 0 for x in d)}/{len(d)} | {sum(x['seconds'] for x in m) / len(m):.1f} |")
        lines.append("")
    (out / f"summary_{split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(MEMBERS))
    p.add_argument("--source", default="runs/temporal_v1")
    p.add_argument("--teacher", default="runs/teacher")
    p.add_argument("--data-root", default="data/coperception_uav")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--ack-loss", type=float, default=0.0)
    p.add_argument("--out", default=f"runs/{VERSION}")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    decoder = Decoder(args.teacher, args.device)
    with audit.gpu_lock():
        seq = Sequences(args.split, args.teacher, args.data_root, args.device)
        members = {s: load_temporal(s, args) for s in MEMBERS}
        for seed in args.seeds:
            audit.atomic_json(out / f"s{seed}_{args.split}.json", run(seed, args, seq, decoder, members))
    summarize(out, args.split, args.seeds, args.ack_loss)
    print("SELECT DONE", flush=True)


if __name__ == "__main__":
    main()
