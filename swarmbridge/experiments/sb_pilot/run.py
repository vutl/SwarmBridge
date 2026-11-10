"""Pilot after the root audit: train on the SAME corruption chain used at evaluation.

python -m swarmbridge.experiments.sb_pilot.run --check      # coverage stats, no training
python -m swarmbridge.experiments.sb_pilot.run              # 6 models x seed 0 x 6k updates

Old audit trained on "whole footprint through channel, erasure 0-0.9" but evaluated
"select a fraction of blocks, then channel, erasure 0-0.6". Here training draws a sender
budget from the evaluation levels, selects blocks, then applies the evaluation channel.
Every model below uses this sampler, so old-sampler checkpoints are reference only.

Factors (one at a time against the matched mapping):
  input:    x  = x_t only;  c = [x_t, z_received, r]  (mapping gets [z_rx, z_rx, r])
  endpoint: p10 / p50 = fraction of bridge examples drawn exactly at t=1
Pre-registered gate (from reports/archive/sb-town3/sb-training-root-audit-2026-10-31.md): an SB variant passes if
  mIoU_SB - mIoU_map >= -0.005 at BOTH 15% and 30% sent, and
  mean vehicle IoU at 70/100% is not below the matched mapping by more than 0.005.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from ...bridges.audit import AuditDenoiser
from ...channel import aggregate, sample_params, transmit
from ..q2_progressive.run import EVAL_CFG
from ..q2_retx.run import to_blocks, to_cells
from ..sb_audit import run as audit

VERSION = "sb_pilot_matched_v1"
# ordered so the most informative pairs finish first
MODELS = ("map_x", "sb_x_p50", "map_c", "sb_c_p10", "sb_x_p10", "sb_c_p50")
SEND_LEVELS = audit.FRACTIONS


def make_model(name, channels, width, endpoint_fraction=None):
    kind, family, *rest = name.split("_")
    conditioned = family == "c"
    if kind == "map":
        return AuditDenoiser(channels, bridge=False, condition_source=conditioned, csi=conditioned, width=width)
    return AuditDenoiser(channels, bridge=True, condition_source=conditioned, csi=conditioned, width=width,
                         endpoint_fraction=int(rest[0][1:]) / 100)


def select_blocks(candidates, fractions, generator):
    """Per-grid random subset of candidate blocks; same rounding as q2_progressive.random_subset."""
    scores = torch.rand(candidates.shape, device=candidates.device, generator=generator)
    scores = torch.where(candidates, scores, torch.full_like(scores, -1.0))
    selected = torch.zeros_like(candidates)
    for i in range(len(candidates)):
        k = int(round(fractions[i].item() * candidates[i].sum().item()))
        if k:
            selected[i].view(-1)[torch.topk(scores[i].flatten(), k).indices] = True
    return selected


class MatchedBevData(audit.BevData):
    def batch(self, batch_size, generator):
        ds = self.ds
        idx = torch.randint(len(ds.train["z"]), (batch_size,), generator=torch.Generator().manual_seed(
            int(torch.randint(2**31, (1,), device=self.device, generator=generator).item())))
        z = ds.train["z"][idx].to(self.device).float()
        fov = ds.train["fov48"][idx].to(self.device).float()
        levels = torch.tensor(SEND_LEVELS, device=self.device)
        fractions = levels[torch.randint(len(levels), (batch_size,), device=self.device, generator=generator)]
        selected = select_blocks(to_blocks(fov) > 0, fractions, generator)
        params = sample_params(EVAL_CFG, batch_size, 1, self.device, generator, None)
        received = transmit(z[:, None], torch.zeros_like(z[:, None]), (fov * to_cells(selected))[:, None],
                            params, EVAL_CFG, generator)
        z_p, r = aggregate(*received)
        return {"z_star": z, "z_p": z_p, "r": r, "fov": fov, "y": ds.train["y"][idx].to(self.device).long()}


def coverage_check(data, n=4000):
    """Fraction of the footprint actually received (r>0) under the matched training sampler."""
    generator = torch.Generator(device=data.device).manual_seed(7)
    fractions = []
    for _ in range(n // 100):
        b = data.batch(100, generator)
        received = ((b["r"] > 0).float() * b["fov"]).flatten(1).sum(1)
        fractions.append(received / b["fov"].flatten(1).sum(1).clamp_min(1))
    frac = torch.cat(fractions)
    return dict(mean_received_area=frac.mean().item(), p_le_15=(frac <= .15).float().mean().item(),
                p_le_10=(frac <= .10).float().mean().item(), n=len(frac))


def write_summary(out):
    out = Path(out)
    rows = {}
    for file in sorted(out.glob("*/evaluation_*.json")):
        rows[file.parent.name] = json.loads(file.read_text(encoding="utf-8"))["fractions"]
    audit.atomic_json(out / "results.json", rows)
    lines = ["**Matched-sampler SB pilot (6k, EMA, one-step; sb_x_p70 is exploratory, added after seed-0 results)**", "",
             "| Model | " + " | ".join(f"{f:.0%} mIoU" for f in SEND_LEVELS) + " | veh 70% | veh 100% |",
             "|---|" + "---:|" * (len(SEND_LEVELS) + 2)]
    for name, fr in rows.items():
        m = [fr[str(f)]["one_step"]["miou"] for f in SEND_LEVELS]
        v = [fr[str(f)]["one_step"]["vehicle_iou"] for f in (0.7, 1.0)]
        lines.append(f"| {name} | " + " | ".join(f"{x:.4f}" for x in m) + f" | {v[0]:.4f} | {v[1]:.4f} |")
    lines += ["", "Gate vs matched mapping, same seed (pass: d15>=-0.005, d30>=-0.005, d_veh(70/100)>=-0.005).",
              "The pre-registered decision uses seed 0; 'mean' rows average the paired differences over finished seeds.", ""]
    get = lambda n, f, k: rows[n][str(f)]["one_step"][k]
    diffs = {}
    for name in rows:
        if not name.startswith("sb_"):
            continue
        model, seed = name.rsplit("_s", 1)
        ref = "map_" + name.split("_")[1] + "_s" + seed
        if ref not in rows:
            continue
        d = (get(name, .15, "miou") - get(ref, .15, "miou"), get(name, .3, "miou") - get(ref, .3, "miou"),
             sum(get(name, f, "vehicle_iou") - get(ref, f, "vehicle_iou") for f in (.7, 1.0)) / 2)
        diffs.setdefault(model, []).append(d)
        ok = all(x >= -.005 for x in d)
        lines.append(f"- {name} vs {ref}: d15={d[0]:+.4f} d30={d[1]:+.4f} d_veh={d[2]:+.4f} -> {'PASS' if ok else 'fail'}")
    lines.append("")
    for model, ds in diffs.items():
        mean = [sum(x[i] for x in ds) / len(ds) for i in range(3)]
        signs = [sum(x[i] > 0 for x in ds) for i in range(3)]
        lines.append(f"- mean {model} over {len(ds)} seed(s): d15={mean[0]:+.4f} d30={mean[1]:+.4f} "
                     f"d_veh={mean[2]:+.4f}; seeds with SB ahead: {signs[0]}/{signs[1]}/{signs[2]}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=f"runs/{VERSION}")
    p.add_argument("--teacher", default="runs/teacher")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--check", action="store_true")
    # identical to the audit defaults
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--microbatch", type=int, default=16)
    p.add_argument("--image-batch", type=int, default=64)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--bev-eval", type=int, default=200)
    p.add_argument("--endpoint-fraction", type=float, default=.1)  # unused: encoded in model name
    p.add_argument("--task-weight", type=float, default=1.)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ema", type=float, default=.999)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    data = MatchedBevData(args.teacher, args.device)
    if args.check:
        print(json.dumps(coverage_check(data)), flush=True)
        return
    # the audit's train loop builds models through its module-level factory
    audit.make_model = make_model
    Path(args.out).mkdir(parents=True, exist_ok=True)
    status = dict(state="starting", started_at=audit.now(), pid=os.getpid(), version=VERSION)
    with audit.gpu_lock():
        try:
            for seed in args.seeds:
                for name in args.models:
                    audit.train_until(name, data, seed, args.steps, args, status)
                    write_summary(args.out)
            status.update(state="completed", updated_at=audit.now())
        except BaseException as error:
            status.update(state="failed", error=f"{type(error).__name__}: {error}", updated_at=audit.now())
            raise
        finally:
            audit.atomic_json(Path(args.out) / "status.json", status)
            write_summary(args.out)
    print("PILOT COMPLETED", flush=True)


if __name__ == "__main__":
    main()
