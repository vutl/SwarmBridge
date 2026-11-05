"""Paired receiver adaptation: same initialization, schedules, optimizer and steps.

A single invocation trains both arms round-robin so an interrupted run cannot
spend the whole budget on one arm. Resume stores both optimizers, RNG and exact
schedule position. The 45 minute envelope includes validation and checkpoint I/O.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "uav3d_comm"))
import core as C
import receiver as R
from ft_tools import setup

SEED, MAX_EPOCHS = 0, 3


def make_schedule(split):
    rng = np.random.default_rng(SEED)
    train = []
    for epoch in range(MAX_EPOCHS):
        train.append([C.context(split, tok, rng) for tok in rng.permutation(split["tokens"]["train"])])
    val = []
    for tok in split["tokens"]["val"]:
        g = np.random.default_rng(int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16))
        ctx = C.context(split, tok, g)
        for k in [0, 16, 64]: val.append(dict(ctx, k=k, rank="obj"))
    return dict(seed=SEED, epochs=train, validation=val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("out")
    ap.add_argument("--budget-seconds", type=float, default=2700)
    args = ap.parse_args()
    job_start = time.monotonic()
    repo, root = Path(args.repo), Path(args.out)
    prep, out = root / "prepared", root / "matched"
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((prep / "split.json").read_text())
    schedule = make_schedule(split)
    sig = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
    if (out / "result.json").exists():
        old = json.loads((out / "result.json").read_text())
        assert old["schedule_hash"] == sig
        print("SKIP matched training already completed", flush=True); return
    C.save_json(out / "schedule.json", schedule)
    torch.set_num_threads(4)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = "configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/lowerbound/default.yaml"
    dataset, model0, ds0 = setup(cfg, str(repo / "external/uav3d_ckpt/lowerbound.pth"),
                                str(repo / "runs/uav3d_comm/lowerbound/infos_all.pkl"), "ego",
                                str(repo / "runs/uav3d_comm/fusion/ego_s2/decoder.pt"))
    R.expand_input(ds0)
    # Decoder training does not need the camera encoder. Keep the original model
    # structure for the known head API, but release its frozen backbone weights.
    model0.encoders["camera"]["backbone"].cpu()
    model0.encoders["camera"]["neck"].cpu()
    # Copy after moving the unused encoder to CPU; two decoders fit comfortably.
    models = {"box": model0, "joint": copy.deepcopy(model0)}
    dss = {k: m.encoders["camera"]["vtransform"].downsample for k, m in models.items()}
    opts = {}
    for arm, model in models.items():
        for p in model.parameters(): p.requires_grad_(False)
        params = list(dss[arm].parameters()) + list(model.decoder.parameters()) + list(model.heads["object"].parameters())
        for p in params: p.requires_grad_(True)
        opts[arm] = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-2)
        model.heads["object"].test_cfg["post_max_size"] = 500
    by = {i["token"]: k for k, i in enumerate(dataset.data_infos)}
    data = R.ReceiverData(prep)
    def loss(arm, ctx):
        x, metas, _ = data.build(ctx, with_tiles=arm == "joint")
        pred = R.forward(models[arm], dss[arm], x, metas)
        ann = dataset.get_ann_info(by[ctx["token"]])
        boxes = ann["gt_bboxes_3d"]
        mask = boxes.in_range_bev([-102.4, -102.4, 102.4, 102.4])
        losses = models[arm].heads["object"].loss([boxes[mask].to("cuda")],
                    [torch.as_tensor(ann["gt_labels_3d"])[mask.cpu()].cuda()], pred)
        return sum(v for v in losses.values() if torch.is_tensor(v))
    resume = out / "resume.pt"
    epoch0, step0, previous_seconds = 0, 0, 0.0
    best = {k: float("inf") for k in models}; history = []
    steps_per_epoch = len(schedule["epochs"][0])
    if resume.exists():
        saved = torch.load(str(resume), map_location="cuda")
        assert saved["schedule_hash"] == sig
        for arm in models:
            R.load_state(models[arm], dss[arm], saved["models"][arm]); opts[arm].load_state_dict(saved["opts"][arm])
        epoch0, step0 = saved["epoch"], saved["step"]
        previous_seconds, best, history = saved["seconds"], saved["best"], saved["history"]
        steps_per_epoch = saved["steps_per_epoch"]
        torch.set_rng_state(saved["rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
    start = job_start
    def elapsed(): return previous_seconds + time.monotonic() - start
    def checkpoint(epoch, step):
        blob = dict(schedule_hash=sig, models={k: R.state(m, dss[k]) for k,m in models.items()},
                    opts={k:o.state_dict() for k,o in opts.items()}, epoch=epoch, step=step,
                    seconds=elapsed(), best=best, history=history, steps_per_epoch=steps_per_epoch,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        p = out / "resume.tmp.pt"; torch.save(blob, str(p)); p.replace(resume)
    for model in models.values(): model.eval()
    # Non-mutating optimizer probe: no update or BN statistics change.
    if not resume.exists():
        probe = schedule["epochs"][0][0]
        timings = []
        for arm in models:
            t0 = time.monotonic()
            l = loss(arm, probe); l.backward(); opts[arm].zero_grad(set_to_none=True)
            torch.cuda.synchronize(); timings.append(time.monotonic() - t0)
        pair_s = max(sum(timings), .1)
        # Reserve a conservative validation/checkpoint allowance before choosing
        # a common schedule length. Never silently run one arm further.
        available = max(0, args.budget_seconds - elapsed() - 420)
        steps_per_epoch = min(steps_per_epoch, int(available / (MAX_EPOCHS * pair_s * 1.5)))
        if steps_per_epoch < 1: raise RuntimeError("Insufficient envelope for a matched experiment")
        C.save_json(out / "benchmark.json", dict(probe_seconds=timings, pair_s=pair_s,
                    steps_per_epoch=steps_per_epoch, budget_seconds=args.budget_seconds,
                    channels=R.CHANNELS, peak_vram_bytes=torch.cuda.max_memory_allocated()))
        checkpoint(0, 0)
        print("BENCHMARK", pair_s, "steps/epoch", steps_per_epoch, flush=True)
    stopped = False
    for ep in range(epoch0, MAX_EPOCHS):
        sums = {k:0.0 for k in models}
        lo = step0 if ep == epoch0 else 0
        for it in range(lo, steps_per_epoch):
            if elapsed() > args.budget_seconds - 150:
                checkpoint(ep, it); stopped = True; break
            ctx = schedule["epochs"][ep][it]
            for arm, model in models.items():
                R.train_modules(model, dss[arm])
                opts[arm].zero_grad(set_to_none=True)
                l = loss(arm, ctx)
                if not torch.isfinite(l): raise RuntimeError("Nonfinite loss")
                l.backward(); opts[arm].step()
                sums[arm] += float(l.detach())
            if it % 40 == 0:
                print("TRAIN", ep, it, "loss", {a:round(v/max(it-lo+1,1),3) for a,v in sums.items()},
                      "s", round(elapsed(),1), flush=True)
            if (it + 1) % 80 == 0: checkpoint(ep, it + 1)
        if stopped: break
        vals = {}
        for arm, model in models.items():
            model.eval(); vl=[]
            with torch.no_grad():
                for ctx in schedule["validation"]:
                    if elapsed() > args.budget_seconds - 30:
                        stopped = True; break
                    vl.append(float(loss(arm, ctx)))
            if stopped: break
            vals[arm] = float(np.mean(vl))
            if vals[arm] < best[arm]:
                best[arm] = vals[arm]
                torch.save(dict(weights=R.state(model, dss[arm]), epoch=ep,
                                validation_loss=best[arm], schedule_hash=sig), str(out / (arm + "_best.pt")))
        if stopped:
            checkpoint(ep, steps_per_epoch); break
        history.append(dict(epoch=ep, validation=vals, seconds=elapsed()))
        checkpoint(ep + 1, 0)
        print("VALIDATION", ep, vals, "s",round(elapsed(),1), flush=True)
    result = dict(schedule_hash=sig, history=history, best_validation=best,
                  steps_per_epoch=steps_per_epoch, epochs_completed=len(history),
                  steps_per_arm=steps_per_epoch * len(history), gpu_job_seconds=elapsed(),
                  status="budget_limited" if stopped else "complete", seed=0,
                  checkpoint_hashes={a:C.sha(out / (a + "_best.pt")) for a in models if (out / (a + "_best.pt")).exists()},
                  initialization_sha=C.sha(repo / "runs/uav3d_comm/fusion/ego_s2/decoder.pt"),
                  peak_vram_bytes=torch.cuda.max_memory_allocated())
    C.save_json(out / "result.json", result)
    print("MATCHED_DONE", json.dumps(result), flush=True)


if __name__ == "__main__": main()
