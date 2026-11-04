"""Resumable, single-GPU audit of Claude's proposed bridge recipe.

python -m swarmbridge.experiments.sb_audit.run --prepare
python -m swarmbridge.experiments.sb_audit.run --suite all

Main BEV pair: no-CSI mapping vs VE xt-only bridge, identical U-Net and task loss.
The CSI/residual mapping is an additional strong baseline, not the matched-input pair.
CIFAR-10 is a small I2SB-style protocol adaptation, NOT ImageNet paper reproduction.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

from ...bridges.audit import AuditDenoiser, VarianceClock
from ...channel import aggregate, sample_params, transmit
from ...data.real import CachedLatents
from ...teacher.model import IGNORE
from ..q2_progressive.run import EVAL_CFG, TRAIN_CFG, random_subset
from ..q2_retx.run import to_blocks, to_cells


VERSION = "sb_audit_ve_v1"
BEV_MODELS = ("mapping_xt", "sb_ve_xt_endpoint", "mapping_csi_residual")
IMAGE_MODELS = ("image_mapping", "image_i2sb")
FRACTIONS = (0.15, 0.3, 0.5, 0.7, 1.0)


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(name, channels, width, endpoint_fraction):
    if name == "mapping_xt":
        return AuditDenoiser(channels, bridge=False, width=width)
    if name == "sb_ve_xt_endpoint":
        return AuditDenoiser(channels, bridge=True, width=width, endpoint_fraction=endpoint_fraction)
    if name == "mapping_csi_residual":
        return AuditDenoiser(channels, bridge=False, csi=True, residual=True, width=width)
    if name == "image_mapping":
        return AuditDenoiser(channels, bridge=False, condition_source=True, csi=True, width=width)
    if name == "image_i2sb":
        return AuditDenoiser(channels, bridge=True, condition_source=True, csi=True, width=width,
                             prediction="noise", endpoint_fraction=0.0,
                             clock=VarianceClock(kind="brownian", q=1.0))
    raise ValueError(name)


_WINDOWS_HOLDER = r"""
import msvcrt, sys
f = open(sys.argv[1], 'a+b'); f.seek(0)
try:
    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
except OSError:
    print('BUSY', flush=True); sys.exit(1)
print('LOCKED', flush=True)
sys.stdin.read()
f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
"""


@contextlib.contextmanager
def _wsl_gpu_lock(path):
    """Under WSL, Linux flock is invisible to Windows msvcrt locks, so take the WINDOWS lock through a child
    Windows python that holds it until our stdin pipe closes (also when this process is killed)."""
    import subprocess
    windows_python = Path(".venv/Scripts/python.exe")
    holder = subprocess.Popen([str(windows_python), "-c", _WINDOWS_HOLDER, str(path)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if holder.stdout.readline().strip() != "LOCKED":
        holder.wait()
        raise RuntimeError("Another worker (Windows or WSL) holds the GPU lock")
    try:
        yield
    finally:
        holder.stdin.close()
        holder.wait()


@contextlib.contextmanager
def gpu_lock():
    """An OS lock is automatically released on exit; PID text is informational."""
    path = Path("runs/.swarmbridge-gpu.lock")
    path.parent.mkdir(exist_ok=True)
    if os.name != "nt" and "microsoft" in platform.release().lower():
        with _wsl_gpu_lock(path):
            yield
        return
    with path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another audit worker holds the GPU lock") from error
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        stream.write(str(os.getpid()).encode("ascii"))
        stream.truncate()
        stream.flush()
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class BevData:
    channels = 16

    def __init__(self, teacher, device):
        self.ds = CachedLatents(teacher, device, eval_split="val")
        self.ds.cfg = TRAIN_CFG
        self.channels, self.device = self.ds.C, device

    def batch(self, batch_size, generator):
        return self.ds.batch(batch_size, generator)

    def auxiliary_loss(self, prediction, batch):
        logits = self.ds.dec(prediction * self.ds.sd + self.ds.mu)
        return F.cross_entropy(logits.float(), batch["y"], ignore_index=IGNORE)

    def validation(self, limit):
        data = self.ds.eval
        return {key: data[key][:limit].to(self.device).float() for key in ("z", "fov48")}, data["y"][:limit].to(self.device)


class ImageData:
    channels = 3

    def __init__(self, root, device):
        data = torchvision.datasets.CIFAR10(root, train=True, download=False).data
        images = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous()
        # Fixed train/validation split, independent of experimental seeds.
        order = torch.randperm(len(images), generator=torch.Generator().manual_seed(20261030))
        self.train = images[order[:40000]].to(device)
        self.val = images[order[40000:]].to(device)
        self.device = device

    def corrupt(self, clean, generator):
        known = torch.ones(len(clean), 1, 32, 32, device=clean.device)
        known[:, :, 8:24, 8:24] = 0
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
        source = known * clean + (1 - known) * noise
        return source, known

    def batch(self, batch_size, generator):
        idx = torch.randint(len(self.train), (batch_size,), device=self.device, generator=generator)
        clean = self.train[idx].float() / 127.5 - 1
        source, known = self.corrupt(clean, generator)
        return dict(z_star=clean, z_p=source, r=known, loss_mask=1 - known)

    def auxiliary_loss(self, prediction, batch):
        return prediction.new_zeros(())


def confusion(prediction, target):
    valid = target != IGNORE
    return torch.bincount((3 * target[valid] + prediction[valid]).long(), minlength=9).reshape(3, 3)


def confusion_metrics(matrix):
    matrix = matrix.double()
    intersection = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    iou = intersection / union.clamp_min(1)
    return dict(miou=iou[union > 0].mean().item(), vehicle_iou=iou[2].item(), per_class_iou=iou.tolist())


@torch.no_grad()
def evaluate_bev(model, data, seed, limit, batch_size):
    values, target = data.validation(limit)
    clean, fov = values["z"], values["fov48"]
    generator = torch.Generator(device=data.device).manual_seed(3000 + seed)
    parameters = sample_params(EVAL_CFG, len(clean), 1, data.device, generator, None)
    candidates = to_blocks(fov) > 0
    results = {}
    modes = (("one_step", 1, False),) if not model.bridge else (
        ("one_step", 1, False), ("sde_4", 4, True), ("mean_path_4", 4, False))
    for fraction in FRACTIONS:
        selected = random_subset(candidates, fraction, torch.Generator(device=data.device).manual_seed(21 + seed))
        received = transmit(clean[:, None], torch.zeros_like(clean[:, None]),
                            (fov * to_cells(selected))[:, None], parameters, EVAL_CFG,
                            torch.Generator(device=data.device).manual_seed(31 + seed))
        source, reliability = aggregate(*received)
        row = {}
        for label, nfe, stochastic in modes:
            matrix = torch.zeros(3, 3, device=data.device, dtype=torch.long)
            error, count = 0.0, 0
            sampling = torch.Generator(device=data.device).manual_seed(44 + seed)
            begin = time.perf_counter()
            for offset in range(0, len(clean), batch_size):
                sl = slice(offset, offset + batch_size)
                with torch.autocast(data.device, dtype=torch.bfloat16, enabled=data.device == "cuda"):
                    output = model.sample(source[sl], reliability[sl], nfe=nfe,
                                          stochastic=stochastic, generator=sampling)
                    logits = data.ds.dec(output.float() * data.ds.sd + data.ds.mu)
                matrix += confusion(logits.argmax(1), target[sl])
                error += (output.float() - clean[sl]).square().sum().item()
                count += clean[sl].numel()
            if data.device == "cuda":
                torch.cuda.synchronize()
            row[label] = dict(**confusion_metrics(matrix), latent_mse=error / count,
                              inference_s=time.perf_counter() - begin, nfe=nfe)
        results[str(fraction)] = row
    return dict(split="val", n=len(clean), primary_mode="one_step", fractions=results)


def ssim(x, y):
    """11x11 Gaussian-window SSIM, [0,1], valid spatial support."""
    coordinates = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-coordinates.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(3, 1, 11, 11)
    ux, uy = F.conv2d(x, window, groups=3), F.conv2d(y, window, groups=3)
    vx = F.conv2d(x.square(), window, groups=3) - ux.square()
    vy = F.conv2d(y.square(), window, groups=3) - uy.square()
    covariance = F.conv2d(x * y, window, groups=3) - ux * uy
    result = ((2 * ux * uy + .01**2) * (2 * covariance + .03**2)) / (
        (ux.square() + uy.square() + .01**2) * (vx + vy + .03**2))
    return result.mean((1, 2, 3))


@torch.no_grad()
def evaluate_images(model, data, seed, limit, batch_size, job_dir, step, perceptual):
    clean_all = data.val[:limit].float() / 127.5 - 1
    modes = (("one_step", 1, False),) if not model.bridge else (
        ("one_step", 1, False), ("sde_4", 4, True), ("sde_16", 16, True))
    results = {}
    for label, nfe, stochastic in modes:
        generator = torch.Generator(device=data.device).manual_seed(998 + seed)
        sampling = torch.Generator(device=data.device).manual_seed(999 + seed)
        sums = dict(psnr=0.0, hole_psnr=0.0, ssim=0.0, lpips_alex_64=0.0)
        maximum_known_error, visual = 0.0, None
        for offset in range(0, len(clean_all), batch_size):
            clean = clean_all[offset:offset + batch_size]
            source, known = data.corrupt(clean, generator)
            with torch.autocast(data.device, dtype=torch.bfloat16, enabled=data.device == "cuda"):
                output = model.sample(source, known, nfe=nfe, stochastic=stochastic,
                                      generator=sampling, known=known, clip=True)
            output = output.float()
            x, y = (output + 1) / 2, (clean + 1) / 2
            error = (x - y).square()
            sums["psnr"] += (-10 * error.mean((1, 2, 3)).clamp_min(1e-12).log10()).sum().item()
            hole = (error * (1 - known)).sum((1, 2, 3)) / (3 * (1 - known).sum((1, 2, 3)))
            sums["hole_psnr"] += (-10 * hole.clamp_min(1e-12).log10()).sum().item()
            sums["ssim"] += ssim(x, y).sum().item()
            # Explicit resize because 32x32 is small for a standard perceptual network.
            px = F.interpolate(output, size=(64, 64), mode="bilinear", align_corners=False)
            py = F.interpolate(clean, size=(64, 64), mode="bilinear", align_corners=False)
            sums["lpips_alex_64"] += perceptual(px, py).sum().item()
            maximum_known_error = max(maximum_known_error, ((output - clean).abs() * known).max().item())
            if visual is None:
                # Each row: clean, damaged, reconstruction. The source noise is clipped only for display.
                visual = torch.stack([clean[:12], source[:12].clamp(-1, 1), output[:12]], 1).flatten(0, 1)
        results[label] = {key: value / len(clean_all) for key, value in sums.items()}
        results[label]["max_known_pixel_error"] = maximum_known_error
        torchvision.utils.save_image((visual + 1) / 2, job_dir / f"preview_{step}_{label}.png", nrow=3)
    return dict(split="CIFAR10 train-heldout val", n=len(clean_all), modes=results,
                scope="Small I2SB-style adaptation, not paper-scale reproduction or posterior calibration.")


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def capture_rng(data_generator, bridge_generator, device):
    return dict(data=data_generator.get_state().cpu(), bridge=bridge_generator.get_state().cpu(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if device == "cuda" else None)


def restore_rng(state, data_generator, bridge_generator, device):
    data_generator.set_state(state["data"].cpu())
    bridge_generator.set_state(state["bridge"].cpu())
    torch.set_rng_state(state["cpu"].cpu())
    if device == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def train_until(name, data, seed, stop_step, args, status, perceptual=None):
    image = name in IMAGE_MODELS
    job = f"{name}_s{seed}"
    directory = Path(args.out) / job
    directory.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.manual_seed(seed)
    model = make_model(name, data.channels, args.width, args.endpoint_fraction).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
    data_generator = torch.Generator(device=device).manual_seed(100 + seed)
    bridge_generator = torch.Generator(device=device).manual_seed(1000 + seed)
    checkpoint = directory / "last.pt"
    step, elapsed = 0, 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        ema = {key: value.to(device) for key, value in saved["ema"].items()}
        step, elapsed = saved["step"], saved["train_s"]
        restore_rng(saved["rng"], data_generator, bridge_generator, device)
        print(f"RESUME {job}: step {step}", flush=True)
    if step >= stop_step and (directory / f"evaluation_{stop_step}.json").exists():
        print(f"SKIP {job}: {step} >= {stop_step}", flush=True)
        return
    if step > stop_step:
        # An older milestone has already been passed; never label later weights as earlier.
        print(f"SKIP old milestone {job}: current step {step} > {stop_step}", flush=True)
        return
    model.train()
    microbatch = args.image_batch if image else args.microbatch
    batch = args.image_batch if image else args.batch
    if batch % microbatch:
        raise ValueError("batch must be divisible by microbatch")
    updates = batch // microbatch
    lam = 0.0 if image else args.task_weight
    interval_begin, run_begin = time.perf_counter(), time.perf_counter()
    averages = [0.0, 0.0, 0]

    def save():
        if device == "cuda":
            torch.cuda.synchronize()
        total_elapsed = elapsed + time.perf_counter() - run_begin
        atomic_checkpoint(checkpoint, dict(model=cpu_state(model),
            ema={k: v.detach().cpu().clone() for k, v in ema.items()},
            optimizer=optimizer.state_dict(), step=step, train_s=total_elapsed,
            rng=capture_rng(data_generator, bridge_generator, device),
            version=VERSION, name=name, seed=seed))
        status.update(state="training", job=job, step=step, target_step=stop_step,
                      updated_at=now(), job_train_s=total_elapsed,
                      checkpoint=str(checkpoint), pid=os.getpid())
        atomic_json(Path(args.out) / "status.json", status)

    try:
        while step < stop_step:
            optimizer.zero_grad(set_to_none=True)
            latent_value, auxiliary_value = 0., 0.
            for _ in range(updates):
                batch_data = data.batch(microbatch, data_generator)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = model.loss(batch_data["z_p"], batch_data["r"], batch_data["z_star"],
                                      generator=bridge_generator, loss_mask=batch_data.get("loss_mask"))
                    auxiliary = data.auxiliary_loss(model.last_pred, batch_data) if lam else loss.new_zeros(())
                    combined = (loss + lam * auxiliary) / updates
                if not torch.isfinite(combined):
                    raise FloatingPointError(f"Non-finite loss in {job} at step {step}")
                combined.backward()
                latent_value += loss.detach().item() / updates
                auxiliary_value += auxiliary.detach().item() / updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    ema[key].lerp_(value, 1 - args.ema)
            step += 1
            averages[0] += latent_value
            averages[1] += auxiliary_value
            averages[2] += 1
            if step % args.log_every == 0 or step == stop_step:
                if device == "cuda":
                    torch.cuda.synchronize()
                record = dict(step=step, reconstruction_loss=averages[0] / averages[2],
                              task_ce=averages[1] / averages[2],
                              interval_s=time.perf_counter() - interval_begin,
                              updates=averages[2], updated_at=now())
                with (directory / "learning_curve.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(f"{job} {step}/{stop_step} loss={record['reconstruction_loss']:.5f} "
                      f"CE={record['task_ce']:.5f} seconds/update={record['interval_s']/record['updates']:.3f}",
                      flush=True)
                interval_begin = time.perf_counter()
                averages = [0.0, 0.0, 0]
                status.update(state="training", job=job, step=step, target_step=stop_step,
                              updated_at=now(), pid=os.getpid())
                atomic_json(Path(args.out) / "status.json", status)
            if step % args.save_every == 0 or step == stop_step:
                save()
    except BaseException:
        # Only save completed optimizer steps. Partial accumulated gradients are not checkpointed.
        optimizer.zero_grad(set_to_none=True)
        raise
    status.update(state="evaluating", job=job, step=step, target_step=stop_step, updated_at=now())
    atomic_json(Path(args.out) / "status.json", status)
    model.load_state_dict(ema)
    model.eval()
    evaluation = evaluate_images(model, data, seed, args.image_eval, args.eval_batch,
                                 directory, step, perceptual) if image else evaluate_bev(
                                     model, data, seed, args.bev_eval, args.eval_batch)
    atomic_json(directory / f"evaluation_{step}.json", evaluation)
    atomic_checkpoint(directory / f"ema_{step}.pt", cpu_state(model))
    print(f"EVALUATED {job} at {step}: {json.dumps(evaluation)[:650]}", flush=True)
    del model, optimizer, ema
    if device == "cuda":
        torch.cuda.empty_cache()


def prepare(args):
    root = Path(args.image_root)
    torchvision.datasets.CIFAR10(root, train=True, download=True)
    torchvision.datasets.CIFAR10(root, train=False, download=True)
    import lpips
    lpips.LPIPS(net="alex", verbose=False).eval()
    print("Prepared CIFAR-10 and LPIPS AlexNet weights. No training.", flush=True)


def manifest(args):
    settings = vars(args).copy()
    for key in ("prepare",):
        settings.pop(key, None)
    files = ["swarmbridge/bridges/audit.py", "swarmbridge/bridges/backbone.py",
             "swarmbridge/channel/channel.py", "swarmbridge/data/real.py",
             "swarmbridge/teacher/model.py", "swarmbridge/experiments/sb_audit/run.py",
             "swarmbridge/experiments/q2_progressive/run.py", "swarmbridge/experiments/q2_retx/run.py"]
    provenance = {p: fingerprint(p) for p in files}
    teacher_files = ("teacher.pt", "latent_stats.pt", "latents_train.pt", "latents_val.pt")
    if args.suite in ("all", "bev"):
        provenance.update({str(Path(args.teacher) / p): fingerprint(Path(args.teacher) / p) for p in teacher_files})
    value = dict(version=VERSION, settings=settings, hashes=provenance,
                 environment=dict(torch=torch.__version__, torchvision=torchvision.__version__,
                                  python=platform.python_version(),
                                  gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None),
                 ve=asdict(VarianceClock()), sources=[
                     "https://www.isca-archive.org/interspeech_2024/jukic24_interspeech.pdf",
                     "https://arxiv.org/html/2510.16834v2",
                     "https://github.com/NVlabs/I2SB/blob/master/i2sb/runner.py",
                     "https://github.com/NVlabs/I2SB/blob/master/i2sb/diffusion.py"],
                 limitations=["BEV adaptation uses a U-Net and segmentation CE, not the speech architecture/loss.",
                              "Endpoint fraction is an audit design choice, not a claimed paper hyperparameter.",
                              "CIFAR diagnostic is not reproduction of I2SB ImageNet results.",
                              "Validation-only audit; no test-set recipe selection.",
                              "Finite-step sample variance is not assumed calibrated."])
    path = Path(args.out) / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != value:
            raise RuntimeError("Configuration/code/data differ from existing manifest; choose a new --out")
    else:
        atomic_json(path, value)


def write_summary(args):
    output = Path(args.out)
    results = {}
    for file in sorted(output.glob("*/evaluation_*.json")):
        results[str(file.relative_to(output))] = json.loads(file.read_text(encoding="utf-8"))
    atomic_json(output / "results.json", results)
    lines = ["**SB recipe audit — running results**", "",
             f"Updated {now()}. Experiment: {output}.", "",
             "The primary BEV comparison is one-step mapping_xt vs sb_ve_xt_endpoint.",
             "mapping_csi_residual is the strong CSI/residual baseline. No statistical conclusion from a partial queue.",
             "", "| Job / checkpoint | Sent | Mode | mIoU | Vehicle IoU |",
             "|---|---:|---|---:|---:|"]
    for name, result in results.items():
        if "fractions" in result:
            for fraction, modes in result["fractions"].items():
                for mode, values in modes.items():
                    lines.append(f"| {name} | {float(fraction):.0%} | {mode} | {values['miou']:.4f} | {values['vehicle_iou']:.4f} |")
    lines += ["", "Image diagnostics (CIFAR validation; not paper-scale reproduction):", "",
              "| Job / checkpoint | Mode | PSNR | Hole PSNR | SSIM | LPIPS AlexNet, resized 64 |",
              "|---|---|---:|---:|---:|---:|"]
    for name, result in results.items():
        if "modes" in result:
            for mode, values in result["modes"].items():
                lines.append(f"| {name} | {mode} | {values['psnr']:.3f} | {values['hole_psnr']:.3f} | {values['ssim']:.4f} | {values['lpips_alex_64']:.4f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "bev", "images"), default="all")
    parser.add_argument("--out", default=f"runs/{VERSION}")
    parser.add_argument("--teacher", default="runs/teacher")
    parser.add_argument("--image-root", default="data/cifar10")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--image-steps", type=int, default=10000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--image-batch", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--bev-eval", type=int, default=200)
    parser.add_argument("--image-eval", type=int, default=1000)
    parser.add_argument("--endpoint-fraction", type=float, default=.1)
    parser.add_argument("--task-weight", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ema", type=float, default=.999)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--prepare", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(Path("data/torch_hub").resolve()))
    torch.set_num_threads(4)
    if args.prepare:
        prepare(args)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the scheduled GPU audit")
    torch.backends.cudnn.benchmark = args.device == "cuda"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    status = dict(state="starting", started_at=now(), pid=os.getpid())
    with gpu_lock():
        manifest(args)
        atomic_json(Path(args.out) / "status.json", status)
        try:
            if args.suite in ("all", "images"):
                import lpips
                perceptual = lpips.LPIPS(net="alex", verbose=False).to(args.device).eval()
                perceptual.requires_grad_(False)
                data = ImageData(args.image_root, args.device)
                for seed in args.seeds:
                    for name in IMAGE_MODELS:
                        train_until(name, data, seed, args.image_steps, args, status, perceptual)
                        write_summary(args)
                del data, perceptual
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            if args.suite in ("all", "bev"):
                data = BevData(args.teacher, args.device)
                milestones = sorted(set([min(args.steps, 6000), min(args.steps, 15000), args.steps]))
                for stage in milestones:
                    for seed in args.seeds:
                        for name in BEV_MODELS:
                            train_until(name, data, seed, stage, args, status)
                            write_summary(args)
            status.update(state="completed", updated_at=now())
            atomic_json(Path(args.out) / "status.json", status)
            write_summary(args)
            print("AUDIT COMPLETED", flush=True)
        except BaseException as error:
            status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=f"{type(error).__name__}: {error}", updated_at=now())
            atomic_json(Path(args.out) / "status.json", status)
            write_summary(args)
            raise


if __name__ == "__main__":
    main()
