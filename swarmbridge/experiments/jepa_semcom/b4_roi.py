"""B-E8 (review #34): causal task-aware / event-aware spatial bit allocation inside a conventional low-delay HEVC
encoder, on the VisDrone-VID frames of B-E7, evaluated with the same receiver (decoder -> frozen DINOv2-S -> fixed
linear patch head) at matched total bitstream rate.

Encoder: x265roi.c (libx265 4.1 API, x265_picture.quantOffsets = one QP offset per 16 x 16 block per frame, added by
x265 on top of CRF rate control + adaptive quantisation), with exactly the B-E7 HEVC setting (preset slow, tune
zerolatency, keyint inf, bframes 0, rc-lookahead 0, scenecut 0, info 0). RGB -> yuv420p by ffmpeg (same swscale
defaults as B-E7), decoded by ffmpeg. The QP offsets are coded in the HEVC bitstream itself (cu_qp_delta), so the
receiver needs no side information: the bit count is the whole raw stream (parameter sets + all frames), as in B-E7.

Importance map of frame t (computed at the sender from frames <= t only; frame t is available when it is encoded):
  uniform    no offsets (plain CRF; reference)
  motion     global-motion-compensated luma residual |I_t - shift(I_t-1)| (integer global shift by phase correlation,
             camera motion removed), block mean / 16 grey levels, clipped to 1; frame 0: no offsets
  task       sender runs DINOv2-S + the fixed head on raw frame t: p = max-class sigmoid per 14 x 14 patch, block max
  taskev     event-sensitive: max(p_t, |p_t - shift(p_t-1)|) (task-score change after global-motion compensation:
             appearing / disappearing / moving objects), block max
  oracle     DIAGNOSTIC ONLY (not deployable): ground-truth patch labels of frame t (any class), block max
Offsets (rank allocation, strength A, fraction f): the blocks with importance > 0.1, at most round(f * #blocks) of
them with the highest importance, get -A; all other blocks get +A * n_top / (n - n_top) (mean offset 0 in QP units);
no eligible block -> no offsets. The resulting bit rate is not fixed; methods are compared at matched ACTUAL bits by
sweeping CRF and interpolating per clip in log(bits) (as in B-E7).
Protocol: (A, f) of each method is chosen on DEV clips (split-v1 TUNE clips, never evaluation clips) by the mean of
matched-bit AP and changed-patch AP at 6 / 9 / 12 kbit/frame (DEV_CLIPS: the 4 longest TUNE clips from distinct
source videos, 100 frames each); the chosen setting is then run once on the 10 B-E7
evaluation clips. Caveat: the head was fitted on TUNE clips, so dev-clip head scores are optimistic.
Sender cost: motion = numpy time per frame here; task / taskev = one DINOv2-S forward per frame (measured on this GPU,
batch 1, fp16; a UAV would need an embedded accelerator) + negligible map / offset time.

  run      CLIPS VID_ROOT MODEL_DIR TOKENS_DIR B1_WORK OUT_DIR --role dev|eval [--methods ..] [--grid A:f,..]
           (GPU, under the lock; encodes in a CPU pool, scores decoded frames on the GPU)
  eval     TOKENS_DIR CLIPS OUT_DIR OUT_JSON --role dev|eval [--select DEV_JSON]
"""
import argparse
import json
import math
import subprocess
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import b0_pipeline as B0
import b3_video as B3

H_IMG, W_IMG, DIM, FPS = B3.H_IMG, B3.W_IMG, B3.DIM, B3.FPS
BS, PATCH = 16, 14
BH, BW = math.ceil(H_IMG / BS), math.ceil(W_IMG / BS)
GH, GW = H_IMG // PATCH, W_IMG // PATCH
X265_PARAMS = ["keyint=-1", "min-keyint=1", "scenecut=0", "bframes=0", "rc-lookahead=0", "info=0", "log-level=error"]
BIN = Path.home() / "work" / "jepa_semcom" / "bin" / "x265roi"
ELIG = 0.1
METHODS = ("uniform", "motion", "task", "taskev", "oracle")
DEV_CLIPS = "uav0000288_00001_v,uav0000099_02109_v,uav0000273_00001_v,uav0000243_00001_v"   # 4 longest TUNE clips of
#                                                              distinct source videos (none is an evaluation source)


# ---------------------------------------------------------------- encoder adapter
def to_yuv(frames_rgb):
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W_IMG}x{H_IMG}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
                          input=np.ascontiguousarray(frames_rgb).tobytes(), capture_output=True, check=True).stdout


def code_roi(frames_rgb, crf, qoff, tmp, binp=BIN, tag="v", yuv=None, T=None):
    """frames (T, H, W, 3) uint8 (or yuv: the to_yuv bytes and T), qoff None or (T, BH, BW) float -> (decoded frames,
    total bits, per-frame bits with the parameter sets counted in frame 0, encode seconds)."""
    tmp = Path(tmp)
    out, qf = tmp / f"{tag}.hevc", tmp / f"{tag}.qoff"
    T = len(frames_rgb) if yuv is None else T
    if qoff is not None:
        assert qoff.shape == (T, BH, BW)
        np.ascontiguousarray(qoff, np.float32).tofile(qf)
    if yuv is None:
        yuv = to_yuv(frames_rgb)
    t0 = time.time()
    r = subprocess.run([str(binp), str(W_IMG), str(H_IMG), str(FPS), str(crf), str(qf) if qoff is not None else "-",
                        str(out)] + X265_PARAMS, input=yuv, capture_output=True, check=True)
    enc_s = time.time() - t0
    hdr, pk = 0, []
    for ln in r.stderr.decode().splitlines():
        w = ln.split()
        if w and w[0] == "hdr":
            hdr = int(w[1]) * 8
        elif w and w[0] == "pkt":
            assert int(w[1]) == len(pk)
            pk.append(int(w[2]) * 8)
    assert len(pk) == T
    pk = np.asarray(pk, np.int64)
    pk[0] += hdr
    total = out.stat().st_size * 8
    assert total == pk.sum(), (total, pk.sum())
    dec = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(out), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    d = np.frombuffer(dec, np.uint8)
    assert d.size == T * H_IMG * W_IMG * 3
    out.unlink()
    if qoff is not None:
        qf.unlink()
    return d.reshape(T, H_IMG, W_IMG, 3), int(total), pk, enc_s


# ---------------------------------------------------------------- importance maps (causal)
def gray(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def global_shift(g0, g1, max_shift=40):
    """integer (dy, dx) with g1(y, x) ~ g0(y - dy, x - dx), phase correlation on mean-removed, windowed luma."""
    win = np.outer(np.hanning(g0.shape[0]), np.hanning(g0.shape[1]))
    F0, F1 = np.fft.fft2((g0 - g0.mean()) * win), np.fft.fft2((g1 - g1.mean()) * win)
    R = F1 * np.conj(F0)
    r = np.real(np.fft.ifft2(R / np.maximum(np.abs(R), 1e-9)))
    H, W = r.shape
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    dy, dx = (dy - H if dy > H // 2 else dy), (dx - W if dx > W // 2 else dx)
    if abs(dy) > max_shift or abs(dx) > max_shift:
        return 0, 0
    return int(dy), int(dx)


def shift_img(a, dy, dx):
    """b(y, x) = a(y - dy, x - dx), edge-replicated."""
    H, W = a.shape[:2]
    yi = np.clip(np.arange(H) - dy, 0, H - 1)
    xi = np.clip(np.arange(W) - dx, 0, W - 1)
    return a[yi][:, xi]


def block_pool(a, how="mean"):
    H, W = a.shape
    p = np.pad(a, ((0, BH * BS - H), (0, BW * BS - W)), mode="edge").reshape(BH, BS, BW, BS)
    return p.mean((1, 3)) if how == "mean" else p.max((1, 3))


def patch_to_pix(p):
    return np.repeat(np.repeat(p.reshape(GH, GW), PATCH, 0), PATCH, 1)


def shifts(rgb):
    g = gray(rgb.astype(np.float32))
    return [(0, 0)] + [global_shift(g[t - 1], g[t]) for t in range(1, len(rgb))], g


def imp_motion(rgb, sh=None, g=None):
    if sh is None:
        sh, g = shifts(rgb)
    out = np.zeros((len(rgb), BH, BW), np.float32)
    for t in range(1, len(rgb)):
        res = np.abs(g[t] - shift_img(g[t - 1], *sh[t]))
        out[t] = np.minimum(block_pool(res, "mean") / 16.0, 1.0)
    return out


def task_prob(scores):
    return 1.0 / (1.0 + np.exp(-scores.reshape(len(scores), -1, 4).max(-1)))     # (T, 858)


def imp_task(scores):
    p = task_prob(scores)
    return np.stack([block_pool(patch_to_pix(p[t]), "max") for t in range(len(p))]).astype(np.float32)


def imp_taskev(scores, sh):
    p = task_prob(scores)
    out = []
    for t in range(len(p)):
        cur = patch_to_pix(p[t])
        m = cur if t == 0 else np.maximum(cur, np.abs(cur - shift_img(patch_to_pix(p[t - 1]), *sh[t])))
        out.append(block_pool(m, "max"))
    return np.stack(out).astype(np.float32)


def imp_oracle(labels, valid):
    T = len(valid)
    y = ((labels.reshape(T, -1, 4) > 0).any(-1) & valid.reshape(T, -1).astype(bool)).astype(np.float32)
    return np.stack([block_pool(patch_to_pix(y[t]), "max") for t in range(T)]).astype(np.float32)


def offsets(imp, A, f):
    """rank allocation (module docstring); imp (T, BH, BW) -> (T, BH, BW) QP offsets."""
    T = len(imp)
    n = BH * BW
    q = np.zeros((T, n), np.float32)
    for t in range(T):
        v = imp[t].reshape(-1)
        elig = int((v > ELIG).sum())
        k = min(int(round(f * n)), elig)
        if k == 0:
            continue
        order = np.lexsort((np.arange(n), -v))
        q[t] = A * k / (n - k)
        q[t, order[:k]] = -A
    return q.reshape(T, BH, BW)


# ---------------------------------------------------------------- run
def _enc_job(j):
    yuv_path, T, crf, qoff, key = j
    with tempfile.TemporaryDirectory() as tmp:
        dec, tot, pk, es = code_roi(None, crf, qoff, tmp, tag=key, yuv=Path(yuv_path).read_bytes(), T=T)
    return key, dec, tot, pk, es


def configs(methods, grid):
    out = []
    for m in methods:
        if m == "uniform":
            out.append(("uniform", 0.0, 0.0))
        else:
            out += [(m, A, f) for A, f in grid]
    return out


def cfg_key(m, A, f):
    return "uniform" if m == "uniform" else f"{m}_A{A:g}_f{f:g}"


def run(a):
    import torch
    from PIL import Image
    from transformers import Dinov2Model
    dev = "cuda"
    mdl = Dinov2Model.from_pretrained(a.model_dir, use_safetensors=True).eval().to(dev).half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    Hh = np.load(Path(a.b1_work) / "head.npz")
    W, b = Hh["W"], Hh["b"]

    def scores(frames, bs=16):
        out = []
        for i in range(0, len(frames), bs):
            with torch.no_grad():
                xt = torch.from_numpy(np.stack(frames[i:i + bs])).to(dev).permute(0, 3, 1, 2).float().div(255)
                tk = mdl(pixel_values=((xt - mean) / std).half()).last_hidden_state[:, 1:, :].float().cpu().numpy()
            out.append(tk.astype(np.float16).astype(np.float32) @ W + b)
        return np.concatenate(out)

    clips_all = json.loads(Path(a.clips).read_text())
    if a.role == "dev":
        names = [(c, "tune") for c in a.dev_clips.split(",")]
        assert all(c in clips_all["tune"] for c, _ in names)
    else:
        names = B3.eval_clips(a.clips, a.tokens_dir)
    grid = [tuple(float(x) for x in g.split(":")) for g in a.grid.split(",")]
    cfgs = configs(a.methods.split(","), grid)
    if a.select:
        sel = json.loads(Path(a.select).read_text())["selected"]
        cfgs = [("uniform", 0.0, 0.0)] + [(m, s["A"], s["f"]) for m, s in sel.items()]
    crfs = [int(x) for x in a.crfs.split(",")]
    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    stats = json.loads((outd / "run_stats.json").read_text()) if (outd / "run_stats.json").exists() else {}
    # sender-side cost: DINOv2-S, batch 1, fp16 on this GPU
    x1 = [np.zeros((H_IMG, W_IMG, 3), np.uint8)]
    for _ in range(5):
        scores(x1, 1)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50):
        scores(x1, 1)
    torch.cuda.synchronize()
    cost = dict(dinov2_head_ms_batch1=(time.perf_counter() - t0) / 50 * 1e3)
    t_all = time.time()
    with Pool(a.procs) as pool:
        for c, role in names:
            if (outd / f"{c}.npz").exists():
                continue
            D = np.load(Path(a.tokens_dir) / f"{c}.npz")
            split = "val" if role == "aux_val" else "train"
            files = sorted((Path(a.vid_root) / f"VisDrone2019-VID-{split}" / "sequences" / c).glob("*.jpg"))[::3][:100]
            assert np.array_equal(np.array([int(f.stem) for f in files]), D["frames"]), c
            rgb = np.stack([np.asarray(Image.open(f).convert("RGB").resize((W_IMG, H_IMG), Image.BICUBIC)) for f in files])
            s_orig = scores(list(rgb))                         # = what a sender running DINOv2 + head computes
            chk = float(np.abs(s_orig.reshape(-1, 4) - (D["tokens"].reshape(-1, DIM).astype(np.float32) @ W + b)).max())
            t0 = time.perf_counter()
            sh, g = shifts(rgb)
            imps = {"motion": imp_motion(rgb, sh, g)}
            cost.setdefault("motion_ms_per_frame", []).append((time.perf_counter() - t0) / len(rgb) * 1e3)
            t0 = time.perf_counter()
            imps["task"] = imp_task(s_orig)
            imps["taskev"] = imp_taskev(s_orig, sh)
            cost.setdefault("taskmap_ms_per_frame", []).append((time.perf_counter() - t0) / len(rgb) * 1e3 / 2)
            imps["oracle"] = imp_oracle(D["labels"], D["valid"])
            yuv_path = outd / f"_{c}.yuv"
            yuv_path.write_bytes(to_yuv(rgb))
            jobs = []
            for m, A, f in cfgs:
                qo = None if m == "uniform" else offsets(imps[m], A, f)
                for crf in crfs:
                    jobs.append((str(yuv_path), len(rgb), crf, qo, f"{cfg_key(m, A, f)}_c{crf}"))
            res = dict(orig_scores=s_orig.astype(np.float32), shifts=np.array(sh))
            st = dict(role=role, frames=len(rgb), orig_score_maxdiff=chk, points={},
                      top_frac={m: float((offsets(imps[m], *grid[0]) < 0).mean()) for m in imps})
            for key, dec, tot, pk, es in pool.imap_unordered(_enc_job, jobs):
                res[f"{key}_scores"] = scores(list(dec)).astype(np.float32)
                res[f"{key}_pkt_bits"] = pk
                psnr = float(np.mean([10 * np.log10(255 ** 2 / max(np.mean((dec[i].astype(np.float32) - rgb[i]) ** 2), 1e-9))
                                      for i in range(len(rgb))]))
                st["points"][key] = dict(bits_per_frame=tot / len(rgb), first_pkt=int(pk[0]), psnr=psnr, enc_s=es)
            yuv_path.unlink()
            np.savez(outd / f"{c}.npz", **res)
            stats[c] = st
            (outd / "run_stats.json").write_text(json.dumps(stats, indent=1))
            print(c, role, len(rgb), f"maxdiff {chk:.2g}", len(jobs), "encodes", f"{time.time() - t_all:.0f}s", flush=True)
    cost = {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in cost.items()}
    stats["_cost"] = cost
    (outd / "run_stats.json").write_text(json.dumps(stats, indent=1))
    print("cost", cost)


# ---------------------------------------------------------------- evaluation
def evaluate(a):
    import b2_pipeline as B2
    stats = json.loads((Path(a.out_dir) / "run_stats.json").read_text())
    names = [c for c in stats if not c.startswith("_") and stats[c]["role"] in (("tune",) if a.role == "dev" else ("gate", "aux_val"))]
    targets = [float(x) for x in a.targets.split(",")]
    curves = {}
    keys = sorted({k.rsplit("_c", 1)[0] for c in names for k in stats[c]["points"]})
    orig = {}
    for c in names:
        D = np.load(Path(a.tokens_dir) / f"{c}.npz")
        V = np.load(Path(a.out_dir) / f"{c}.npz")
        T = len(D["frames"])
        Y, M = D["labels"].reshape(-1, 4), D["valid"].reshape(-1).astype(bool)
        orig[c] = B3.metrics(V["orig_scores"], Y, M, T)
        for key, p in stats[c]["points"].items():
            ck = key.rsplit("_c", 1)[0]
            curves.setdefault((ck, c), []).append((p["bits_per_frame"], B3.metrics(V[f"{key}_scores"], Y, M, T), p["psnr"]))
    out = dict(role=a.role, clips=names, targets=targets, cost=stats.get("_cost"),
               original_tokens={k: float(np.mean([orig[c][k] for c in names])) for k in orig[names[0]]},
               matched={}, diffs={})
    for metric in ("ap", "ap_change", "ap_event", "psnr"):
        for t in targets:
            tab = {}
            for ck in keys:
                cv = [curves[(ck, c)] for c in names]
                tab[ck] = [B3.interp_log([x[0] for x in v], [(x[2] if metric == "psnr" else x[1][metric]) for x in v], t) for v in cv]
            common = np.all([np.isfinite(v) for v in tab.values()], 0)       # clips where every method is in range
            out["matched"][f"{metric}@{t:g}"] = {ck: dict(mean=float(np.nanmean(v)) if np.isfinite(v).any() else None,
                                                         n=int(np.isfinite(v).sum()), per_clip=[float(x) for x in v],
                                                         mean_common=float(np.mean(np.asarray(v)[common])) if common.any() else None,
                                                         n_common=int(common.sum()))
                                                 for ck, v in tab.items()}
            for ck in keys:
                if ck != "uniform":
                    out["diffs"][f"{metric}@{t:g}:{ck}-uniform"] = B3.ci(np.array(tab[ck]) - np.array(tab["uniform"]))
            tk = [k for k in keys if k.startswith("task_A")]
            te = [k for k in keys if k.startswith("taskev_A")]
            if len(tk) == 1 and len(te) == 1:
                out["diffs"][f"{metric}@{t:g}:{te[0]}-{tk[0]}"] = B3.ci(np.array(tab[te[0]]) - np.array(tab[tk[0]]))
    # bit saving at equal quality: per clip, uniform metric at target bits -> bits the method needs for that value
    out["bit_saving"] = {}
    for metric in ("ap", "ap_change", "ap_event"):
        for t in targets:
            for ck in keys:
                if ck == "uniform":
                    continue
                sv = []
                for c in names:
                    u = curves[("uniform", c)]
                    target_v = B3.interp_log([x[0] for x in u], [x[1][metric] for x in u], t)
                    m = sorted(curves[(ck, c)])
                    v = np.array([x[1][metric] for x in m])
                    lb = np.log([x[0] for x in m])
                    if not np.isfinite(target_v) or np.any(np.diff(v) <= 0) or not v[0] <= target_v <= v[-1]:
                        sv.append(np.nan)
                        continue
                    sv.append(1.0 - float(np.exp(np.interp(target_v, v, lb))) / t)
                out["bit_saving"][f"{metric}@{t:g}:{ck}"] = B3.ci(sv)
    if a.role == "dev":                                  # selection of (A, f) per method
        sel = {}
        for m in METHODS[1:]:
            best = None
            for ck in keys:
                if not ck.startswith(m + "_A"):
                    continue
                vals = [out["diffs"].get(f"{met}@{t:g}:{ck}-uniform", {}).get("mean") for met in ("ap", "ap_change")
                        for t in (6000, 9000, 12000)]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                sc = float(np.mean(vals))
                if best is None or sc > best[0]:
                    best = (sc, ck)
            if best:
                A, f = best[1].split("_A")[1].split("_f")
                sel[m] = dict(A=float(A), f=float(f), key=best[1], score=float(best[0]))
        out["selected"] = sel
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("original tokens", out["original_tokens"], "cost", out["cost"])
    for metric in ("ap", "ap_change", "ap_event"):
        for t in targets:
            row = out["matched"][f"{metric}@{t:g}"]
            print(f"{metric:9s}@{t / 1e3:4.1f}k (common clips {row[keys[0]]['n_common']})",
                  " ".join(f"{ck}:{row[ck]['mean_common']:.3f}" if row[ck]["mean_common"] is not None else f"{ck}:-" for ck in keys))
    for k, v in out["diffs"].items():
        if k.split("@")[0] in ("ap", "ap_change", "ap_event") and v["mean"] is not None and v["lo"] is not None:
            print(f"{k:48s} {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] pos {v['pos']}/{v['n']}")
    for k, v in out["bit_saving"].items():
        if v["mean"] is not None and v["lo"] is not None and "_A" in k and (a.role == "eval" or k.startswith("ap@")):
            print(f"saving {k:44s} {v['mean'] * 100:+.1f}% [{v['lo'] * 100:+.1f},{v['hi'] * 100:+.1f}] n {v['n']}")
    if "selected" in out:
        print("selected", out["selected"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for n in ("clips", "vid_root", "model_dir", "tokens_dir", "b1_work", "out_dir"):
        r.add_argument(n)
    r.add_argument("--role", choices=("dev", "eval"), required=True)
    r.add_argument("--dev-clips", default=DEV_CLIPS)
    r.add_argument("--methods", default=",".join(METHODS))
    r.add_argument("--grid", default="4:0.15,4:0.3,8:0.15,8:0.3")
    r.add_argument("--crfs", default="30,33,36,39,42,45")
    r.add_argument("--select", default="")
    r.add_argument("--procs", type=int, default=3)
    e = sub.add_parser("eval")
    for n in ("tokens_dir", "clips", "out_dir", "out"):
        e.add_argument(n)
    e.add_argument("--role", choices=("dev", "eval"), required=True)
    e.add_argument("--targets", default="4500,6000,9000,12000,18000")
    a = ap.parse_args()
    dict(run=run, eval=evaluate)[a.cmd](a)


if __name__ == "__main__":
    main()
