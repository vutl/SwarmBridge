"""B-E7 (review #33): inter-frame video codec baselines (HEVC / AV1) on the exact evaluated VisDrone-VID frames, and a
per-frame closed-loop check of the B-E6 training rate surrogate.

Video baseline (sender: video encoder; receiver: decoder -> frozen DINOv2-S -> the SAME fixed linear patch head):
  * frames: exactly the frames of the B-E1 token extraction of each evaluation clip (sorted *.jpg, stride 3, at most
    100), PIL bicubic resize to 546 x 308 (identical to b1_pipeline.extract); frame ids are checked against the token
    file, and the original-frame tokens recomputed here are checked against the stored ones.
  * the frames are encoded as ONE causal low-latency video per clip and rate point: first frame intra, then only
    P / inter frames (no B-frames, no lookahead, no scene-cut keyframes, no temporal filtering with future frames),
    so the decoded frame t depends only on frames <= t; constant-quality rate control (CRF) per rate point.
      hevc: libx265 --preset slow --tune zerolatency, keyint inf, bframes 0, rc-lookahead 0, scenecut 0, info 0 (no
            encoder-version SEI), raw Annex-B stream (VPS/SPS/PPS once).
      av1 : libsvtav1 preset 6, pred-struct 1 (low delay), lookahead 0, scd 0, keyint inf, raw OBU stream (sequence
            header and temporal delimiters included).
    Encoders run with their default threading; slow / good-quality presets are used on purpose (a strong compression
    reference, not a real-time UAV encoder).
  * bits: the size of the complete raw bitstream file (every header, keyframe, inter frame and OBU / NAL overhead)
    divided by the number of frames; per-frame packet sizes (ffprobe) are stored as well.
  * colour: rgb24 -> yuv420p -> rgb24 with ffmpeg's default (BT.601) matrices in both directions.
Head scores (T, 858, 4) of each decoded sequence are stored (not the tokens), evaluated with the B-E6 metrics (AP on all
valid patches, on changed patches and on event frames).

--steady: the same comparison with rates that exclude the first frame (video: first packet = parameter sets +
keyframe; semantic: the initial all-token message), i.e. the steady-state cost of a long stream; AP still over all
frames.
Comparison at matched ACTUAL bits per frame against the semantic codec rows of a b2_pipeline eval JSON (mc predictor,
k = 16..256): per clip, metric interpolated in log(bits); semantic representations averaged over their 3 training
seeds per clip; differences with a t-interval over the 10 clips.

Surrogate check (surrdiag): closed-loop transmission (b1_pipeline.transmit, mc predictor, k in --ks) of the codes of
each representation; per frame t >= 1: actual message bits vs (a) the TRAINING surrogate (mean log2(1 + |z_t - z_t-1|
/ 0.25) over all tokens, true previous codes = hold) and (b) the same form on the actually transmitted residuals
(top-k tokens, mc prediction from the receiver state). Within-clip correlations (all frames, event frames), per-clip
mean correlations across models, and per-clip model ranking agreement.

  extract  CLIPS VID_ROOT MODEL_DIR TOKENS_DIR B1_WORK OUT_DIR [--hevc 22,27,.. --av1 30,38,..]   (GPU, under the lock)
  eval     TOKENS_DIR CLIPS B1_WORK OUT_DIR SEM_EVAL_JSON OUT_JSON
  surrdiag TOKENS_DIR CLIPS B1_WORK B4_WORK OUT_JSON [--reps task,smooth,rateonly03 --seeds 0,1,2 --ks 64,128]
"""
import argparse
import json
import subprocess
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import b0_pipeline as B0
import b1_pipeline as B1

H_IMG, W_IMG, N_TOK, DIM = B0.H_IMG, B0.W_IMG, B0.N_TOK, B0.DIM
FPS = 10                                            # nominal: VID at ~30 fps, stride 3


def eval_clips(clips_json, tokens_dir):
    clips = json.loads(Path(clips_json).read_text())
    return [(c, r) for r in ("gate", "aux_val") for c in clips.get(r, []) if (Path(tokens_dir) / f"{c}.npz").exists()]


# ---------------------------------------------------------------- codecs
def enc_cmd(codec, q, out):
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W_IMG}x{H_IMG}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p"]
    if codec == "hevc":
        return base + ["-c:v", "libx265", "-preset", "slow", "-tune", "zerolatency", "-crf", str(q), "-x265-params",
                       "keyint=-1:min-keyint=1:scenecut=0:bframes=0:rc-lookahead=0:info=0:log-level=error",
                       "-f", "hevc", str(out)]
    if codec == "av1":
        return base + ["-c:v", "libsvtav1", "-preset", "6", "-crf", str(q), "-g", "100000",
                       "-svtav1-params", "pred-struct=1:lookahead=0:scd=0", "-f", "obu", str(out)]
    raise ValueError(codec)


def code_video(frames_rgb, codec, q, tmp):
    """frames_rgb (T, H, W, 3) uint8 -> (decoded (T, H, W, 3) uint8, total bits, per-packet bits, encode s)."""
    out = Path(tmp) / f"v_{codec}_{q}.{'hevc' if codec == 'hevc' else 'obu'}"
    t0 = time.time()
    subprocess.run(enc_cmd(codec, q, out), input=np.ascontiguousarray(frames_rgb).tobytes(), check=True)
    enc_s = time.time() - t0
    total = out.stat().st_size * 8
    pk = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "packet=size", "-of", "csv=p=0", str(out)],
                        capture_output=True, text=True, check=True).stdout.split()
    pk_bits = np.array([int(x) * 8 for x in pk])
    dec = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(out), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    T = len(frames_rgb)
    d = np.frombuffer(dec, np.uint8)
    assert d.size == T * H_IMG * W_IMG * 3, (d.size, T)
    out.unlink()
    return d.reshape(T, H_IMG, W_IMG, 3), int(total), pk_bits, enc_s


def extract(a):
    import torch
    from PIL import Image
    from transformers import Dinov2Model
    dev = "cuda"
    m = Dinov2Model.from_pretrained(a.model_dir, use_safetensors=True).eval().to(dev).half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    Hh = np.load(Path(a.b1_work) / "head.npz")
    W, b = Hh["W"], Hh["b"]

    def scores(frames):
        out = []
        for i in range(0, len(frames), 16):
            with torch.no_grad():
                xt = torch.from_numpy(np.stack(frames[i:i + 16])).to(dev).permute(0, 3, 1, 2).float().div(255)
                tk = m(pixel_values=((xt - mean) / std).half()).last_hidden_state[:, 1:, :].float().cpu().numpy()
            out.append(tk.astype(np.float16).astype(np.float32) @ W + b)    # same fp16 rounding as stored tokens
        return np.concatenate(out)

    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    pts = [("hevc", int(q)) for q in a.hevc.split(",") if q] + [("av1", int(q)) for q in a.av1.split(",") if q]
    stats, t_all = {}, time.time()
    for c, role in eval_clips(a.clips, a.tokens_dir):
        if (outd / f"{c}.npz").exists():
            continue
        D = np.load(Path(a.tokens_dir) / f"{c}.npz")
        split = "val" if role == "aux_val" else "train"
        root = Path(a.vid_root) / f"VisDrone2019-VID-{split}"
        files = sorted((root / "sequences" / c).glob("*.jpg"))[::3][:100]
        assert np.array_equal(np.array([int(f.stem) for f in files]), D["frames"]), c
        rgb = np.stack([np.asarray(Image.open(f).convert("RGB").resize((W_IMG, H_IMG), Image.BICUBIC)) for f in files])
        s_orig = scores(list(rgb))
        s_stored = D["tokens"].reshape(-1, DIM).astype(np.float32) @ W + b
        chk = float(np.abs(s_orig.reshape(-1, 4) - s_stored).max())
        res = dict(orig_scores=s_orig.astype(np.float32))
        st = dict(role=role, frames=len(files), orig_score_maxdiff=chk, points={})
        with tempfile.TemporaryDirectory() as tmp:
            for codec, q in pts:
                dec, tot, pk, es = code_video(rgb, codec, q, tmp)
                key = f"{codec}{q}"
                res[f"{key}_scores"] = scores(list(dec)).astype(np.float32)
                res[f"{key}_pkt_bits"] = pk
                res[f"{key}_total_bits"] = np.array(tot)
                psnr = float(np.mean([10 * np.log10(255 ** 2 / max(np.mean((dec[i].astype(np.float32) - rgb[i]) ** 2), 1e-9))
                                      for i in range(len(rgb))]))
                st["points"][key] = dict(bits_per_frame=tot / len(rgb), n_packets=int(len(pk)), pkt_sum=int(pk.sum()),
                                         first_pkt=int(pk[0]) if len(pk) else 0, psnr=psnr, enc_s=es)
        np.savez(outd / f"{c}.npz", **res)
        stats[c] = st
        print(c, role, len(files), f"maxdiff {chk:.3g}", {k: round(v["bits_per_frame"]) for k, v in st["points"].items()},
              f"{time.time() - t_all:.0f}s", flush=True)
    prev = json.loads((outd / "extract_stats.json").read_text()) if (outd / "extract_stats.json").exists() else {}
    prev.update(stats)
    (outd / "extract_stats.json").write_text(json.dumps(prev, indent=1))


# ---------------------------------------------------------------- evaluation
def metrics(sc, Y, M, T):
    import b2_pipeline as B2
    ch, ev = B2.change_masks(Y, M, T)
    sc = sc.reshape(-1, 4)
    return dict(ap=B0.task_ap(sc, Y, M), ap_change=B0.task_ap(sc, Y, M & ch), ap_event=B0.task_ap(sc, Y, M & ev))


def interp_log(bits, vals, target):
    o = np.argsort(bits)
    lb, v = np.log(np.asarray(bits, float)[o]), np.asarray(vals, float)[o]
    lt = np.log(target)
    if lt < lb[0] - 1e-9 or lt > lb[-1] + 1e-9:
        return np.nan
    return float(np.interp(lt, lb, v))


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return dict(mean=float(np.mean(x)) if len(x) else None, lo=None, hi=None, n=int(len(x)), pos=int((x > 0).sum()))
    from scipy import stats
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h), n=int(len(x)), pos=int((x > 0).sum()))


def evaluate(a):
    clips = eval_clips(a.clips, a.tokens_dir)
    est = json.loads((Path(a.out_dir) / "extract_stats.json").read_text())
    sem = json.loads(Path(a.sem_eval).read_text())["runs"]
    targets = [float(x) for x in a.targets.split(",")]
    curves = {}                       # (method, clip) -> list of (bits, metrics)
    ref = {}
    for c, _ in clips:
        D = np.load(Path(a.tokens_dir) / f"{c}.npz")
        V = np.load(Path(a.out_dir) / f"{c}.npz")
        T = len(D["frames"])
        Y, M = D["labels"].reshape(-1, 4), D["valid"].reshape(-1).astype(bool)
        ref[c] = metrics(V["orig_scores"], Y, M, T)
        for key, p in est[c]["points"].items():
            codec = "hevc" if key.startswith("hevc") else "av1"
            bpf = (p["bits_per_frame"] * T - p["first_pkt"]) / (T - 1) if a.steady else p["bits_per_frame"]
            curves.setdefault((codec, c), []).append((bpf, metrics(V[f"{key}_scores"], Y, M, T), key))
        Hh = np.load(Path(a.b1_work) / "head.npz")
        for q in (5, 10, 20, 35, 50):
            if f"avif{q}_tokens" in D:
                sc = D[f"avif{q}_tokens"].reshape(-1, DIM).astype(np.float32) @ Hh["W"] + Hh["b"]
                curves.setdefault(("avif_intra", c), []).append((float(D[f"avif{q}_bits"].mean()), metrics(sc, Y, M, T), f"avif{q}"))
    sem_ceiling = {}
    for r in sem:
        rep = r["rep"].rsplit("_s", 1)[0]
        if r["pred"] == "mc":
            T = est[r["clip"]]["frames"]
            bpf = (r["bits_mean_all"] * T - r["bits_init"]) / (T - 1) if a.steady else r["bits_mean_all"]
            curves.setdefault((r["rep"], r["clip"]), []).append((bpf, {k: r[k] for k in ("ap", "ap_change", "ap_event")}, r["k"]))
        elif r["pred"] == "none":
            sem_ceiling.setdefault(rep, []).append(r["ap"])
    reps = sorted({r["rep"].rsplit("_s", 1)[0] for r in sem})
    names = [c for c, _ in clips]
    out = dict(clips=names, targets=targets, steady=bool(a.steady),
               init_bits={"semantic_" + rp: float(np.mean([r["bits_init"] for r in sem if r["pred"] == "mc" and r["rep"].startswith(rp + "_s")]))
                          for rp in reps}, original_tokens={k: float(np.mean([ref[c][k] for c in names])) for k in ref[names[0]]},
               semantic_ceiling_ap={k: float(np.mean(v)) for k, v in sem_ceiling.items()},
               codec_points={codec: {key: dict(bits_per_frame=float(np.mean([est[c]["points"][key]["bits_per_frame"] for c in names])),
                                               psnr=float(np.mean([est[c]["points"][key]["psnr"] for c in names])),
                                               first_pkt_bits=float(np.mean([est[c]["points"][key]["first_pkt"] for c in names])),
                                               **{m: float(np.mean([next(x[1][m] for x in curves[(codec, c)] if x[2] == key) for c in names]))
                                                  for m in ("ap", "ap_change", "ap_event")})
                                     for key in est[names[0]]["points"] if key.startswith(codec)}
                            for codec in ("hevc", "av1")},
               matched={}, diffs={})
    methods = ["hevc", "av1", "avif_intra"] + reps
    for metric in ("ap", "ap_change", "ap_event"):
        for t in targets:
            tab = {}
            for meth in methods:
                per_clip = []
                for c in names:
                    if meth in reps:
                        vs = [interp_log([x[0] for x in curves[(f"{meth}_s{s}", c)]], [x[1][metric] for x in curves[(f"{meth}_s{s}", c)]], t)
                              for s in (0, 1, 2) if (f"{meth}_s{s}", c) in curves]
                        per_clip.append(float(np.mean(vs)) if vs and np.all(np.isfinite(vs)) else np.nan)
                    else:
                        cv = curves[(meth, c)]
                        per_clip.append(interp_log([x[0] for x in cv], [x[1][metric] for x in cv], t))
                tab[meth] = per_clip
            out["matched"][f"{metric}@{t:g}"] = {m: dict(per_clip=v, mean=float(np.nanmean(v)) if np.isfinite(v).any() else None,
                                                         n=int(np.isfinite(v).sum())) for m, v in tab.items()}
            for meth in reps:
                for base in ("hevc", "av1"):
                    out["diffs"][f"{metric}@{t:g}:{meth}-{base}"] = ci(np.array(tab[meth]) - np.array(tab[base]))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("original tokens", out["original_tokens"], "semantic ceiling", out["semantic_ceiling_ap"])
    for codec, pts in out["codec_points"].items():
        for key, p in pts.items():
            print(f"{key:8s} kbit/frame {p['bits_per_frame'] / 1e3:7.2f} first {p['first_pkt_bits'] / 1e3:7.1f} psnr {p['psnr']:5.2f} "
                  f"ap {p['ap']:.3f} ch {p['ap_change']:.3f} ev {p['ap_event']:.3f}")
    for metric in ("ap", "ap_change"):
        for t in targets:
            row = out["matched"][f"{metric}@{t:g}"]
            print(f"{metric}@{t / 1e3:g}k", " ".join(f"{m} {row[m]['mean']:.3f}({row[m]['n']})" if row[m]["mean"] is not None else f"{m} -"
                                                    for m in methods))


# ---------------------------------------------------------------- surrogate check
G = {}


def _sinit(b1_work, b4_work, tokens_dir):
    import torch
    torch.set_num_threads(1)
    G.update(b1=Path(b1_work), work=Path(b4_work), tok=Path(tokens_dir), models={})


def surr_job(j):
    import b2_pipeline as B2
    clip, rep_name, k = j
    if rep_name not in G["models"]:
        G["models"][rep_name] = B2.Rep(G["work"] / f"{rep_name}_d32.pt", 32)
    rep = G["models"][rep_name]
    D = np.load(G["tok"] / f"{clip}.npz")
    Z = rep.codes(D["tokens"].astype(np.float32))
    rx, bits = B1.transmit(Z, "mc", k, 0.25)
    T = len(Z)
    s_train = np.array([np.log2(1 + np.abs(Z[t] - Z[t - 1]) / 0.25).mean() for t in range(1, T)])
    s_cl = []
    for t in range(1, T):
        p = B1.shift_map(rx[t - 1], *B1.best_mv(rx[t - 1], Z[t]))
        err = ((Z[t] - p) ** 2).sum(1)
        idx = np.argpartition(-err, min(k, N_TOK) - 1)[:min(k, N_TOK)]
        s_cl.append(np.log2(1 + np.abs(Z[t][idx] - p[idx]) / 0.25).mean())
    Y, M = D["labels"].reshape(-1, 4), D["valid"].reshape(-1).astype(bool)
    import b2_pipeline as B2b
    _, ev = B2b.change_masks(Y, M, T)
    evf = ev.reshape(T, -1)[:, 0][1:]
    return dict(clip=clip, rep=rep_name, k=k, bits=bits[1:].astype(float).tolist(), s_train=s_train.tolist(),
                s_cl=np.asarray(s_cl).tolist(), event=evf.astype(int).tolist())


def surrdiag(a):
    from scipy import stats
    clips = [c for c, _ in eval_clips(a.clips, a.tokens_dir)]
    reps = [f"{r}_s{s}" for r in a.reps.split(",") for s in a.seeds.split(",")]
    ks = [int(x) for x in a.ks.split(",")]
    jobs = [(c, r, k) for c in clips for r in reps for k in ks]
    with Pool(a.procs, initializer=_sinit, initargs=(a.b1_work, a.b4_work, a.tokens_dir)) as p:
        res = p.map(surr_job, jobs, chunksize=1)
    out = dict(clips=clips, reps=reps, ks=ks, by_k={})
    for k in ks:
        rs = [r for r in res if r["k"] == k]
        w = {"train": [], "cl": [], "train_sp": [], "train_event": [], "cl_event": []}
        for r in rs:
            b, s1, s2, e = map(np.asarray, (r["bits"], r["s_train"], r["s_cl"], r["event"]))
            if b.std() > 0:
                w["train"].append(np.corrcoef(b, s1)[0, 1]); w["cl"].append(np.corrcoef(b, s2)[0, 1])
                w["train_sp"].append(stats.spearmanr(b, s1)[0])
                if e.sum() >= 5:
                    w["train_event"].append(np.corrcoef(b[e == 1], s1[e == 1])[0, 1])
                    w["cl_event"].append(np.corrcoef(b[e == 1], s2[e == 1])[0, 1])
        mb = np.array([np.mean(r["bits"]) for r in rs]); ms1 = np.array([np.mean(r["s_train"]) for r in rs])
        ms2 = np.array([np.mean(r["s_cl"]) for r in rs])
        rank = []
        for c in clips:
            cc = [r for r in rs if r["clip"] == c]
            rank.append(stats.spearmanr([np.mean(r["bits"]) for r in cc], [np.mean(r["s_train"]) for r in cc])[0])
        per_rep = {}
        for rp in a.reps.split(","):
            sel = [r for r in rs if r["rep"].rsplit("_s", 1)[0] == rp]
            per_rep[rp] = dict(bits_mean=float(np.mean([np.mean(r["bits"]) for r in sel])),
                               s_train_mean=float(np.mean([np.mean(r["s_train"]) for r in sel])))
        out["by_k"][str(k)] = dict(
            within_clip_pearson_train=dict(mean=float(np.mean(w["train"])), q=[float(x) for x in np.percentile(w["train"], [10, 50, 90])]),
            within_clip_spearman_train=float(np.mean(w["train_sp"])),
            within_clip_pearson_closedloop=float(np.mean(w["cl"])),
            event_frames_pearson_train=float(np.mean(w["train_event"])) if w["train_event"] else None,
            event_frames_pearson_closedloop=float(np.mean(w["cl_event"])) if w["cl_event"] else None,
            per_clip_mean_corr_train=float(np.corrcoef(mb, ms1)[0, 1]),
            per_clip_mean_corr_closedloop=float(np.corrcoef(mb, ms2)[0, 1]),
            model_rank_spearman_within_clip=dict(mean=float(np.nanmean(rank)), per_clip=[float(x) for x in rank]),
            per_rep=per_rep, n_series=len(rs))
        print(k, json.dumps(out["by_k"][str(k)], indent=None)[:900], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    for n in ("clips", "vid_root", "model_dir", "tokens_dir", "b1_work", "out_dir"):
        e.add_argument(n)
    e.add_argument("--hevc", default="22,27,32,37,42,47,51")
    e.add_argument("--av1", default="30,38,45,50,55,60,63")
    v = sub.add_parser("eval")
    for n in ("tokens_dir", "clips", "b1_work", "out_dir", "sem_eval", "out"):
        v.add_argument(n)
    v.add_argument("--targets", default="3000,4500,6000,9000,12000,18000,25000,35000")
    v.add_argument("--steady", action="store_true", help="rates exclude the first (intra / init) frame: (total - first) / (T - 1)")
    s = sub.add_parser("surrdiag")
    for n in ("tokens_dir", "clips", "b1_work", "b4_work", "out"):
        s.add_argument(n)
    s.add_argument("--reps", default="task,smooth,rateonly03")
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--ks", default="64,128")
    s.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    dict(extract=extract, eval=evaluate, surrdiag=surrdiag)[a.cmd](a)


if __name__ == "__main__":
    main()
