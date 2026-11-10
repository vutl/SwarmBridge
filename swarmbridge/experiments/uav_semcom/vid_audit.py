"""VisDrone2019-VID train / val data-integrity audit (no training, no pilot).

  1. structure: clip ids, frames per clip, resolution, frame-index continuity vs annotation;
  2. annotation schema (VID toolkit: frame_index, target_id, left, top, width, height, score, category, truncation,
     occlusion): value ranges, ignored regions, per-class counts;
  3. track statistics: tracks per clip, track length (frames), fragmentation (gaps inside a track), objects per frame,
     small (< 32^2 px) fraction;
  4. DET leakage: (a) metadata - numeric source ids in DET names (0000013_..._d_...) vs VID clip ids (uav0000013_...);
     (b) 64-bit DCT perceptual hash of EVERY VID frame vs every DET-train / DET-val / DET-test-dev image, Hamming
     distance; pairs <= --thr are suspects, confirmed with a 64x64 grey normalised correlation; side-by-side JPEGs of
     the suspects are written for manual inspection (outside git).
Usage: python -I vid_audit.py VID_ROOT OUT_DIR SIDE_DIR --det NAME=DIR ... [--procs 6] [--thr 10]
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

N = 32
_C = np.array([[np.sqrt((1 if k == 0 else 2) / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N)) for n in range(N)]
               for k in range(N)])
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def phash(path):
    """(64-bit pHash as 8 uint8, 64x64 grey thumbnail, (w, h))."""
    im = Image.open(path)
    w, h = im.size
    g = im.convert("L")
    a = np.asarray(g.resize((N, N), Image.BILINEAR), np.float64)
    D = _C @ a @ _C.T
    low = D[:8, :8].flatten()
    bits = low > np.median(low[1:])
    th = np.asarray(g.resize((64, 64), Image.BILINEAR), np.uint8)
    return np.packbits(bits), th, (w, h)


def hash_job(p):
    try:
        h, th, wh = phash(p)
        return str(p), h, th, wh
    except Exception as e:      # noqa: BLE001
        return str(p), None, None, repr(e)


def read_ann(path):
    A = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2) if path.stat().st_size else np.zeros((0, 10), np.int64)
    return A


def ncc(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vid_root")
    ap.add_argument("out")
    ap.add_argument("side")
    ap.add_argument("--det", action="append", default=[])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--thr", type=int, default=10)
    a = ap.parse_args()
    root, out, side = Path(a.vid_root), Path(a.out), Path(a.side)
    out.mkdir(parents=True, exist_ok=True); side.mkdir(parents=True, exist_ok=True)
    rep = dict(splits={}, leakage={})
    vid_frames = []
    for split in ("train", "val"):
        sd = root / f"VisDrone2019-VID-{split}"
        if not sd.exists():
            print("missing", sd); continue
        clips = sorted(p.name for p in (sd / "sequences").iterdir() if p.is_dir())
        S = dict(n_clips=len(clips), clips={}, schema_errors=[], category_counts=Counter())
        for c in clips:
            frames = sorted((sd / "sequences" / c).glob("*.jpg"))
            vid_frames += [(split, c, f) for f in frames]
            idx = [int(f.stem) for f in frames]
            A = read_ann(sd / "annotations" / f"{c}.txt")
            err = []
            if A.shape[1] != 10:
                err.append(f"{c}: {A.shape[1]} columns")
            else:
                fi, tid, l, t, w, h, sc, cat, tr, oc = A.T
                if fi.min() < 1 or fi.max() > max(idx):
                    err.append(f"{c}: frame_index range {fi.min()}..{fi.max()} vs frames 1..{max(idx)}")
                if not set(np.unique(sc)) <= {0, 1}: err.append(f"{c}: score values {sorted(set(np.unique(sc)))}")
                if not set(np.unique(cat)) <= set(range(12)): err.append(f"{c}: category {sorted(set(np.unique(cat)))}")
                if not set(np.unique(tr)) <= {0, 1, 2}: err.append(f"{c}: truncation {sorted(set(np.unique(tr)))}")
                if not set(np.unique(oc)) <= {0, 1, 2}: err.append(f"{c}: occlusion {sorted(set(np.unique(oc)))}")
                if (w <= 0).any() or (h <= 0).any(): err.append(f"{c}: {int(((w <= 0) | (h <= 0)).sum())} non-positive boxes")
                S["category_counts"].update(Counter(int(x) for x in cat))
                obj = (cat > 0) & (cat < 11) & (sc == 1)      # evaluated object classes (others / ignored excluded)
                tracks = defaultdict(list)
                for f_, i_ in zip(fi[obj], tid[obj]):
                    tracks[int(i_)].append(int(f_))
                lens = np.array([len(v) for v in tracks.values()]) if tracks else np.zeros(0)
                gaps = np.array([max(v) - min(v) + 1 - len(set(v)) for v in tracks.values()]) if tracks else np.zeros(0)
                dup = sum(len(v) - len(set(v)) for v in tracks.values())
                if dup: err.append(f"{c}: {dup} duplicate (frame, target_id) rows")
                per_frame = np.bincount(fi[obj], minlength=max(idx) + 1)[1:]
                small = (w[obj] * h[obj] < 32 ** 2).mean() if obj.any() else float("nan")
                ign = int((cat == 0).sum())
            wh = Image.open(frames[0]).size if frames else None
            S["clips"][c] = dict(frames=len(frames), first=min(idx) if idx else None, last=max(idx) if idx else None,
                                 contiguous=bool(idx == list(range(min(idx), max(idx) + 1))) if idx else None,
                                 wh=wh, ann_rows=int(len(A)), tracks=int(len(lens)),
                                 track_len_median=float(np.median(lens)) if len(lens) else None,
                                 track_len_p10=float(np.percentile(lens, 10)) if len(lens) else None,
                                 track_len_max=int(lens.max()) if len(lens) else None,
                                 tracks_with_gaps=int((gaps > 0).sum()), gap_frames=int(gaps.sum()),
                                 obj_per_frame_mean=float(per_frame.mean()), obj_per_frame_max=int(per_frame.max()),
                                 frames_without_objects=int((per_frame == 0).sum()), small_frac=float(small),
                                 ignored_regions=ign)
            S["schema_errors"] += err
        S["category_counts"] = dict(sorted(S["category_counts"].items()))
        S["frames"] = sum(v["frames"] for v in S["clips"].values())
        S["resolutions"] = dict(Counter(str(v["wh"]) for v in S["clips"].values()))
        rep["splits"][split] = S
        print(split, S["n_clips"], "clips", S["frames"], "frames", "schema errors:", len(S["schema_errors"]), flush=True)

    # ---- leakage (a): metadata ids
    vid_ids = {int(re.match(r"uav(\d+)_", c).group(1)) for s in rep["splits"].values() for c in s["clips"]}
    det_dirs = dict(x.split("=", 1) for x in a.det)
    det_imgs = {}
    for nm, dd in det_dirs.items():
        det_imgs[nm] = sorted(Path(dd).glob("*.jpg"))
        ids = Counter(int(p.name.split("_")[0]) for p in det_imgs[nm])
        rep["leakage"][f"id_overlap_{nm}"] = dict(n_det_ids=len(ids), n_vid_ids=len(vid_ids),
                                                  overlap=sorted(set(ids) & vid_ids),
                                                  det_images_with_overlapping_id=int(sum(ids[i] for i in set(ids) & vid_ids)))
        print("id overlap", nm, rep["leakage"][f"id_overlap_{nm}"]["overlap"][:30], flush=True)

    # ---- leakage (b): perceptual hash
    paths = [str(f) for _, _, f in vid_frames] + [str(p) for v in det_imgs.values() for p in v]
    H, TH, WH = {}, {}, {}
    with Pool(a.procs) as pool:
        for k, (p, h, th, wh) in enumerate(pool.imap_unordered(hash_job, paths, chunksize=32)):
            if h is None:
                print("hash fail", p, wh); continue
            H[p], TH[p], WH[p] = h, th, wh
            if k % 5000 == 0:
                print(f"hashed {k}/{len(paths)}", flush=True)
    for nm, imgs in det_imgs.items():
        dp = [str(p) for p in imgs if str(p) in H]
        DH = np.stack([H[p] for p in dp])
        best = []
        hist = Counter()
        sus = []
        for split, c, f in vid_frames:
            f = str(f)
            if f not in H:
                continue
            dist = POP[np.bitwise_xor(DH, H[f][None, :])].sum(1)
            j = int(np.argmin(dist))
            hist[int(dist[j])] += 1
            if dist[j] <= a.thr:
                sus.append((split, c, f, dp[j], int(dist[j]), ncc(TH[f], TH[dp[j]])))
        per_clip = Counter((s[0], s[1]) for s in sus)
        conf = [s for s in sus if s[5] >= 0.9]
        rep["leakage"][f"phash_{nm}"] = dict(
            n_det=len(dp), thr=a.thr, min_dist_hist={str(k): v for k, v in sorted(hist.items())},
            n_suspect_frames=len(sus), n_confirmed_ncc90=len(conf),
            suspects_per_clip={f"{k[0]}/{k[1]}": v for k, v in per_clip.most_common()},
            confirmed_per_clip=dict(Counter(f"{s[0]}/{s[1]}" for s in conf).most_common()),
            det_images_matched=len({s[3] for s in conf}),
            examples=[dict(split=s[0], clip=s[1], vid=Path(s[2]).name, det=Path(s[3]).name, hamming=s[4], ncc=round(s[5], 3))
                      for s in sorted(sus, key=lambda s: (s[4], -s[5]))[:60]])
        # side-by-side for manual inspection: best few per clip
        seen = Counter()
        for s in sorted(sus, key=lambda s: (s[4], -s[5])):
            key = (s[0], s[1])
            if seen[key] >= 3:
                continue
            seen[key] += 1
            A_, B_ = Image.open(s[2]).convert("RGB"), Image.open(s[3]).convert("RGB")
            A_.thumbnail((640, 640)); B_.thumbnail((640, 640))
            cv = Image.new("RGB", (A_.width + B_.width + 10, max(A_.height, B_.height)), "white")
            cv.paste(A_, (0, 0)); cv.paste(B_, (A_.width + 10, 0))
            cv.save(side / f"{nm}_{s[1]}_{Path(s[2]).stem}__{Path(s[3]).stem}_h{s[4]}_n{s[5]:.2f}.jpg", quality=80)
        print(nm, "suspects", len(sus), "confirmed", len(conf), dict(list(per_clip.items())[:10]), flush=True)
    # control: within-VID near-duplicates across splits (train vs val clip) at the same threshold
    tr = [str(f) for s, _, f in vid_frames if s == "train" and str(f) in H][::5]
    va = [(c, str(f)) for s, c, f in vid_frames if s == "val" and str(f) in H]
    if tr and va:
        TH_ = np.stack([H[p] for p in tr])
        cross = Counter()
        for c, f in va:
            if POP[np.bitwise_xor(TH_, H[f][None, :])].sum(1).min() <= a.thr:
                cross[c] += 1
        rep["leakage"]["vid_val_vs_vid_train_every5th"] = dict(cross)
    (out / "vid_audit.json").write_text(json.dumps(rep, indent=1, default=str))
    print("VID_AUDIT_DONE")


if __name__ == "__main__":
    sys.exit(main())
