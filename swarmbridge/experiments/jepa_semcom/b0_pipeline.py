"""B-E0 (reports/two-track/swarmbridge-two-track-plan.md, review #27): minimal predictive token-transmission pipeline.

  video frame -> frozen DINOv2-S patch tokens (22 x 39 grid at 308 x 546) -> receiver-synchronised predictor
  (B-E0: hold-last) -> top-k innovation patches -> per-token quantisation -> real bitstream -> decode ->
  receiver token map -> patch-level task head.

Synchronisation: the sender keeps an exact replica of the receiver token map and updates it with the DECODED tokens
(never the raw ones), so both ends hold bit-identical state after every message (asserted). Channel lossless in B-E0.
Message of frame t: header (frame 16 bit, k 11 bit, q 4 bit, index-mode 1 bit) + patch indices (bitmap of 858 bits
or delta exp-Golomb, whichever is shorter) + per token an fp16 scale and 384 q-bit signed integers. Frame 0 of a clip
is the initial message (all patches). Bits are counted from the packed byte string.
Task: per-patch presence of 4 classes (person {1,2}, car {4}, large vehicle {5,6,9}, two/three-wheeler {3,7,8,10})
from VisDrone-VID boxes (patch whose centre lies in a box, or that contains the box centre); patches whose centre
lies in an 'ignored region' (category 0) are excluded. Linear head trained on TUNE clips (split v1, train sources),
evaluated on held-out clips (split v1 'gate' = held-out train sources, 'aux_val' = VID-val clips with no DET-train
match). No DET-trained detector is used.

  extract  CLIPS_JSON VID_ROOT MODEL_DIR OUT_DIR [--stride 3 --max-frames 150 --batch 16]     (GPU, under the lock)
  run      TOKENS_DIR CLIPS_JSON OUT_JSON                                                   (CPU)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

H_IMG, W_IMG, PATCH = 308, 546, 14
GH, GW, DIM = H_IMG // PATCH, W_IMG // PATCH, 384
N_TOK = GH * GW
CLASSES = {0: (1, 2), 1: (4,), 2: (5, 6, 9), 3: (3, 7, 8, 10)}
K_IDX = 2                                       # exp-Golomb order for index deltas


# ---------------------------------------------------------------- bit packing (numpy)
class Bits:
    """append-only bit buffer: fixed-width fields vectorised, exp-Golomb for small integer lists."""

    def __init__(self):
        self.parts = []

    def u(self, vals, n):
        v = np.atleast_1d(np.asarray(vals, dtype=np.int64))
        assert (v >= 0).all() and (v < (1 << n)).all()
        self.parts.append(((v[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.uint8).ravel())

    def ue(self, v, k):
        x = v + (1 << k)
        n = x.bit_length() - 1
        self.parts.append(np.zeros(n - k, np.uint8))
        self.u(x, n + 1)

    def raw(self, bits):
        self.parts.append(np.asarray(bits, np.uint8))

    def tobytes(self):
        b = np.concatenate(self.parts) if self.parts else np.zeros(0, np.uint8)
        return np.packbits(b).tobytes(), int(b.size)


class Reader:
    def __init__(self, data, nbits):
        self.b = np.unpackbits(np.frombuffer(data, np.uint8))[:nbits]
        self.i = 0

    def u(self, n, count=1):
        seg = self.b[self.i:self.i + n * count].reshape(count, n).astype(np.int64)
        self.i += n * count
        v = (seg << np.arange(n - 1, -1, -1)).sum(1)
        return v if count > 1 else int(v[0])

    def ue(self, k):
        z = 0
        while self.b[self.i] == 0:
            z += 1
            self.i += 1
        return self.u(z + k + 1) - (1 << k)

    def raw(self, n):
        seg = self.b[self.i:self.i + n]
        self.i += n
        return seg


def eg_len(v, k):
    return 2 * int(math.floor(math.log2(v + (1 << k)))) - k + 1


# ---------------------------------------------------------------- token quantisation and messages
def quantise(tok, q):
    """per-token symmetric uniform quantiser: fp16 scale + q-bit signed ints in [-(2^(q-1)-1), 2^(q-1)-1]."""
    L = (1 << (q - 1)) - 1
    scale = (np.abs(tok).max(1) / L).astype(np.float16)
    s = np.maximum(scale.astype(np.float32), 1e-8)[:, None]
    qi = np.clip(np.rint(tok / s), -L, L).astype(np.int64)
    return scale, qi


def dequantise(scale, qi):
    return (qi * scale.astype(np.float32)[:, None]).astype(np.float32)


def encode_msg(frame, idx, tok, q):
    """idx: sorted patch indices; tok: (len(idx), DIM) float. Returns (bytes, nbits, decoded tokens)."""
    idx = np.asarray(sorted(idx), np.int64)
    w = Bits()
    w.u(frame % 65536, 16); w.u(len(idx), 11); w.u(q, 4)
    d = np.diff(np.r_[-1, idx]) - 1
    eg = sum(eg_len(int(x), K_IDX) for x in d)
    use_bitmap = eg >= N_TOK
    w.u(int(use_bitmap), 1)
    if use_bitmap:
        bm = np.zeros(N_TOK, np.uint8); bm[idx] = 1
        w.raw(bm)
    else:
        for x in d:
            w.ue(int(x), K_IDX)
    if len(idx):
        scale, qi = quantise(tok, q)
        L = (1 << (q - 1)) - 1
        w.raw(np.unpackbits(scale.astype('>f2').view(np.uint8)))        # 16 bits per token, big-endian fp16
        w.u((qi + L).ravel(), q)
        dec = dequantise(scale, qi)
    else:
        dec = np.zeros((0, DIM), np.float32)
    data, nbits = w.tobytes()
    return data, nbits, idx, dec


def decode_msg(data, nbits):
    r = Reader(data, nbits)
    frame, k, q, use_bitmap = r.u(16), r.u(11), r.u(4), r.u(1)
    if use_bitmap:
        idx = np.flatnonzero(r.raw(N_TOK)).astype(np.int64)
    else:
        idx, prev = [], -1
        for _ in range(k):
            prev = prev + 1 + r.ue(K_IDX)
            idx.append(prev)
        idx = np.asarray(idx, np.int64)
    assert len(idx) == k
    if k:
        scale = np.packbits(r.raw(16 * k)).view('>f2').astype(np.float16)
        L = (1 << (q - 1)) - 1
        qi = np.asarray(r.u(q, k * DIM)).reshape(k, DIM) - L
        dec = dequantise(scale, qi)
    else:
        dec = np.zeros((0, DIM), np.float32)
    assert r.i == nbits
    return frame, idx, dec


# ---------------------------------------------------------------- predictive transmission (B-E0: hold-last)
def transmit_clip(tokens, k, q, predictor="hold"):
    """tokens: (T, N_TOK, DIM). Returns receiver token maps (T, N_TOK, DIM) and bits per frame.
    The sender replica is updated only from the DECODED message, exactly like the receiver."""
    assert predictor == "hold"
    T = len(tokens)
    rx = np.zeros((N_TOK, DIM), np.float32)            # receiver state
    rep = np.zeros((N_TOK, DIM), np.float32)           # sender's replica of it
    out, bits = np.zeros((T, N_TOK, DIM), np.float16), np.zeros(T, np.int64)
    for t in range(T):
        x = tokens[t].astype(np.float32)
        pred = rep                                      # hold-last prediction of the receiver's map
        if t == 0:
            sel = np.arange(N_TOK)
        else:
            err = np.linalg.norm(x - pred, axis=1)
            kk = min(k, N_TOK)
            sel = np.sort(np.argpartition(-err, kk - 1)[:kk]) if kk > 0 else np.zeros(0, np.int64)
        data, nb, idx, dec_tx = encode_msg(t, sel, x[sel], q)
        rep[idx] = dec_tx
        f, idx_rx, dec_rx = decode_msg(data, nb)                # receiver side
        assert f == t % 65536 and np.array_equal(idx_rx, idx) and np.array_equal(dec_rx, dec_tx)
        rx[idx_rx] = dec_rx
        assert np.array_equal(rx, rep), "sender replica and receiver state diverged"
        out[t], bits[t] = rx, nb
    return out, bits


# ---------------------------------------------------------------- labels
def patch_labels(ann_rows, frame_hw):
    """ann_rows: (n, 10) VisDrone-VID rows of one frame. -> labels (N_TOK, 4) uint8, valid (N_TOK,) bool."""
    h0, w0 = frame_hw
    sx, sy = W_IMG / w0, H_IMG / h0
    cy, cx = (np.arange(GH) + 0.5) * PATCH, (np.arange(GW) + 0.5) * PATCH
    CY, CX = np.meshgrid(cy, cx, indexing="ij")
    lab = np.zeros((GH, GW, 4), np.uint8)
    valid = np.ones((GH, GW), bool)
    for r in ann_rows:
        x, y, w, h, cat = r[2] * sx, r[3] * sy, r[4] * sx, r[5] * sy, int(r[7])
        inside = (CX >= x) & (CX < x + w) & (CY >= y) & (CY < y + h)
        if cat == 0:
            valid &= ~inside
            continue
        c = next((c for c, cats in CLASSES.items() if cat in cats), None)
        if c is None:
            continue
        lab[..., c] |= inside
        gy, gx = min(int((y + h / 2) // PATCH), GH - 1), min(int((x + w / 2) // PATCH), GW - 1)
        lab[gy, gx, c] = 1
    return lab.reshape(N_TOK, 4), valid.reshape(N_TOK)


# ---------------------------------------------------------------- extraction (GPU)
def extract(a):
    import torch
    from PIL import Image
    from transformers import Dinov2Model
    clips = json.loads(Path(a.clips).read_text())
    dev = "cuda"
    m = Dinov2Model.from_pretrained(a.model_dir, use_safetensors=True).eval().to(dev).half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    stats = {}
    torch.cuda.reset_peak_memory_stats()
    for role, names in clips.items():
        if role.startswith("_"):
            continue
        for c in names:
            split = "val" if role == "aux_val" else "train"
            root = Path(a.vid_root) / f"VisDrone2019-VID-{split}"
            frames = sorted((root / "sequences" / c).glob("*.jpg"))[::a.stride][:a.max_frames]
            ann = np.loadtxt(root / "annotations" / f"{c}.txt", delimiter=",", ndmin=2)
            toks, labs, vals = [], [], []
            t_load = t_gpu = 0.0
            for i in range(0, len(frames), a.batch):
                t0 = time.time()
                ims = [Image.open(f).convert("RGB") for f in frames[i:i + a.batch]]
                hw = [(im.height, im.width) for im in ims]
                x = np.stack([np.asarray(im.resize((W_IMG, H_IMG), Image.BICUBIC)) for im in ims])
                t1 = time.time()
                with torch.no_grad():
                    xt = torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float().div(255)
                    xt = ((xt - mean) / std).half()
                    hs = m(pixel_values=xt).last_hidden_state[:, 1:, :]
                    toks.append(hs.float().cpu().numpy().astype(np.float16))
                torch.cuda.synchronize()
                t2 = time.time()
                t_load += t1 - t0; t_gpu += t2 - t1
                for j, f in enumerate(frames[i:i + a.batch]):
                    fi = int(f.stem)
                    lb, vd = patch_labels(ann[ann[:, 0] == fi], hw[j])
                    labs.append(lb); vals.append(vd)
            T = np.concatenate(toks)
            assert T.shape[1:] == (N_TOK, DIM)
            np.savez(out / f"{c}.npz", tokens=T, labels=np.stack(labs), valid=np.stack(vals),
                     frames=np.array([int(f.stem) for f in frames]))
            stats[c] = dict(role=role, frames=len(frames), load_s=t_load, gpu_s=t_gpu,
                            gpu_fps=len(frames) / max(t_gpu, 1e-9))
            print(c, stats[c], flush=True)
    stats["_peak_gpu_mem_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    stats["_cfg"] = dict(stride=a.stride, max_frames=a.max_frames, batch=a.batch, res=[H_IMG, W_IMG], fp16=True,
                         device=torch.cuda.get_device_name(0))
    (out / "extract_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k.startswith("_")}), flush=True)


# ---------------------------------------------------------------- run (CPU)
def train_head(X, Y, M, epochs=30, seed=0):
    import torch
    torch.manual_seed(seed)
    X, Y = torch.from_numpy(X[M]).float(), torch.from_numpy(Y[M]).float()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lin = torch.nn.Linear(DIM, 4)
    pos = Y.mean(0).clamp(1e-4, 1)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).clamp(max=50))
    opt = torch.optim.Adam(lin.parameters(), 1e-3, weight_decay=1e-4)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 4096):
            b = perm[i:i + 4096]
            opt.zero_grad(); lossf(lin((X[b] - mu) / sd), Y[b]).backward(); opt.step()
    W = lin.weight.detach().numpy(); b = lin.bias.detach().numpy()
    mu, sd = mu.numpy(), sd.numpy()
    return lambda Z: ((Z.astype(np.float32) - mu) / sd) @ W.T + b


def task_ap(score, Y, M):
    from sklearn.metrics import average_precision_score
    s, y = score[M], Y[M]
    aps = [average_precision_score(y[:, c], s[:, c]) for c in range(4) if y[:, c].any()]
    return float(np.mean(aps)) if aps else None


def run(a):
    clips = json.loads(Path(a.clips).read_text())
    D = {p.stem: np.load(p) for p in Path(a.tokens_dir).glob("*.npz")}
    tr = [c for c in clips["tune"] if c in D]
    Xtr = np.concatenate([D[c]["tokens"].reshape(-1, DIM) for c in tr])
    Ytr = np.concatenate([D[c]["labels"].reshape(-1, 4) for c in tr])
    Mtr = np.concatenate([D[c]["valid"].reshape(-1) for c in tr])
    head = train_head(Xtr, Ytr, Mtr)
    ev = [c for r in ("gate", "aux_val") for c in clips.get(r, []) if c in D]
    rows = []
    grid = [(0, 4), (16, 4), (64, 4), (128, 4), (256, 4), (64, 2), (64, 8), (N_TOK, 4)]
    for c in ev:
        T, Y, M = D[c]["tokens"], D[c]["labels"].reshape(-1, 4), D[c]["valid"].reshape(-1)
        base = task_ap(head(T.reshape(-1, DIM)), Y, M)
        row = dict(clip=c, frames=len(T), ap_original_tokens=base, runs=[])
        for k, q in grid:
            t0 = time.time()
            rx, bits = transmit_clip(T, k, q)
            el = time.time() - t0
            Z = rx.reshape(-1, DIM)
            cos = float(np.mean(np.sum(rx.astype(np.float32) * T.astype(np.float32), -1) /
                                (np.linalg.norm(rx.astype(np.float32), axis=-1) *
                                 np.linalg.norm(T.astype(np.float32), axis=-1) + 1e-8)))
            r = dict(k=k, q=q, init_bits=int(bits[0]), mean_bits_after_init=float(bits[1:].mean()) if len(bits) > 1 else None,
                     mean_bits_all=float(bits.mean()), token_cos=cos, ap=task_ap(head(Z), Y, M), sim_s=el)
            row["runs"].append(r)
            print(c, r, flush=True)
        rows.append(row)
    summ = {}
    for k, q in grid:
        rs = [next(x for x in row["runs"] if x["k"] == k and x["q"] == q) for row in rows]
        summ[f"k{k}_q{q}"] = dict(mean_bits_after_init=float(np.mean([x["mean_bits_after_init"] for x in rs])),
                                  init_bits=float(np.mean([x["init_bits"] for x in rs])),
                                  ap=float(np.mean([x["ap"] for x in rs])), token_cos=float(np.mean([x["token_cos"] for x in rs])))
    summ["original_tokens_ap"] = float(np.mean([row["ap_original_tokens"] for row in rows]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(train_clips=tr, eval_clips=ev, grid=grid, summary=summ, per_clip=rows),
                                      indent=1))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    for n in ("clips", "vid_root", "model_dir", "out_dir"):
        e.add_argument(n)
    e.add_argument("--stride", type=int, default=3)
    e.add_argument("--max-frames", type=int, default=150)
    e.add_argument("--batch", type=int, default=16)
    r = sub.add_parser("run")
    for n in ("tokens_dir", "clips", "out"):
        r.add_argument(n)
    a = ap.parse_args()
    (extract if a.cmd == "extract" else run)(a)


if __name__ == "__main__":
    main()
