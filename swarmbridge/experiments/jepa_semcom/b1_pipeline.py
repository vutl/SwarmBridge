"""B-E1 (review #28): compressed latent space + causal predictors in closed loop + AVIF -> DINOv2 baseline.

Representation sent over the channel: z = PCA_d(DINOv2-S patch token) per patch (d = 32 / 64; PCA fitted on TUNE
clips only), normalised per dimension by the train std. Receiver state = a d x 22 x 39 map.
Each frame t: receiver and sender (exact replica) form the same prediction p_t from the RECEIVER's reconstructed
history; the sender picks the k patches with the largest ||z_t - p_t||, sends quantised residuals
q = round((z - p) / step) as signed exp-Golomb codes (order chosen per message, in the header), plus the patch indices
(bitmap or exp-Golomb, shorter) and any side information (motion vector). The receiver sets its map to p_t
everywhere and adds the residuals on the k patches. Frame 0: prediction = 0 (the train mean), all patches sent.
Predictors (all closed loop, from receiver state only):
  hold      p_t = r_{t-1}
  mc        p_t = r_{t-1} shifted by an integer patch motion vector (dx, dy) in [-4, 4]^2 chosen by the SENDER
            (min error against z_t) and transmitted (8 bits / frame, counted); border patches keep r_{t-1}
  learned   p_t = r_{t-1} + f_theta(r_{t-1}, r_{t-2}) (small conv net, MSE in z space; trained on TUNE clips with
            receiver-like corrupted inputs: quantisation noise and stale patches); no side information
Task: the linear 4-class patch head of B-E0, trained on ORIGINAL 384-d TUNE tokens; every method is scored by
mapping its receiver state back to 384-d (PCA inverse) and applying that head. AVIF baseline: each frame resized to
308 x 546, AVIF-encoded (speed 6, 4:2:0), decoded at the receiver, DINOv2-S tokens, same head; bits = AVIF bytes.

  extract  CLIPS VID_ROOT MODEL_DIR OUT_DIR [--stride 3 --max-frames 100 --avif 5,10,20,35,50]   (GPU, lock)
  fit      TOKENS_DIR CLIPS WORK_DIR                                (PCA + head + projected train maps; CPU)
  train    WORK_DIR [--d 64]                                        (learned predictor; GPU, lock)
  eval     TOKENS_DIR CLIPS WORK_DIR OUT_JSON [--procs 6]           (closed loop; CPU)
"""
import argparse
import io
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import b0_pipeline as B0

GH, GW, DIM, N_TOK = B0.GH, B0.GW, B0.DIM, B0.N_TOK
MV_R = 4


# ---------------------------------------------------------------- vectorised signed exp-Golomb
def eg_encode(vals, k):
    """signed ints -> bit array (zigzag, order-k exp-Golomb)."""
    v = np.asarray(vals, np.int64).ravel()
    u = np.where(v >= 0, 2 * v, -2 * v - 1)
    x = u + (1 << k)
    n = np.floor(np.log2(x)).astype(np.int64)
    L = 2 * n - k + 1
    W = int(L.max()) if len(L) else 1
    bits = ((x[:, None] >> np.arange(W - 1, -1, -1)) & 1).astype(np.uint8)
    mask = np.arange(W)[None, :] >= (W - L)[:, None]
    return bits[mask]


def eg_len(vals, k):
    v = np.asarray(vals, np.int64).ravel()
    u = np.where(v >= 0, 2 * v, -2 * v - 1)
    return int((2 * np.floor(np.log2(u + (1 << k))) - k + 1).sum())


def eg_decode(bits, start, count, k):
    out = np.empty(count, np.int64)
    ones = np.flatnonzero(bits[start:]) + start
    oi, i = 0, start
    for j in range(count):
        while ones[oi] < i:
            oi += 1
        p = ones[oi]
        z = p - i
        nb = z + k + 1
        seg = bits[p:p + nb]
        x = int(seg.dot(1 << np.arange(nb - 1, -1, -1, dtype=np.int64)))
        u = x - (1 << k)
        out[j] = u // 2 if u % 2 == 0 else -(u + 1) // 2
        i = p + nb
    return out, i


def best_k(vals):
    return min(range(12), key=lambda k: eg_len(vals, k))


# ---------------------------------------------------------------- messages
def encode_msg(frame, idx, resid_q, mv=None):
    """idx sorted (k,), resid_q (k, d) ints, mv (dx, dy) or None -> (bytes, nbits)."""
    w = B0.Bits()
    k = len(idx)
    d = resid_q.shape[1] if np.ndim(resid_q) == 2 else 0
    w.u(frame % 65536, 16); w.u(k, 11); w.u(d, 7)
    w.u(int(mv is not None), 1)
    if mv is not None:
        w.u(mv[0] + MV_R, 4); w.u(mv[1] + MV_R, 4)
    dl = np.diff(np.r_[-1, idx]) - 1
    use_bm = eg_len(dl, B0.K_IDX) >= N_TOK if k else False
    w.u(int(use_bm), 1)
    if k:
        if use_bm:
            bm = np.zeros(N_TOK, np.uint8); bm[idx] = 1
            w.raw(bm)
        else:
            w.raw(eg_encode(dl, B0.K_IDX))
        kk = best_k(resid_q)
        w.u(kk, 4)
        w.raw(eg_encode(resid_q, kk))
    return w.tobytes()


def decode_msg(data, nbits):
    r = B0.Reader(data, nbits)
    frame, k, d, has_mv = r.u(16), r.u(11), r.u(7), r.u(1)
    mv = (r.u(4) - MV_R, r.u(4) - MV_R) if has_mv else None
    use_bm = r.u(1)
    if not k:
        assert r.i == nbits
        return frame, np.zeros(0, np.int64), np.zeros((0, d), np.int64), mv
    if use_bm:
        idx = np.flatnonzero(r.raw(N_TOK)).astype(np.int64)
    else:
        dl, r.i = eg_decode(r.b, r.i, k, B0.K_IDX)
        idx = np.cumsum(dl + 1) - 1
    kk = r.u(4)
    vals, r.i = eg_decode(r.b, r.i, k * d, kk)
    assert r.i == nbits
    return frame, idx, vals.reshape(k, d), mv


# ---------------------------------------------------------------- predictors
def shift_map(m, dx, dy):
    """m (N_TOK, d) -> shifted by (dx, dy) patches; uncovered border patches keep their old value."""
    g = m.reshape(GH, GW, -1)
    out = g.copy()
    ys, yd = (slice(0, GH - dy), slice(dy, GH)) if dy >= 0 else (slice(-dy, GH), slice(0, GH + dy))
    xs, xd = (slice(0, GW - dx), slice(dx, GW)) if dx >= 0 else (slice(-dx, GW), slice(0, GW + dx))
    out[yd, xd] = g[ys, xs]
    return out.reshape(N_TOK, -1)


def best_mv(prev, target):
    best, be = (0, 0), math.inf
    for dy in range(-MV_R, MV_R + 1):
        for dx in range(-MV_R, MV_R + 1):
            e = float(((shift_map(prev, dx, dy) - target) ** 2).sum())
            if e < be:
                best, be = (dx, dy), e
    return best


class LearnedPred:
    def __init__(self, path, d):
        import torch
        self.torch = torch
        self.net = make_net(d)
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()

    def __call__(self, r1, r2):
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(np.concatenate([r1, r2], 1).T.reshape(1, -1, GH, GW).astype(np.float32))
            return r1 + self.net(x)[0].reshape(r1.shape[1], N_TOK).T.numpy()


class Completer:
    """I-JEPA-style masked latent completion at the receiver (B-E2): after the message of frame t is applied, the
    latents of the patches NOT received are re-predicted from the fresh received patches (context), the mask and the
    receiver's previous map. Target space = frozen DINOv2-S + PCA (no trainable target encoder / EMA: this is masked
    latent prediction with a frozen target encoder, not a full JEPA). Deterministic -> the sender replica runs it too."""

    def __init__(self, path, d):
        import torch
        self.torch = torch
        self.net = make_net(d, in_ch=2 * d + 1)
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()

    def __call__(self, cur, mask, prev):
        t = self.torch
        x = np.concatenate([cur, mask[:, None].astype(np.float32), prev], 1).T.reshape(1, -1, GH, GW)
        with t.no_grad():
            delta = self.net(t.from_numpy(x.astype(np.float32)))[0].reshape(cur.shape[1], N_TOK).T.numpy()
        out = cur.copy()
        out[~mask] += delta[~mask]
        return out


def make_net(d, width=96, in_ch=None):
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(in_ch or 2 * d, width, 5, padding=2), nn.GELU(),
                         nn.Conv2d(width, width, 5, padding=4, dilation=2), nn.GELU(),
                         nn.Conv2d(width, width, 5, padding=4, dilation=2), nn.GELU(),
                         nn.Conv2d(width, d, 3, padding=1))


# ---------------------------------------------------------------- closed loop
def transmit(Z, pred, k, step, learned=None, completer=None):
    """Z (T, N_TOK, d) normalised PCA coefficients. Returns receiver maps (T, N_TOK, d), bits (T,).
    pred 'complete' = hold-last prediction + receiver-side masked completion (completer) after each message."""
    if pred == "complete":
        assert completer is not None
    T, _, d = Z.shape
    zero = np.zeros((N_TOK, d), np.float32)
    rep = [zero.copy(), zero.copy()]          # sender's replica of the receiver history (r_{t-1}, r_{t-2})
    rxh = [zero.copy(), zero.copy()]          # receiver history (kept separately; must stay bit-identical)
    out, bits = np.zeros_like(Z, dtype=np.float32), np.zeros(T, np.int64)

    def predict(h, t, mv):
        if t == 0:
            return np.zeros((N_TOK, d), np.float32)
        if pred in ("hold", "complete"):
            return h[0]
        if pred == "mc":
            return shift_map(h[0], *mv)
        if pred == "learned":
            return learned(h[0], h[1])
        raise ValueError(pred)

    def update(h, p, idx, rq, t):
        new = p.copy()
        new[idx] = p[idx] + rq * step
        if pred == "complete" and t > 0:
            mask = np.zeros(N_TOK, bool); mask[idx] = True
            new = completer(new, mask, h[0])
        return [new, h[0]]

    for t in range(T):
        z = Z[t].astype(np.float32)
        # ---- sender (sees z_t and its replica only)
        mv = best_mv(rep[0], z) if (pred == "mc" and t > 0) else None
        p = predict(rep, t, mv)
        err = ((z - p) ** 2).sum(1)
        kk = N_TOK if t == 0 else min(k, N_TOK)
        idx = np.sort(np.argpartition(-err, kk - 1)[:kk]) if kk else np.zeros(0, np.int64)
        rq = np.rint((z[idx] - p[idx]) / step).astype(np.int64)
        data, nb = encode_msg(t, idx, rq, mv)
        rep = update(rep, p, idx, rq, t)
        # ---- receiver (sees only the bitstream and its own history)
        f, idx2, rq2, mv2 = decode_msg(data, nb)
        rxh = update(rxh, predict(rxh, t, mv2), idx2, rq2, t)
        assert np.array_equal(rxh[0], rep[0]), "receiver state and sender replica diverged"
        out[t], bits[t] = rxh[0], nb
    return out, bits


# ---------------------------------------------------------------- extraction (GPU)
def extract(a):
    import torch
    from PIL import Image
    from transformers import Dinov2Model
    clips = {k: v for k, v in json.loads(Path(a.clips).read_text()).items() if not k.startswith("_")}
    quals = [int(x) for x in a.avif.split(",")] if a.avif else []
    dev = "cuda"
    m = Dinov2Model.from_pretrained(a.model_dir, use_safetensors=True).eval().to(dev).half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    def tok(x):
        with torch.no_grad():
            xt = torch.from_numpy(np.stack(x)).to(dev).permute(0, 3, 1, 2).float().div(255)
            return m(pixel_values=((xt - mean) / std).half()).last_hidden_state[:, 1:, :].float().cpu().numpy().astype(np.float16)

    stats, t_all = {}, time.time()
    for role, names in clips.items():
        for c in names:
            if (out / f"{c}.npz").exists():
                continue
            split = "val" if role == "aux_val" else "train"
            root = Path(a.vid_root) / f"VisDrone2019-VID-{split}"
            frames = sorted((root / "sequences" / c).glob("*.jpg"))[::a.stride][:a.max_frames]
            ann = np.loadtxt(root / "annotations" / f"{c}.txt", delimiter=",", ndmin=2)
            toks, labs, vals = [], [], []
            av = {q: ([], []) for q in quals} if role != "tune" else {}
            for i in range(0, len(frames), a.batch):
                ims = [Image.open(f).convert("RGB") for f in frames[i:i + a.batch]]
                hw = [(im.height, im.width) for im in ims]
                rs = [im.resize((B0.W_IMG, B0.H_IMG), Image.BICUBIC) for im in ims]
                toks.append(tok([np.asarray(r) for r in rs]))
                for q, (tl, bl) in av.items():
                    dec = []
                    for r in rs:
                        buf = io.BytesIO(); r.save(buf, "AVIF", quality=q, speed=6, subsampling="4:2:0")
                        bl.append(len(buf.getvalue()) * 8)
                        dec.append(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
                    tl.append(tok(dec))
                for j, f in enumerate(frames[i:i + a.batch]):
                    lb, vd = B0.patch_labels(ann[ann[:, 0] == int(f.stem)], hw[j])
                    labs.append(lb); vals.append(vd)
            extra = {}
            for q, (tl, bl) in av.items():
                extra[f"avif{q}_tokens"] = np.concatenate(tl)
                extra[f"avif{q}_bits"] = np.array(bl)
            np.savez(out / f"{c}.npz", tokens=np.concatenate(toks), labels=np.stack(labs), valid=np.stack(vals),
                     frames=np.array([int(f.stem) for f in frames]), **extra)
            stats[c] = dict(role=role, frames=len(frames))
            print(c, role, len(frames), f"{time.time() - t_all:.0f}s", flush=True)
    stats["_peak_gpu_mem_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    stats["_wall_s"] = time.time() - t_all
    (out / "extract_stats.json").write_text(json.dumps(stats, indent=1))


# ---------------------------------------------------------------- fit (CPU): PCA, head, projected train maps
def fit(a):
    clips = json.loads(Path(a.clips).read_text())
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tr = [c for c in clips["tune"] if (Path(a.tokens_dir) / f"{c}.npz").exists()]
    Xs, Ys, Ms = [], [], []
    for c in tr:
        D = np.load(Path(a.tokens_dir) / f"{c}.npz")
        X = D["tokens"].reshape(-1, DIM)
        sel = rng.random(len(X)) < 0.08
        Xs.append(X[sel].astype(np.float32)); Ys.append(D["labels"].reshape(-1, 4)[sel]); Ms.append(D["valid"].reshape(-1)[sel])
    X, Y, M = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms)
    mu = X.mean(0)
    Xc = X - mu
    evals, evecs = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))          # 384 x 384 covariance
    order = np.argsort(evals)[::-1]
    var, Vt = evals[order], evecs[:, order].T
    del Xc
    head = B0.train_head(X, Y, M)
    np.savez(work / "pca_head.npz", mu=mu, Vt=Vt.astype(np.float32), var=var.astype(np.float32),
             explained=(var / var.sum()).astype(np.float32))
    # the head is affine: recover (W, b) by evaluating it on 0 and on the basis vectors
    b = head(np.zeros((1, DIM), np.float32))[0]
    Wh = head(np.eye(DIM, dtype=np.float32)) - b
    np.savez(work / "head.npz", W=Wh.astype(np.float32), b=b.astype(np.float32))
    for d in (32, 64):
        V = Vt[:d].astype(np.float32); sd = np.sqrt(var[:d]).astype(np.float32)
        maps = []
        for c in tr:
            D = np.load(Path(a.tokens_dir) / f"{c}.npz")
            Z = ((D["tokens"].astype(np.float32) - mu) @ V.T) / sd
            maps.append(Z.astype(np.float16))
        np.savez(work / f"train_maps_d{d}.npz", **{c: z for c, z in zip(tr, maps)})
    ev = np.cumsum(var / var.sum())
    print("train patches", len(X), "explained 32/64/128:", ev[31], ev[63], ev[127], flush=True)
    (work / "fit_stats.json").write_text(json.dumps(dict(train_clips=tr, n_patches=int(len(X)),
                                                         explained_32=float(ev[31]), explained_64=float(ev[63]),
                                                         explained_128=float(ev[127])), indent=1))


# ---------------------------------------------------------------- train learned predictor (GPU)
def train(a):
    import torch
    work = Path(a.work)
    D = np.load(work / f"train_maps_d{a.d}.npz")
    seqs = [D[c].astype(np.float32) for c in D.files if len(D[c]) >= 3]
    dev = "cuda"
    torch.manual_seed(0)
    comp = a.kind == "complete"
    net = make_net(a.d, in_ch=2 * a.d + 1 if comp else None).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    t0, losses = time.time(), []

    def corrupt(z_prev_true, z_old):
        # receiver-like input: quantisation noise everywhere, and a random fraction of patches stale (from older frame)
        x = z_prev_true + rng.normal(scale=a.step / math.sqrt(12), size=z_prev_true.shape).astype(np.float32)
        stale = rng.random(z_prev_true.shape[0]) < rng.uniform(0.3, 0.9)
        x[stale] = z_old[stale]
        return x

    for it in range(a.iters):
        xb, yb, wb = [], [], []
        for _ in range(a.batch):
            s = seqs[rng.integers(len(seqs))]
            t = int(rng.integers(3, len(s))) if len(s) > 3 else 2
            back = int(rng.integers(1, 4))
            r1 = corrupt(s[t - 1], s[max(t - 1 - back, 0)])
            if comp:
                # masked latent completion: fresh patches = top-k error of the hold prediction (as the sender picks)
                kk = max(1, int(rng.uniform(0.02, 0.3) * N_TOK))
                err = ((s[t] - r1) ** 2).sum(1)
                mask = np.zeros(N_TOK, bool); mask[np.argpartition(-err, kk - 1)[:kk]] = True
                cur = r1.copy()
                cur[mask] = s[t][mask] + rng.normal(scale=a.step / math.sqrt(12), size=(kk, a.d)).astype(np.float32)
                xb.append(np.concatenate([cur, mask[:, None].astype(np.float32), r1], 1).T.reshape(-1, GH, GW))
                yb.append((s[t] - cur).T.reshape(-1, GH, GW))
                wb.append(np.broadcast_to((~mask).reshape(1, GH, GW), (a.d, GH, GW)).astype(np.float32))
            else:
                r2 = corrupt(s[t - 2], s[max(t - 2 - back, 0)])
                xb.append(np.concatenate([r1, r2], 1).T.reshape(-1, GH, GW))
                yb.append((s[t] - r1).T.reshape(-1, GH, GW))
                wb.append(np.ones((a.d, GH, GW), np.float32))
        x = torch.from_numpy(np.stack(xb)).to(dev); y = torch.from_numpy(np.stack(yb)).to(dev)
        w = torch.from_numpy(np.stack(wb)).to(dev)
        loss = (((net(x) - y) ** 2) * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
        if (it + 1) % 500 == 0:
            print(it + 1, np.mean(losses[-500:]), f"{time.time() - t0:.0f}s", flush=True)
    name = f"{'comp' if comp else 'pred'}_d{a.d}"
    torch.save(net.cpu().state_dict(), work / f"{name}.pt")
    (work / f"{name}_stats.json").write_text(json.dumps(dict(params=n_par, iters=a.iters, batch=a.batch,
                                                                 final_loss=float(np.mean(losses[-500:])),
                                                                 train_s=time.time() - t0), indent=1))


# ---------------------------------------------------------------- eval (CPU)
G = {}


def _init(work, tokens_dir):
    import torch
    torch.set_num_threads(1)
    G["work"], G["tok"] = Path(work), Path(tokens_dir)
    P = np.load(G["work"] / "pca_head.npz"); Hh = np.load(G["work"] / "head.npz")
    G["mu"], G["Vt"], G["var"], G["W"], G["b"] = P["mu"], P["Vt"], P["var"], Hh["W"], Hh["b"]


def head_scores(tokens384):
    return tokens384.astype(np.float32) @ G["W"] + G["b"]


def eval_job(j):
    clip, d, pred, k, step = j
    D = np.load(G["tok"] / f"{clip}.npz")
    X = D["tokens"].astype(np.float32)
    Y, M = D["labels"].reshape(-1, 4), D["valid"].reshape(-1)
    V = G["Vt"][:d]; sd = np.sqrt(G["var"][:d])
    Z = ((X - G["mu"]) @ V.T) / sd
    learned = LearnedPred(G["work"] / f"pred_d{d}.pt", d) if pred == "learned" else None
    completer = Completer(G["work"] / f"comp_d{d}.pt", d) if pred == "complete" else None
    t0 = time.time()
    if pred == "pca_only":
        rx, bits = Z, np.zeros(len(Z))
    else:
        rx, bits = transmit(Z, pred, k, step, learned, completer)
    tok384 = (rx * sd) @ V + G["mu"]
    ap = B0.task_ap(head_scores(tok384.reshape(-1, DIM)), Y, M)
    return dict(clip=clip, d=d, pred=pred, k=k, step=step, frames=len(Z), ap=ap,
                bits_mean_all=float(bits.mean()), bits_init=float(bits[0]),
                bits_mean_after_init=float(bits[1:].mean()) if len(bits) > 1 else None, sim_s=time.time() - t0)


def eval_ref(clip):
    D = np.load(G["tok"] / f"{clip}.npz")
    Y, M = D["labels"].reshape(-1, 4), D["valid"].reshape(-1)
    out = dict(clip=clip, frames=len(D["tokens"]),
               ap_original=B0.task_ap(head_scores(D["tokens"].reshape(-1, DIM)), Y, M), avif={})
    for key in D.files:
        if key.endswith("_tokens") and key.startswith("avif"):
            q = key[4:-7]
            out["avif"][q] = dict(ap=B0.task_ap(head_scores(D[key].reshape(-1, DIM)), Y, M),
                                  bits_mean=float(D[f"avif{q}_bits"].mean()))
    return out


def evaluate(a):
    clips = json.loads(Path(a.clips).read_text())
    ev = [c for r in ("gate", "aux_val") for c in clips.get(r, []) if (Path(a.tokens_dir) / f"{c}.npz").exists()]
    preds = a.preds.split(",")
    jobs = []
    for c in ev:
        for d in (32, 64):
            jobs.append((c, d, "pca_only", 0, 0.0))
            for pred in preds:
                for k in (16, 32, 64, 128, 256):
                    jobs.append((c, d, pred, k, a.step))
    with Pool(a.procs, initializer=_init, initargs=(a.work, a.tokens_dir)) as p:
        refs = p.map(eval_ref, ev)
        res = p.map(eval_job, jobs, chunksize=1)
    Path(a.out).write_text(json.dumps(dict(eval_clips=ev, step=a.step, refs=refs, runs=res), indent=1))
    # summary: mean over clips
    import collections
    agg = collections.defaultdict(list)
    for r in res:
        agg[(r["d"], r["pred"], r["k"])].append(r)
    print("original AP", np.mean([r["ap_original"] for r in refs]))
    for q in refs[0]["avif"]:
        print(f"AVIF q{q}: AP {np.mean([r['avif'][q]['ap'] for r in refs]):.3f} bits {np.mean([r['avif'][q]['bits_mean'] for r in refs]):.0f}")
    for key in sorted(agg):
        rs = agg[key]
        print(key, f"AP {np.mean([r['ap'] for r in rs]):.3f} bits_all {np.mean([r['bits_mean_all'] for r in rs]):.0f} "
                   f"after_init {np.mean([r['bits_mean_after_init'] or 0 for r in rs]):.0f}", flush=True)


def diag(a):
    """Open-loop one-step prediction error on the evaluation clips (true history, no channel): hold / mc / learned;
    and masked completion with the top-64 hold-error patches given exactly (error on the other patches)."""
    clips = json.loads(Path(a.clips).read_text())
    ev = [c for r in ("gate", "aux_val") for c in clips.get(r, []) if (Path(a.tokens_dir) / f"{c}.npz").exists()]
    _init(a.work, a.tokens_dir)
    out = {}
    for d in (32, 64):
        V = G["Vt"][:d]; sd = np.sqrt(G["var"][:d])
        lp = LearnedPred(G["work"] / f"pred_d{d}.pt", d)
        cp = Completer(G["work"] / f"comp_d{d}.pt", d)
        acc = {k: [] for k in ("hold", "mc", "learned", "hold_nonfresh64", "complete_nonfresh64")}
        for c in ev:
            Z = ((np.load(G["tok"] / f"{c}.npz")["tokens"].astype(np.float32) - G["mu"]) @ V.T) / sd
            for t in range(2, len(Z)):
                z, r1, r2 = Z[t], Z[t - 1], Z[t - 2]
                acc["hold"].append(((z - r1) ** 2).mean())
                acc["mc"].append(((z - shift_map(r1, *best_mv(r1, z))) ** 2).mean())
                acc["learned"].append(((z - lp(r1, r2)) ** 2).mean())
                err = ((z - r1) ** 2).sum(1)
                m = np.zeros(N_TOK, bool); m[np.argpartition(-err, 63)[:64]] = True
                cur = r1.copy(); cur[m] = z[m]
                acc["hold_nonfresh64"].append(((z - cur)[~m] ** 2).mean())
                acc["complete_nonfresh64"].append(((z - cp(cur, m, r1))[~m] ** 2).mean())
        out[f"d{d}"] = {k: float(np.mean(v)) for k, v in acc.items()}
        print(d, out[f"d{d}"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    for n in ("clips", "vid_root", "model_dir", "out_dir"):
        e.add_argument(n)
    e.add_argument("--stride", type=int, default=3)
    e.add_argument("--max-frames", type=int, default=100)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--avif", default="5,10,20,35,50")
    f = sub.add_parser("fit")
    for n in ("tokens_dir", "clips", "work"):
        f.add_argument(n)
    t = sub.add_parser("train")
    t.add_argument("work")
    t.add_argument("--d", type=int, default=64)
    t.add_argument("--iters", type=int, default=6000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--step", type=float, default=0.25)
    t.add_argument("--kind", default="pred", choices=["pred", "complete"])
    v = sub.add_parser("eval")
    for n in ("tokens_dir", "clips", "work", "out"):
        v.add_argument(n)
    v.add_argument("--preds", default="hold,mc,learned")
    v.add_argument("--step", type=float, default=0.25)
    v.add_argument("--procs", type=int, default=6)
    g = sub.add_parser("diag")
    for n in ("tokens_dir", "clips", "work", "out"):
        g.add_argument(n)
    a = ap.parse_args()
    dict(extract=extract, fit=fit, train=train, eval=evaluate, diag=diag)[a.cmd](a)


if __name__ == "__main__":
    main()
