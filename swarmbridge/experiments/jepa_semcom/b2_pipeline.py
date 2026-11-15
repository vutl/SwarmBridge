"""B-E3 (review #29): task-preserving learned bottleneck, and a matched JEPA vs non-JEPA representation comparison.

Input: frozen DINOv2-S token maps (384 x 22 x 39) of VisDrone-VID (B-E1 extraction, split v1). The representation
that is transmitted is a learned code map z = f(x) (d x 22 x 39), decoded at the receiver by g(z) -> 384-d tokens and
scored with the SAME fixed linear task head as B-E0/B-E1 (trained on original TUNE tokens).
Matched backbone for every representation: encoder f = Conv1x1(384->256) GELU Conv3x3(256->d); decoder
g = Conv3x3(d->256) GELU Conv1x1(256->384).
  task   (non-JEPA)  loss = token reconstruction MSE (normalised tokens) + task BCE of the fixed head on g(f(x))
  jepa               the same losses on the ONLINE encoder, plus a JEPA term: a predictor h (same architecture as
                     the B-E1 learned predictor) maps the online codes of frames t-1, t-2 to the codes of frame t
                     produced by an EMA TARGET encoder (momentum 0.99, stop-gradient), plus VICReg variance /
                     covariance regularisation of the online codes (anti-collapse). The JEPA term makes the code
                     space temporally predictable; reconstruction + task keep it task-preserving.
After training, codes are normalised per dimension (train std). For a fair predictor comparison every
representation gets the SAME post-hoc predictor (B-E1 conv net, MSE, receiver-like corrupted inputs) trained on its
frozen TUNE codes; the JEPA model's own predictor h is evaluated as well ('own').
Evaluation: B-E1 round-trip codec (b1_pipeline.transmit; separate sender / receiver histories asserted identical),
predictors hold / mc / posthoc / own, k in {16, 32, 64, 128, 256}, closed loop, per clip.

  train  TOKENS CLIPS B1_WORK OUT_WORK [--d 32 --iters 4000 --seed S --kinds task,jepa]   (GPU, under the lock)
  eval   TOKENS CLIPS B1_WORK OUT_WORK OUT_JSON [--dims 32 --reps task_s0,jepa_s0,..]   (CPU)

B-E4 (review #30): paired seeds. For each kind the NumPy minibatch sampler AND torch are reset from the same seed S,
so task and jepa models of seed S start from identical encoder/decoder weights and see the identical sequence of
training clips/frames; the post-hoc predictor is likewise seeded by S for both. Models are saved as
{kind}_s{S}_d{d}.pt (B-E3 used a single sampler shared by the two kinds: same init, different minibatches; its
seed-0 task model equals task_s0 of this scheme up to GPU nondeterminism).
"""
import argparse
import copy
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import b0_pipeline as B0
import b1_pipeline as B1

GH, GW, DIM, N_TOK = B0.GH, B0.GW, B0.DIM, B0.N_TOK


def make_codec(d):
    import torch.nn as nn
    enc = nn.Sequential(nn.Conv2d(DIM, 256, 1), nn.GELU(), nn.Conv2d(256, d, 3, padding=1))
    dec = nn.Sequential(nn.Conv2d(d, 256, 3, padding=1), nn.GELU(), nn.Conv2d(256, DIM, 1))
    return enc, dec


def load_maps(tokens_dir, names):
    """list of (T, 384, GH, GW) float16 arrays + labels / valid per clip."""
    out = []
    for c in names:
        p = Path(tokens_dir) / f"{c}.npz"
        if not p.exists():
            continue
        D = np.load(p)
        X = D["tokens"].reshape(-1, GH, GW, DIM).transpose(0, 3, 1, 2)
        out.append((c, X, D["labels"].reshape(-1, GH, GW, 4).transpose(0, 3, 1, 2), D["valid"].reshape(-1, GH, GW)))
    return out


def vicreg(z, gamma=1.0):
    """variance hinge + covariance penalty over patches of a batch of code maps (B, d, H, W)."""
    import torch
    zz = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    zz = zz - zz.mean(0)
    std = torch.sqrt(zz.var(0) + 1e-4)
    var_l = torch.relu(gamma - std).mean()
    cov = (zz.T @ zz) / (len(zz) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_l, (off ** 2).sum() / zz.shape[1]


def train(a):
    import torch
    import torch.nn.functional as F
    dev = "cuda"
    clips = json.loads(Path(a.clips).read_text())
    P = np.load(Path(a.b1_work) / "pca_head.npz"); Hh = np.load(Path(a.b1_work) / "head.npz")
    mu = torch.tensor(P["mu"], device=dev).view(1, DIM, 1, 1)
    sd_tok = torch.tensor(np.sqrt(np.maximum(np.diag(P["Vt"].T @ np.diag(P["var"]) @ P["Vt"]), 1e-6)),
                          device=dev, dtype=torch.float32).view(1, DIM, 1, 1)
    Wh = torch.tensor(Hh["W"], device=dev); bh = torch.tensor(Hh["b"], device=dev)
    X, Yl, Vl = [], [], []                                                  # moved to the GPU clip by clip (RAM)
    for c in clips["tune"]:
        got = load_maps(a.tokens_dir, [c])
        if not got:
            continue
        _, x, y, v = got[0]
        X.append(torch.from_numpy(x).to(dev)); Yl.append(torch.from_numpy(y.astype(np.float32)).to(dev))
        Vl.append(torch.from_numpy(v).to(dev))
        del got, x, y, v
    pos = torch.cat([y.permute(0, 2, 3, 1).reshape(-1, 4) for y in Yl]).mean(0).clamp(1e-4, 1)
    pw = ((1 - pos) / pos).clamp(max=50)
    out = Path(a.out_work); out.mkdir(parents=True, exist_ok=True)
    rng = None
    stats = {}

    def batch(bs, need=3):
        xs, ys, vs = [], [], []
        for _ in range(bs):
            i = int(rng.integers(len(X)))
            while len(X[i]) < need:
                i = int(rng.integers(len(X)))
            t = int(rng.integers(need - 1, len(X[i])))
            xs.append(X[i][t - need + 1:t + 1]); ys.append(Yl[i][t]); vs.append(Vl[i][t])
        x = torch.stack(xs).float()                                         # (B, need, 384, H, W)
        return (x - mu.unsqueeze(1)) / sd_tok.unsqueeze(1), torch.stack(ys), torch.stack(vs)

    def task_loss(dec_norm, y, v):
        tok = dec_norm * sd_tok + mu                                        # back to raw token space
        logit = torch.einsum("bchw,co->bohw", tok, Wh) + bh.view(1, 4, 1, 1)
        l = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw.view(1, 4, 1, 1), reduction="none")
        m = v.unsqueeze(1).float()
        return (l * m).sum() / (m.sum() * 4 + 1e-6)

    for kind in a.kinds.split(","):
        rng = np.random.default_rng(a.seed)                                  # paired: same batch schedule per kind
        torch.manual_seed(a.seed)
        enc, dec = make_codec(a.d)
        enc, dec = enc.to(dev), dec.to(dev)
        params = list(enc.parameters()) + list(dec.parameters())
        pred = None
        if kind == "jepa":
            tgt = copy.deepcopy(enc)
            for p_ in tgt.parameters():
                p_.requires_grad_(False)
            pred = B1.make_net(a.d).to(dev)
            params += list(pred.parameters())
        opt = torch.optim.AdamW(params, 1e-3, weight_decay=1e-4)
        t0, hist = time.time(), []
        for it in range(a.iters):
            x, y, v = batch(a.batch)
            xt = x[:, -1]
            z = enc(xt)
            rec = dec(z)
            l_rec = F.mse_loss(rec, xt)
            l_task = task_loss(rec, y, v)
            loss = l_rec + a.w_task * l_task
            logs = [float(l_rec), float(l_task)]
            if kind == "jepa":
                z1, z2 = enc(x[:, -2]), enc(x[:, -3])
                with torch.no_grad():
                    zt = tgt(xt)
                zp = z1 + pred(torch.cat([z1, z2], 1))
                l_j = F.mse_loss(zp, zt)
                var_l, cov_l = vicreg(z)
                loss = loss + a.w_jepa * l_j + a.w_var * var_l + a.w_cov * cov_l
                logs += [float(l_j), float(var_l), float(cov_l)]
            opt.zero_grad(); loss.backward(); opt.step()
            if kind == "jepa":
                with torch.no_grad():
                    for pt, po in zip(tgt.parameters(), enc.parameters()):
                        pt.mul_(0.99).add_(po.detach(), alpha=0.01)
            hist.append(logs)
            if (it + 1) % 500 == 0:
                print(kind, a.d, it + 1, np.round(np.mean(hist[-500:], 0), 4).tolist(), f"{time.time() - t0:.0f}s", flush=True)
        # code statistics on TUNE (per-dim std) for normalisation
        with torch.no_grad():
            zs = torch.cat([enc(((xx[:: 4].float() - mu) / sd_tok)).permute(0, 2, 3, 1).reshape(-1, a.d) for xx in X])
            zmu, zsd = zs.mean(0).cpu().numpy(), zs.std(0).clamp(min=1e-4).cpu().numpy()
        n_par = sum(p_.numel() for p_ in list(enc.parameters()) + list(dec.parameters()))
        torch.save(dict(enc=enc.cpu().state_dict(), dec=dec.cpu().state_dict(),
                        pred=pred.cpu().state_dict() if pred is not None else None,
                        zmu=zmu, zsd=zsd, mu=P["mu"], sd_tok=sd_tok.cpu().numpy().ravel()),
                   out / f"{kind}_s{a.seed}_d{a.d}.pt")
        stats[kind] = dict(codec_params=int(n_par), pred_params=int(sum(p_.numel() for p_ in pred.parameters())) if pred else 0,
                           iters=a.iters, final=np.mean(hist[-500:], 0).tolist(), train_s=time.time() - t0)
        enc, dec = enc.to(dev), dec.to(dev)
        # post-hoc predictor on frozen normalised codes (identical procedure for every representation)
        with torch.no_grad():
            seqs = []
            for xx in X:
                zc = enc(((xx.float() - mu) / sd_tok)).cpu().numpy()
                seqs.append(((zc.transpose(0, 2, 3, 1).reshape(len(zc), N_TOK, a.d) - zmu) / zsd).astype(np.float32))
        stats[kind]["posthoc"] = train_posthoc(seqs, a.d, out / f"{kind}_s{a.seed}_d{a.d}_posthoc.pt", a.posthoc_iters,
                                               dev, seed=a.seed)
        del enc, dec
        torch.cuda.empty_cache()
    (out / f"train_stats_s{a.seed}_d{a.d}.json").write_text(json.dumps(stats, indent=1))


def train_posthoc(seqs, d, path, iters, dev, step=0.25, bs=32, seed=0):
    """the B-E1 learned predictor procedure (receiver-like corrupted inputs, residual MSE) on given code sequences."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    net = B1.make_net(d).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    seqs = [s for s in seqs if len(s) >= 3]

    def corrupt(z, old):
        x = z + rng.normal(scale=step / math.sqrt(12), size=z.shape).astype(np.float32)
        st = rng.random(len(z)) < rng.uniform(0.3, 0.9)
        x[st] = old[st]
        return x
    losses, t0 = [], time.time()
    for it in range(iters):
        xb, yb = [], []
        for _ in range(bs):
            s = seqs[rng.integers(len(seqs))]
            t = int(rng.integers(2, len(s)))
            back = int(rng.integers(1, 4))
            r1 = corrupt(s[t - 1], s[max(t - 1 - back, 0)]); r2 = corrupt(s[t - 2], s[max(t - 2 - back, 0)])
            xb.append(np.concatenate([r1, r2], 1).T.reshape(-1, GH, GW)); yb.append((s[t] - r1).T.reshape(-1, GH, GW))
        x = torch.from_numpy(np.stack(xb)).to(dev); y = torch.from_numpy(np.stack(yb)).to(dev)
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    torch.save(net.cpu().state_dict(), path)
    return dict(iters=iters, final_loss=float(np.mean(losses[-500:])), train_s=time.time() - t0,
                params=int(sum(p.numel() for p in net.parameters())))


# ---------------------------------------------------------------- evaluation (CPU)
G = {}


def _init(out_work, b1_work, tokens_dir):
    import torch
    torch.set_num_threads(1)
    G["work"], G["b1"], G["tok"] = Path(out_work), Path(b1_work), Path(tokens_dir)
    Hh = np.load(G["b1"] / "head.npz")
    G["W"], G["b"] = Hh["W"], Hh["b"]
    G["models"] = {}


class Rep:
    def __init__(self, path, d):
        import torch
        self.torch = torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.enc, self.dec = make_codec(d)
        self.enc.load_state_dict(ck["enc"]); self.dec.load_state_dict(ck["dec"])
        self.enc.eval(); self.dec.eval()
        self.zmu, self.zsd, self.mu, self.sd = ck["zmu"], ck["zsd"], ck["mu"], ck["sd_tok"]
        self.d = d
        self.own = None
        if ck["pred"] is not None:
            self.own = B1.make_net(d); self.own.load_state_dict(ck["pred"]); self.own.eval()

    def codes(self, X):                         # X (T, N_TOK, 384) raw -> normalised codes (T, N_TOK, d)
        t = self.torch
        with t.no_grad():
            x = ((X - self.mu) / self.sd).reshape(-1, GH, GW, DIM).transpose(0, 3, 1, 2)
            z = self.enc(t.from_numpy(np.ascontiguousarray(x, np.float32))).numpy()
        return ((z.transpose(0, 2, 3, 1).reshape(len(X), N_TOK, self.d) - self.zmu) / self.zsd).astype(np.float32)

    def tokens(self, Z):                        # normalised codes -> raw 384-d tokens
        t = self.torch
        with t.no_grad():
            z = (Z * self.zsd + self.zmu).reshape(-1, GH, GW, self.d).transpose(0, 3, 1, 2)
            x = self.dec(t.from_numpy(np.ascontiguousarray(z, np.float32))).numpy()
        return x.transpose(0, 2, 3, 1).reshape(len(Z), N_TOK, DIM) * self.sd + self.mu

    def own_pred(self, r1, r2):                 # JEPA's own predictor, in normalised code units
        t = self.torch
        z1 = r1 * self.zsd + self.zmu; z2 = r2 * self.zsd + self.zmu
        with t.no_grad():
            x = t.from_numpy(np.concatenate([z1, z2], 1).T.reshape(1, -1, GH, GW).astype(np.float32))
            zp = z1 + self.own(x)[0].reshape(self.d, N_TOK).T.numpy()
        return ((zp - self.zmu) / self.zsd).astype(np.float32)


def eval_job(j):
    clip, rep_name, d, pred, k = j
    key = (rep_name, d)
    if key not in G["models"]:
        G["models"][key] = Rep(G["work"] / f"{rep_name}_d{d}.pt", d)
    rep = G["models"][key]
    D = np.load(G["tok"] / f"{clip}.npz")
    X = D["tokens"].astype(np.float32)
    Y, M = D["labels"].reshape(-1, 4), D["valid"].reshape(-1)
    Z = rep.codes(X)
    learned = None
    if pred == "posthoc":
        learned = B1.LearnedPred(G["work"] / f"{rep_name}_d{d}_posthoc.pt", d)
    elif pred == "own":
        learned = rep.own_pred
    t0 = time.time()
    if pred == "none":
        rx, bits = Z, np.zeros(len(Z))
    else:
        rx, bits = B1.transmit(Z, "learned" if pred in ("posthoc", "own") else pred, k, 0.25, learned)
    tok = rep.tokens(rx)
    ap = B0.task_ap(tok.reshape(-1, DIM) @ G["W"] + G["b"], Y, M)
    return dict(clip=clip, rep=rep_name, d=d, pred=pred, k=k, ap=ap, bits_mean_all=float(bits.mean()),
                bits_init=float(bits[0]), sim_s=time.time() - t0)


def evaluate(a):
    clips = json.loads(Path(a.clips).read_text())
    ev = [c for r in ("gate", "aux_val") for c in clips.get(r, []) if (Path(a.tokens_dir) / f"{c}.npz").exists()]
    jobs = []
    for d in a.dims:
        for rep in a.reps.split(","):
            for c in ev:
                jobs.append((c, rep, d, "none", 0))
                for pred in ("hold", "mc", "posthoc") + (("own",) if rep.startswith("jepa") else ()):
                    for k in (16, 32, 64, 128, 256):
                        jobs.append((c, rep, d, pred, k))
    with Pool(a.procs, initializer=_init, initargs=(a.out_work, a.b1_work, a.tokens_dir)) as p:
        res = p.map(eval_job, jobs, chunksize=1)
    Path(a.out).write_text(json.dumps(dict(eval_clips=ev, runs=res), indent=1))
    import collections
    agg = collections.defaultdict(list)
    for r in res:
        agg[(r["rep"], r["d"], r["pred"], r["k"])].append(r)
    for key in sorted(agg):
        rs = agg[key]
        print(key, f"AP {np.mean([r['ap'] for r in rs]):.3f} bits {np.mean([r['bits_mean_all'] for r in rs]):.0f}", flush=True)


def diag(a):
    """open-loop one-step MSE in each representation's normalised code space (true history)."""
    clips = json.loads(Path(a.clips).read_text())
    ev = [c for r in ("gate", "aux_val") for c in clips.get(r, []) if (Path(a.tokens_dir) / f"{c}.npz").exists()]
    _init(a.out_work, a.b1_work, a.tokens_dir)
    out = {}
    for d in a.dims:
        for name in a.reps.split(","):
            rep = Rep(G["work"] / f"{name}_d{d}.pt", d)
            ph = B1.LearnedPred(G["work"] / f"{name}_d{d}_posthoc.pt", d)
            acc = {k: [] for k in ("hold", "mc", "posthoc", "own")}
            for c in ev:
                Z = rep.codes(np.load(G["tok"] / f"{c}.npz")["tokens"].astype(np.float32))
                for t in range(2, len(Z)):
                    z, r1, r2 = Z[t], Z[t - 1], Z[t - 2]
                    acc["hold"].append(((z - r1) ** 2).mean())
                    acc["mc"].append(((z - B1.shift_map(r1, *B1.best_mv(r1, z))) ** 2).mean())
                    acc["posthoc"].append(((z - ph(r1, r2)) ** 2).mean())
                    if rep.own is not None:
                        acc["own"].append(((z - rep.own_pred(r1, r2)) ** 2).mean())
            out[f"{name}_d{d}"] = {k: float(np.mean(v)) for k, v in acc.items() if v}
            print(name, d, out[f"{name}_d{d}"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    for n in ("tokens_dir", "clips", "b1_work", "out_work"):
        t.add_argument(n)
    t.add_argument("--d", type=int, default=32)
    t.add_argument("--iters", type=int, default=4000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--posthoc-iters", type=int, default=6000)
    t.add_argument("--w-task", type=float, default=1.0)
    t.add_argument("--w-jepa", type=float, default=1.0)
    t.add_argument("--w-var", type=float, default=1.0)
    t.add_argument("--w-cov", type=float, default=0.04)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--kinds", default="task,jepa")
    for name in ("eval", "diag"):
        e = sub.add_parser(name)
        for n in ("tokens_dir", "clips", "b1_work", "out_work", "out"):
            e.add_argument(n)
        e.add_argument("--dims", type=int, nargs="+", default=[32, 64])
        e.add_argument("--procs", type=int, default=10)
        e.add_argument("--reps", default="task_s0,jepa_s0")
    a = ap.parse_args()
    dict(train=train, eval=evaluate, diag=diag)[a.cmd](a)


if __name__ == "__main__":
    main()
