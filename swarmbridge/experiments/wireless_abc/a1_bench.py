"""A1: few-pilot BS->UAV channel inference and beam decision on the Sionna RT traces (CPU).

Measurement (downlink beam training, BS 16-ant ULA, UAV 1 antenna): y = sqrt(Ep) W^H h + n, Np in {4, 8, 16} pilots,
two pilot protocols: 'dft' (partial DFT sweep) and 'random' (fixed random unit-modulus beams, seeded). Ep = 1; the
per-pilot reference SNR snr_ref in {-5, 5, 15} dB sets noise_var = 1 / snr_ref for a channel of median train norm
(global normalisation ||h||^2 = N at the train median; never per-sample).
Data phase: beam b from the 16-beam DFT codebook, data SNR = snr_ref * |w_b^H h|^2, rate log2(1 + SNR), outage if
SNR < GAMMA (0 dB).
Split by episode (60/20/20 of trace episodes). Samples whose true channel power is < FLOOR below the train median are
dropped from all sets (unusable links) and counted.
Methods (all get y, W, Ep, noise_var; learned ones also get the LMMSE estimate as an input feature):
  ls      pseudo-inverse (min-norm for Np < N)            lmmse   Gaussian prior fit on train (analytic posterior)
  omp     OMP on a 4x oversampled angular dictionary       gmm     32-component complex GMM prior (analytic posterior)
  map     MLP regression (MSE)                             gauss   MLP heteroscedastic diagonal Gaussian (NLL)
  (training: every train channel x 3 SNRs x REPS=2 noise draws, pre-generated; 3000 AdamW steps, batch 512)
  ddpm    conditional DDPM (eps-pred, 100 steps), sampled with DDIM (eta=1, stochastic) at NFE 1/4/8
  flow    conditional flow matching from N(0, I), Euler NFE 1/4/8
  bridge  Brownian bridge from the LMMSE estimate (I2SB-style, sigma=1), stochastic sampler NFE 1/4/8
  bridge0 same bridge with sigma=0 (deterministic; ablation: does stochasticity matter?)
  sweep   (dft protocol only) pick the pilot beam with the largest |y_k| (classical beam sweep, no estimation)
Decision rules: 'mean' = argmax |w^H h_hat|^2 (point estimate); 'egain' = argmax E|w^H h|^2; 'outage' = argmin
P(SNR_b < GAMMA). Probabilistic methods use S=16 samples (analytic for lmmse/gmm).
Dev/test are capped at NEVAL = 8000 fixed random samples. Output: OUT/a1_results.json (+ per-config rows), printed table. Usage: python a1_bench.py TRACE_DIR OUT [--seeds 0 1 2]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ncx2

import linear as Lc

N = 16
GAMMA = 1.0
FLOOR_DB = 40.0
SNRS_DB = [-5, 5, 15]
NPS = [4, 8, 16]
PROTOS = ["dft", "random"]
S = 16
NFES = [1, 4, 8]
STEPS = 3000
REPS = 2
NEVAL = 8000
DEV = "cpu"
torch.set_num_threads(6)


# ----------------------------------------------------------------------------- data
def load_traces(tdir):
    files = sorted(Path(tdir).glob("ep*.npz"))
    H, ep = [], []
    for f in files:
        d = np.load(f)
        h = d["a"].sum(2).reshape(-1, N)                         # narrowband channel at the RT sample time
        H.append(h)
        ep += [int(f.stem[2:])] * len(h)
    return np.concatenate(H).astype(np.complex128), np.array(ep), len(files)


def split(H, ep, n_ep):
    a, b = int(0.6 * n_ep), int(0.8 * n_ep)
    tr, dv, te = ep < a, (ep >= a) & (ep < b), ep >= b
    p = np.sum(np.abs(H) ** 2, 1)
    med = np.median(p[tr & (p > 0)])
    keep = p > med * 10 ** (-FLOOR_DB / 10)
    scale = np.sqrt(med / N)
    Hn = H / scale
    info = dict(n_total=len(H), dropped_frac=round(float(1 - keep.mean()), 4), scale=float(scale),
                n_train=int((tr & keep).sum()), n_dev=int((dv & keep).sum()), n_test=int((te & keep).sum()))
    return Hn[tr & keep], Hn[dv & keep], Hn[te & keep], info


# ----------------------------------------------------------------------------- classical estimators
def batch_lmmse(Y, A, nv, mu, C):
    Sg = A @ C @ A.conj().T + nv * np.eye(A.shape[0])
    K = C @ A.conj().T @ np.linalg.inv(Sg)
    M = mu[None] + (Y - (A @ mu)[None]) @ K.T
    P = C - K @ A @ C
    return M, 0.5 * (P + P.conj().T)


def fit_gmm(X, K=32, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    mu = X[rng.choice(n, K, replace=False)]
    C0 = np.cov(X.T) + 1e-3 * np.eye(d)
    covs = np.repeat(C0[None], K, 0)
    pis = np.full(K, 1 / K)
    reg = 1e-4 * np.real(np.trace(C0)) / d
    for _ in range(iters):
        logp = np.stack([np.log(pis[k]) + cn_logpdf(X, mu[k], covs[k]) for k in range(K)], 1)
        lse = np.logaddexp.reduce(logp, 1, keepdims=True)
        R = np.exp(logp - lse)
        Nk = R.sum(0) + 1e-9
        pis = Nk / n
        mu = (R.T @ X) / Nk[:, None]
        for k in range(K):
            D = X - mu[k]
            covs[k] = (D.conj().T * R[:, k]) @ D / Nk[k] + reg * np.eye(d)
    return pis, mu, covs


def cn_logpdf(X, m, C):
    L = np.linalg.cholesky(C)
    Z = np.linalg.solve(L, (X - m).T)
    return -X.shape[1] * np.log(np.pi) - 2 * np.sum(np.log(np.real(np.diag(L)))) - np.sum(np.abs(Z) ** 2, 0)


def batch_gmm(Y, A, nv, gmm):
    pis, mus, covs = gmm
    logw, M, P = [], [], []
    for pi, m, C in zip(pis, mus, covs):
        Sg = A @ C @ A.conj().T + nv * np.eye(A.shape[0])
        logw.append(np.log(pi) + cn_logpdf(Y, A @ m, Sg))
        Mk, Pk = batch_lmmse(Y, A, nv, m, C)
        M.append(Mk), P.append(Pk)
    logw = np.stack(logw, 1)
    W = np.exp(logw - np.logaddexp.reduce(logw, 1, keepdims=True))
    return W, np.stack(M, 1), np.stack(P)                    # (n,K), (n,K,N), (K,N,N)


def omp(Y, A, nv, dict_os=4, max_k=4):
    G = np.exp(1j * np.pi * np.outer(np.arange(N), np.linspace(-1, 1, dict_os * N, endpoint=False))) / np.sqrt(N)
    D = A @ G
    out = np.zeros((len(Y), N), complex)
    for i, y in enumerate(Y):
        r, S_ = y.copy(), []
        for _ in range(max_k):
            if np.sum(np.abs(r) ** 2) <= nv * len(y):
                break
            S_.append(int(np.argmax(np.abs(D.conj().T @ r))))
            x, *_ = np.linalg.lstsq(D[:, S_], y, rcond=None)
            r = y - D[:, S_] @ x
        if S_:
            out[i] = G[:, S_] @ x
    return out


# ----------------------------------------------------------------------------- learned models
class MLP(nn.Module):
    def __init__(self, din, dout, w=256, depth=4):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, w), nn.SiLU()]
            d = w
        layers.append(nn.Linear(d, dout))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


def temb(t):
    f = torch.exp(torch.linspace(0, 4, 8, device=t.device))
    return torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], 1)


class Learned:
    """All learned methods share: condition c = [y/scale (2Np), LMMSE estimate (2N), snr_db/10]."""

    def __init__(self, kind, dc, seed):
        torch.manual_seed(seed)
        self.kind = kind
        dx = 2 * N
        if kind == "map":
            self.net = MLP(dc, dx)
        elif kind == "gauss":
            self.net = MLP(dc, 2 * dx)
        else:
            self.net = MLP(dc + dx + 16, dx)
        self.sigma = {"bridge": 1.0, "bridge0": 0.0}.get(kind)
        self.ab = torch.cos((torch.arange(101) / 100 + 0.008) / 1.008 * np.pi / 2) ** 2   # cosine alpha-bar
        self.ab = (self.ab / self.ab[0]).clamp(min=1e-4)

    def loss(self, x, c, x1):
        k = self.kind
        if k == "map":
            return ((self.net(c) - x) ** 2).mean()
        if k == "gauss":
            o = self.net(c)
            m, lv = o[:, :2 * N], o[:, 2 * N:].clamp(-12, 8)
            return (0.5 * (lv + (x - m) ** 2 / lv.exp())).mean()
        n = len(x)
        if k == "ddpm":
            ti = torch.randint(1, 101, (n,), device=x.device)
            ab = self.ab.to(x.device)[ti][:, None]
            e = torch.randn_like(x)
            xt = ab.sqrt() * x + (1 - ab).sqrt() * e
            return ((self.net(torch.cat([xt, c, temb(ti / 100.0)], 1)) - e) ** 2).mean()
        if k == "flow":
            t = torch.rand(n, device=x.device)[:, None]
            x0 = torch.randn_like(x)
            xt = (1 - t) * x0 + t * x
            return ((self.net(torch.cat([xt, c, temb(t[:, 0])], 1)) - (x - x0)) ** 2).mean()
        # bridge: x_t = (1 - t) x + t x1 + sigma sqrt(t (1 - t)) eps, predict x
        t = torch.rand(n, device=x.device)[:, None]
        xt = (1 - t) * x + t * x1 + self.sigma * (t * (1 - t)).sqrt() * torch.randn_like(x)
        return ((self.net(torch.cat([xt, c, temb(t[:, 0])], 1)) - x) ** 2).mean()

    @torch.no_grad()
    def sample(self, c, x1, nfe, gen):
        k, n = self.kind, len(c)
        if k == "map":
            return self.net(c)[None]
        if k == "gauss":
            o = self.net(c)
            m, lv = o[:, :2 * N], o[:, 2 * N:].clamp(-12, 8)
            return m[None] + (0.5 * lv).exp()[None] * torch.randn((S, n, 2 * N), generator=gen, device=c.device)
        c = c.repeat(S, 1)
        x1 = x1.repeat(S, 1)
        if k == "ddpm":
            ts = np.linspace(100, 0, nfe + 1).round().astype(int)
            x = torch.randn((S * n, 2 * N), generator=gen, device=c.device)
            ab_all = self.ab.to(c.device)
            for a, b in zip(ts[:-1], ts[1:]):
                aa, ab_ = ab_all[a], ab_all[b]
                e = self.net(torch.cat([x, c, temb(torch.full((len(x),), a / 100.0, device=c.device))], 1))
                x0 = (x - (1 - aa).sqrt() * e) / aa.sqrt()
                if b == 0:
                    x = x0
                    break
                sig = ((1 - ab_) / (1 - aa) * (1 - aa / ab_)).sqrt()          # DDIM eta = 1
                x = ab_.sqrt() * x0 + (1 - ab_ - sig ** 2).clamp(min=0).sqrt() * e + sig * torch.randn(x.shape, generator=gen, device=c.device)
            return x.reshape(S, n, -1)
        if k == "flow":
            x = torch.randn((S * n, 2 * N), generator=gen, device=c.device)
            for i in range(nfe):
                t = torch.full((len(x),), i / nfe, device=c.device)
                x = x + self.net(torch.cat([x, c, temb(t)], 1)) / nfe
            return x.reshape(S, n, -1)
        # bridge sampler: start at x1 (t = 1), pinned posterior steps toward the predicted x
        ts = np.linspace(1, 0, nfe + 1)
        x = x1.clone()
        for a, b in zip(ts[:-1], ts[1:]):
            x0 = self.net(torch.cat([x, c, temb(torch.full((len(x),), float(a), device=c.device))], 1))
            if b == 0:
                x = x0
                break
            mean = x0 + (b / a) * (x - x0)
            var = self.sigma ** 2 * b * (a - b) / a
            x = mean + np.sqrt(var) * torch.randn(x.shape, generator=gen, device=c.device)
        return x.reshape(S, n, -1)


# ----------------------------------------------------------------------------- evaluation
def cond_feats(Y, Ml, snr_db, ysc):
    return np.concatenate([Lc.c2r(Y) / ysc, Lc.c2r(Ml), np.full((len(Y), 1), snr_db / 10)], 1).astype(np.float32)


def decide_from_samples(Hs, snr):
    """Hs: (S, n, N) complex samples -> beams for egain / outage rules, plus predicted gain samples."""
    F = Lc.dft_codebook(N)
    G = np.abs(Hs @ F.conj()) ** 2                                    # (S, n, B)
    return np.argmax(G.mean(0), 1), np.argmin((snr * G < GAMMA).mean(0) - 1e-6 * G.mean(0), 1), G


def outcomes(H, beams, snr):
    F = Lc.dft_codebook(N)
    g = np.abs(H @ F.conj()) ** 2
    gb = g[np.arange(len(H)), beams]
    best = g.max(1)
    r, rb = Lc.rate(gb, snr), Lc.rate(best, snr)
    return dict(rate=float(r.mean()), outage=float((snr * gb < GAMMA).mean()), regret=float((rb - r).mean()),
                acc=float((beams == g.argmax(1)).mean()))


def coverage(Gs, H):
    """Coverage of central 80% / 95% predictive intervals of every beam's gain."""
    F = Lc.dft_codebook(N)
    g = np.abs(H @ F.conj()) ** 2
    out = {}
    for q in (0.8, 0.95):
        lo, hi = np.quantile(Gs, (1 - q) / 2, axis=0), np.quantile(Gs, 1 - (1 - q) / 2, axis=0)
        out[f"cov{int(q * 100)}"] = float(((g >= lo) & (g <= hi)).mean())
    return out


def energy_score(Hs, H, nmax=2000):
    X = np.concatenate([Hs.real, Hs.imag], -1)[:, :nmax]
    x = np.concatenate([H.real, H.imag], -1)[:nmax]
    t1 = np.linalg.norm(X - x[None], axis=-1).mean(0)
    t2 = np.linalg.norm(X[:, None] - X[None], axis=-1).mean((0, 1))
    return float((t1 - 0.5 * t2).mean())


def nmse_db(Hhat, H):
    return float(10 * np.log10(np.sum(np.abs(Hhat - H) ** 2) / np.sum(np.abs(H) ** 2)))


def analytic_samples(rng, M, P, Wk=None):
    """Draw S posterior samples from Gaussian (M (n,N), P (N,N)) or GMM (Wk (n,K), M (n,K,N), P (K,N,N))."""
    if Wk is None:
        L = np.linalg.cholesky(P + 1e-9 * np.eye(N))
        return M[None] + Lc.crandn(rng, S, len(M), N) @ L.T
    Ls = np.linalg.cholesky(P + 1e-9 * np.eye(N))
    n = len(M)
    out = np.empty((S, n, N), complex)
    for s in range(S):
        u = rng.random(n)[:, None]
        k = (np.cumsum(Wk, 1) < u).sum(1).clip(max=Wk.shape[1] - 1)
        z = Lc.crandn(rng, n, N)
        out[s] = M[np.arange(n), k] + np.einsum("nij,nj->ni", Ls[k], z)
    return out


def run(Htr, Hdv, Hte, proto, n_p, seed, gmm, rows, log):
    rng = np.random.default_rng([seed, n_p, PROTOS.index(proto)])
    W = Lc.pilot_beams(N, n_p, proto, rng)
    A = W.conj().T                                                     # Ep = 1
    mu, C = Htr.mean(0), np.cov(Htr.T)
    # ---- train learned models on mixed SNR
    t0 = time.time()
    # pre-generate the training pairs once: every train channel at every SNR, REPS noise draws (fresh noise per draw)
    g = np.random.default_rng([seed, 7, n_p, PROTOS.index(proto)])
    XS, CS, X1 = [], [], []
    for rep in range(REPS):
        for s in SNRS_DB:
            nv = 10 ** (-s / 10)
            Y = Htr @ A.T + np.sqrt(nv) * Lc.crandn(g, len(Htr), n_p)
            Ml = batch_lmmse(Y, A, nv, mu, C)[0]
            XS.append(Lc.c2r(Htr)), CS.append(cond_feats(Y, Ml, s, np.sqrt(1 + nv))), X1.append(Lc.c2r(Ml))
    XS = torch.tensor(np.concatenate(XS), dtype=torch.float32, device=DEV)
    CS = torch.tensor(np.concatenate(CS), dtype=torch.float32, device=DEV)
    X1 = torch.tensor(np.concatenate(X1), dtype=torch.float32, device=DEV)
    models = {}
    for kind in ["map", "gauss", "ddpm", "flow", "bridge", "bridge0"]:
        m = Learned(kind, 2 * n_p + 2 * N + 1, seed)
        m.net.to(DEV)
        opt = torch.optim.AdamW(m.net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        gt = torch.Generator(device="cpu").manual_seed(seed * 100 + n_p)
        for it in range(STEPS):
            idx = torch.randint(0, len(XS), (512,), generator=gt).to(DEV)
            loss = m.loss(XS[idx], CS[idx], X1[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        models[kind] = m
    log(f"  trained 6 models in {time.time() - t0:.0f}s")
    # ---- evaluate on dev and test at each SNR
    for split_name, Hs_ in (("dev", Hdv), ("test", Hte)):
        for sdb in SNRS_DB:
            snr, nv = 10 ** (sdb / 10), 10 ** (-sdb / 10)
            ge = np.random.default_rng([seed, 11, n_p, sdb + 100, split_name == "test"])
            Y = Hs_ @ A.T + np.sqrt(nv) * Lc.crandn(ge, len(Hs_), n_p)
            base = dict(split=split_name, proto=proto, n_p=n_p, snr_db=sdb, seed=seed)

            def add(method, rule, Hhat, beams, nfe=0, extra=None, secs=None):
                r = dict(base, method=method, rule=rule, nfe=nfe, nmse_db=nmse_db(Hhat, Hs_) if Hhat is not None else None,
                         **outcomes(Hs_, beams, snr), **(extra or {}))
                if secs is not None:
                    r["ms_per_sample"] = round(1000 * secs / len(Hs_), 4)
                rows.append(r)

            F = Lc.dft_codebook(N)
            # LS
            Hls = Y @ np.linalg.pinv(A).T
            add("ls", "mean", Hls, np.argmax(np.abs(Hls @ F.conj()), 1))
            # sweep (dft only)
            if proto == "dft":
                pil = np.round(np.linspace(0, N, n_p, endpoint=False)).astype(int)
                add("sweep", "pilot", None, pil[np.argmax(np.abs(Y), 1)])
            # LMMSE
            t1 = time.time()
            Ml, Pl = batch_lmmse(Y, A, nv, mu, C)
            secs = time.time() - t1
            add("lmmse", "mean", Ml, np.argmax(np.abs(Ml @ F.conj()), 1), secs=secs)
            Hs_l = analytic_samples(ge, Ml, Pl)
            be, bo, G = decide_from_samples(Hs_l, snr)
            gain_exact = np.real(np.einsum("nb,nm,mb->b", F.conj(), Pl, F))[None] + np.abs(Ml @ F.conj()) ** 2
            ext = dict(**coverage(G, Hs_), es=energy_score(Hs_l, Hs_))
            add("lmmse", "egain", Ml, np.argmax(gain_exact, 1), extra=ext)
            v = np.real(np.einsum("nb,nm,mb->b", F.conj(), Pl, F))[None]
            mu_b = np.abs(Ml @ F.conj()) ** 2
            p_out = ncx2.cdf(2 * (GAMMA / snr) / v, df=2, nc=2 * mu_b / v)
            add("lmmse", "outage", Ml, np.argmin(p_out - 1e-9 * gain_exact, 1), extra=ext)
            # OMP
            t1 = time.time()
            Ho = omp(Y, A, nv)
            add("omp", "mean", Ho, np.argmax(np.abs(Ho @ F.conj()), 1), secs=time.time() - t1)
            # GMM
            t1 = time.time()
            Wk, Mk, Pk = batch_gmm(Y, A, nv, gmm)
            secs = time.time() - t1
            Mg = np.einsum("nk,nki->ni", Wk, Mk)
            add("gmm", "mean", Mg, np.argmax(np.abs(Mg @ F.conj()), 1), secs=secs)
            vk = np.real(np.einsum("bi,kij,jb->kb", F.conj().T, Pk, F))          # (K, B)
            mk = np.abs(np.einsum("nki,ib->nkb", Mk, F.conj())) ** 2                # (n, K, B)
            eg = np.einsum("nk,nkb->nb", Wk, mk + vk[None])
            po = np.einsum("nk,nkb->nb", Wk, ncx2.cdf(2 * (GAMMA / snr) / vk[None], df=2, nc=2 * mk / vk[None]))
            Hs_g = analytic_samples(ge, Mk, Pk, Wk)
            _, _, G = decide_from_samples(Hs_g, snr)
            ext = dict(**coverage(G, Hs_), es=energy_score(Hs_g, Hs_))
            add("gmm", "egain", Mg, np.argmax(eg, 1), extra=ext)
            add("gmm", "outage", Mg, np.argmin(po - 1e-9 * eg, 1), extra=ext)
            # learned
            ysc = np.sqrt(1 + nv)
            c = torch.tensor(cond_feats(Y, Ml, sdb, ysc), device=DEV)
            x1 = torch.tensor(Lc.c2r(Ml), dtype=torch.float32, device=DEV)
            gen = torch.Generator(device=DEV).manual_seed(seed * 1000 + sdb + 50)
            for kind, m in models.items():
                for nfe in ([0] if kind in ("map", "gauss") else NFES):
                    t1 = time.time()
                    Xs = m.sample(c, x1, nfe, gen).cpu().double().numpy()
                    if DEV != "cpu":
                        torch.cuda.synchronize()
                    secs = time.time() - t1
                    Hs_m = Lc.r2c(Xs)
                    Hm = Hs_m.mean(0)
                    add(kind, "mean", Hm, np.argmax(np.abs(Hm @ F.conj()), 1), nfe=nfe, secs=secs)
                    if kind != "map":
                        be, bo, G = decide_from_samples(Hs_m, snr)
                        ext = dict(**coverage(G, Hs_), es=energy_score(Hs_m, Hs_))
                        add(kind, "egain", Hm, be, nfe=nfe, extra=ext)
                        add(kind, "outage", Hm, bo, nfe=nfe, extra=ext)
            log(f"  {split_name} {proto} Np={n_p} snr={sdb:+d} done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("out"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--nps", type=int, nargs="+", default=None)
    args = ap.parse_args()
    global STEPS, DEV
    DEV = args.device
    if args.steps:
        STEPS = args.steps
    nps = args.nps or NPS
    if args.quick:
        STEPS = 200
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "a1_log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    H, ep, n_ep = load_traces(args.trace)
    Htr, Hdv, Hte, info = split(H, ep, n_ep)
    sub = np.random.default_rng(123)
    Hdv = Hdv[np.sort(sub.permutation(len(Hdv))[:NEVAL])]
    Hte = Hte[np.sort(sub.permutation(len(Hte))[:NEVAL])]
    info.update(n_dev_eval=len(Hdv), n_test_eval=len(Hte), device=DEV)
    log(f"data {info}")
    rows = []
    res_path = out / "a1_results.json"
    for seed in args.seeds:
        t0 = time.time()
        gmm = fit_gmm(Htr, K=32, seed=seed)
        log(f"seed {seed}: gmm fit {time.time() - t0:.0f}s")
        for proto in PROTOS:
            for n_p in nps:
                log(f"seed {seed} {proto} Np={n_p}")
                run(Htr, Hdv, Hte, proto, n_p, seed, gmm, rows, log)
                res_path.write_text(json.dumps(dict(info=info, gamma=GAMMA, steps=STEPS, rows=rows)))
    res_path.write_text(json.dumps(dict(info=info, gamma=GAMMA, steps=STEPS, rows=rows)))


if __name__ == "__main__":
    main()
