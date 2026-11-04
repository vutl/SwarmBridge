"""B, learned fusion: a small attention network chooses the beam from own pilots + map prior + other UAVs' messages.

Same traces, split, measurement and message content as b_coop.py (messages = other UAVs' own LMMSE estimates under their
map prior, within R m, age <= MAX_AGE s, strictly earlier than t). Per decision point the network sees:
  own   : y / sqrt(1 + noise_var) (2 Np), snr_db / 10, map beam profile log10(diag(F^H C_map F)) (16), has_map flag
  msgs  : up to M = 16 most recent messages: log10 beam profile of the sender's estimate (16), relative position / R (3),
          age / MAX_AGE (1); masked when absent
Output: 16 beam logits; trained to maximise expected rate sum_b softmax_b * rate_b (rate under the true channel) on the
TRAIN episodes (map prior for a train point excludes channels from its own episode, so no leakage), model selected on
dev by rate, read on test. Variants: 'attn' (with messages) and 'attn_nomsg' (messages always masked) to isolate the
value of the messages. Reference rows from b_coop.py are joined by the summary.
Output OUT/b_attn.json. Usage: python b_attn.py TRACE_DIR OUT [--seeds 0 1 2] [--device cuda]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

import linear as Lc
from a1_bench import batch_lmmse, N, GAMMA, FLOOR_DB
from b_coop import load, tnorm, cov_of, R, MIN_NB, MAX_AGE, DT

M, STRIDE, STEPS = 16, 2, 3000
SNRS_DB = [-5, 5, 15]
PROTOS = ["dft", "random"]
NP = 4
F = Lc.dft_codebook(N)


def profile(X):
    """log10 beam power profile, normalised by its sum (direction structure only)."""
    g = np.abs(X @ F.conj()) ** 2
    return np.log10(g / np.maximum(g.sum(-1, keepdims=True), 1e-30) + 1e-6)


class Net(nn.Module):
    def __init__(self, d_own, d_msg=20, w=128):
        super().__init__()
        self.own = nn.Sequential(nn.Linear(d_own, w), nn.SiLU(), nn.Linear(w, w))
        self.msg = nn.Sequential(nn.Linear(d_msg, w), nn.SiLU(), nn.Linear(w, w))
        self.att = nn.MultiheadAttention(w, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * w, w), nn.SiLU(), nn.Linear(w, N))

    def forward(self, own, msg, mask):
        q = self.own(own)[:, None]
        k = self.msg(msg)
        allmask = mask.all(1)
        mask = mask.clone()
        mask[allmask, 0] = False                                  # avoid NaN for rows without messages
        a, _ = self.att(q, k, k, key_padding_mask=mask)
        a = torch.where(allmask[:, None, None], torch.zeros_like(a), a)
        return self.head(torch.cat([q[:, 0], a[:, 0]], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("out"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu"); ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    steps = args.steps
    dev = args.device
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eps = load(args.trace)
    n_ep = len(eps)
    a_, b_ = int(0.6 * n_ep), int(0.8 * n_ep)
    tr = [e for e in eps if e["ep"] < a_]
    allp = np.concatenate([np.sum(np.abs(e["h"]) ** 2, -1).ravel() for e in tr])
    med = np.median(allp[allp > 0])
    scale = np.sqrt(med / N)
    for e in eps:
        e["ok"] = np.sum(np.abs(e["h"]) ** 2, -1) > med * 10 ** (-FLOOR_DB / 10)
        e["h"] = e["h"] / scale
    Ptr = np.concatenate([e["pos"][e["ok"]] for e in tr])
    Htr = np.concatenate([e["h"][e["ok"]] for e in tr])
    Etr = np.concatenate([np.full(e["ok"].sum(), e["ep"]) for e in tr])
    tree = cKDTree(Ptr)
    C_glob = tnorm(np.cov(Htr.T))
    t0 = time.time()
    for e in eps:                                         # map prior per point (train points exclude own episode)
        U, T, _ = e["h"].shape
        e["cm"] = np.empty((U, T, N, N), complex)
        e["has_map"] = np.zeros((U, T), bool)
        for u in range(U):
            for t in range(T):
                nb = tree.query_ball_point(e["pos"][u, t], R)
                nb = [i for i in nb if Etr[i] != e["ep"]]
                if len(nb) >= MIN_NB:
                    e["cm"][u, t], e["has_map"][u, t] = cov_of(Htr[nb]), True
                else:
                    e["cm"][u, t] = C_glob
        e["mprof"] = np.log10(np.maximum(np.real(np.einsum("bi,utij,jb->utb", F.conj().T, e["cm"], F)), 1e-30) / N + 1e-6)
    print(f"map priors {time.time() - t0:.0f}s", flush=True)
    rows = []
    for seed in args.seeds:
        for proto in PROTOS:
            rng = np.random.default_rng([seed, NP, PROTOS.index(proto), 99])
            W = Lc.pilot_beams(N, NP, proto, rng)
            A = W.conj().T
            data = {s: {} for s in ("train", "dev", "test")}
            for sdb in SNRS_DB:
                snr, nv = 10 ** (sdb / 10), 10 ** (-sdb / 10)
                feats = {s: dict(own=[], msg=[], mask=[], rate=[], hm=[], hc=[], kind=[]) for s in data}
                for e in eps:
                    sp = "train" if e["ep"] < a_ else ("dev" if e["ep"] < b_ else "test")
                    g = np.random.default_rng([seed, e["ep"], sdb + 100, NP, PROTOS.index(proto)])
                    U, T, _ = e["h"].shape
                    y = e["h"] @ A.T + np.sqrt(nv) * Lc.crandn(g, U, T, NP)
                    CAh = e["cm"] @ A.conj().T                                         # (U, T, N, Np)
                    Sg = A @ CAh + nv * np.eye(NP)                                      # (U, T, Np, Np)
                    est = np.einsum("utnk,utk->utn", CAh, np.linalg.solve(Sg, y[..., None])[..., 0])
                    eprof = profile(est)
                    rate = Lc.rate(np.abs(e["h"] @ F.conj()) ** 2, snr)            # (U, T, B)
                    for u in range(U):
                        for t in range(0, T, STRIDE):
                            if not e["ok"][u, t]:
                                continue
                            p = e["pos"][u, t]
                            t_lo = max(0, t - int(MAX_AGE / DT))
                            cand = []
                            for v in range(U):
                                if v == u or t_lo >= t:
                                    continue
                                d = np.linalg.norm(e["pos"][v, t_lo:t] - p, axis=1)
                                for k in np.where(d <= R)[0]:
                                    cand.append((t - (t_lo + k), v, t_lo + k))
                            cand.sort()
                            msg = np.zeros((M, 20), np.float32)
                            mask = np.ones(M, bool)
                            for i, (age, v, tt) in enumerate(cand[:M]):
                                msg[i, :16] = eprof[v, tt]
                                msg[i, 16:19] = (e["pos"][v, tt] - p) / R
                                msg[i, 19] = age * DT / MAX_AGE
                                mask[i] = False
                            own = np.concatenate([Lc.c2r(y[u, t]) / np.sqrt(1 + nv), [sdb / 10], e["mprof"][u, t],
                                                  [float(e["has_map"][u, t])]]).astype(np.float32)
                            f = feats[sp]
                            f["own"].append(own); f["msg"].append(msg); f["mask"].append(mask)
                            f["rate"].append(rate[u, t].astype(np.float32)); f["hm"].append(e["has_map"][u, t])
                            f["hc"].append(bool(cand)); f["kind"].append(e["kind"])
                for s in data:
                    data[s][sdb] = {k: np.array(v) for k, v in feats[s].items()}
                print(f"  features seed {seed} {proto} snr {sdb:+d}: train {len(data['train'][sdb]['own'])} {time.time() - t0:.0f}s", flush=True)
            for variant in ("attn", "attn_nomsg"):
                torch.manual_seed(seed)
                d_own = 2 * NP + 1 + 16 + 1
                net = Net(d_own).to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
                cat = lambda s, k: np.concatenate([data[s][sdb][k] for sdb in SNRS_DB])
                Tr = {k: torch.tensor(cat("train", k), device=dev) for k in ("own", "msg", "mask", "rate")}
                if variant == "attn_nomsg":
                    Tr["mask"] = torch.ones_like(Tr["mask"])
                gt = torch.Generator().manual_seed(seed)
                best, best_state = -1e9, None
                for it in range(steps):
                    idx = torch.randint(0, len(Tr["own"]), (512,), generator=gt).to(dev)
                    logit = net(Tr["own"][idx], Tr["msg"][idx], Tr["mask"][idx])
                    loss = -(torch.softmax(logit, 1) * Tr["rate"][idx]).sum(1).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                    if (it + 1) % max(steps // 6, 1) == 0:
                        with torch.no_grad():
                            r = 0.0
                            for sdb in SNRS_DB:
                                D = data["dev"][sdb]
                                mk = torch.tensor(D["mask"], device=dev)
                                if variant == "attn_nomsg":
                                    mk = torch.ones_like(mk)
                                b = net(torch.tensor(D["own"], device=dev), torch.tensor(D["msg"], device=dev), mk).argmax(1).cpu().numpy()
                                r += D["rate"][np.arange(len(b)), b].mean()
                        if r > best:
                            best, best_state = r, {k: v.clone() for k, v in net.state_dict().items()}
                net.load_state_dict(best_state)
                with torch.no_grad():
                    for sdb in SNRS_DB:
                        for sp in ("dev", "test"):
                            D = data[sp][sdb]
                            mk = torch.tensor(D["mask"], device=dev)
                            if variant == "attn_nomsg":
                                mk = torch.ones_like(mk)
                            b = net(torch.tensor(D["own"], device=dev), torch.tensor(D["msg"], device=dev), mk).argmax(1).cpu().numpy()
                            rb = D["rate"][np.arange(len(b)), b]
                            snr = 10 ** (sdb / 10)
                            gain = (2 ** rb - 1) / snr
                            outg = snr * gain < GAMMA
                            for kind in ("convoy", "indep"):
                                for hm in (True, False):
                                    for hc in (True, False):
                                        sel = (D["kind"] == kind) & (D["hm"] == hm) & (D["hc"] == hc)
                                        if sel.any():
                                            rows.append(dict(seed=seed, proto=proto, n_p=NP, snr_db=sdb, split=sp, kind=kind,
                                                             has_map=bool(hm), has_coop=bool(hc), method=variant, rule="learned",
                                                             n=int(sel.sum()), rate_sum=float(rb[sel].sum()),
                                                             outage_sum=float(outg[sel].sum()), regret_sum=0.0, msgs_sum=0))
                print(f"seed {seed} {proto} {variant} trained (dev rate sum {best:.3f})", flush=True)
            (out / "b_attn.json").write_text(json.dumps(dict(steps=steps, M=M, rows=rows)))


if __name__ == "__main__":
    main()
