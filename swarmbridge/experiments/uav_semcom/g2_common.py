"""G-L2 shared code (reports/archive/r1-visdrone/uav-semcom-g-l2.md section 0): per-image task outcomes, matched-bitrate evaluation of
frame-level mode-choice policies, and the content-agnostic baselines F / MX / RV / CB.

Static frame-level evaluation (0.2): every image of a pool is sent once in the mode its policy chooses;
recall = sum TP / sum objects, bits = mean bits per frame. Every policy knows the true payload b_i(m) of every mode.

A policy FAMILY is a list of per-image choice vectors (one per knob value: lambda, cap, ...); its achievable curve is
the upper concave hull of its (mean bits, recall) points (two neighbouring members mixed at random reach any point of
a hull segment). curve_at(B) returns the recall at mean bits B together with the per-image expected TP / bits of that
mixture (for per-video / per-group decompositions).
"""
import json
from pathlib import Path

import numpy as np

import a5_common as C
import a6l1_task as TK

G2 = C.A5.parent / "g_l2"
BUDGET_MODES = ("mbt2018-mean_q3", "avif_50", "avif_80")
LAMS = np.concatenate([[0.0], np.logspace(-9, -2.5, 700), [np.inf]])


class Data:
    """Per-image arrays of one build ('val': train / val / test pools; 'testdev')."""

    def __init__(self, testdev=False):
        self.tab, self.pools = C.load_table(testdev)
        tt = TK.task_table(self.tab)
        self.modes = list(self.tab.modes)
        self.bits = self.tab.bits.copy()
        self.tp, self.fp, self.tps = tt.tp.copy(), tt.fp.copy(), tt.tps.copy()
        self.ngt, self.nsmall = tt.ngt.copy(), tt.nsmall.copy()
        man = {e["name"]: e for e in C.manifest()["images"]}
        self.seq = np.array([man[n]["seq"] if n in man else "" for n in self.tab.images])
        self.images = list(self.tab.images)

    def flip(self, path=G2 / "flip" / "per_image_flip.jsonl"):
        """Second measurement (TP / FP / TP small) from the flipped-image run; NaN where not measured."""
        tp2, fp2, tps2 = (np.full_like(self.tp, np.nan) for _ in range(3))
        idx = {n: i for i, n in enumerate(self.images)}
        for line in open(path):
            r = json.loads(line)
            if r["mode"] in self.modes and r["image"] in idx:
                i, j = idx[r["image"]], self.modes.index(r["mode"])
                tp2[i, j], fp2[i, j], tps2[i, j] = r["tp"], r["n_det"] - r["tp"], r["tp_small"]
        return tp2, fp2, tps2


def budgets(d):
    """B1..B3 = train-mean bits of the fixed modes BUDGET_MODES (rule 0.2)."""
    tr = d.pools["train"]
    return {m: float(d.bits[tr, d.modes.index(m)].mean()) for m in BUDGET_MODES}


def hull(pts):
    """Indices of the upper concave hull of points (x, y), sorted by x (Andrew monotone chain, upper part), after
    dropping points dominated in (x up, y down)."""
    o = sorted(range(len(pts)), key=lambda k: (pts[k][0], -pts[k][1]))
    up = []
    for k in o:
        while len(up) >= 2:
            (x1, y1), (x2, y2), (x3, y3) = pts[up[-2]], pts[up[-1]], pts[k]
            if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) >= 0:
                up.pop()
            else:
                break
        up.append(k)
    best, out = -np.inf, []
    for k in up:                     # keep the increasing part (more bits must buy more recall)
        if pts[k][1] > best + 1e-15:
            out.append(k)
            best = pts[k][1]
    return out


class Curve:
    """Achievable (bits, recall) curve of a family of choice vectors on pool I, scored with tp (and fp / tps)."""

    def __init__(self, choices, I, bits, tp, ngt, fp=None, tps=None, nsmall=None):
        self.I = np.asarray(I)
        self.ch = [np.asarray(c) for c in choices]
        self.b = [bits[self.I, c] for c in self.ch]
        self.t = [tp[self.I, c] for c in self.ch]
        self.f = [fp[self.I, c] for c in self.ch] if fp is not None else None
        self.s = [tps[self.I, c] for c in self.ch] if tps is not None else None
        self.N = float(ngt[self.I].sum())
        self.Ns = float(nsmall[self.I].sum()) if nsmall is not None else None
        self.pts = [(float(b.mean()), float(t.sum() / self.N)) for b, t in zip(self.b, self.t)]
        self.h = hull(self.pts)

    def at(self, B):
        """(recall, dict(per-image expected tp / bits / fp / tps, weights)) at mean bits B; NaN if B < cheapest."""
        h, P = self.h, self.pts
        if B < P[h[0]][0] - 1e-9:
            return np.nan, None
        if B >= P[h[-1]][0]:
            k = h[-1]
            return P[k][1], self._mix([(k, 1.0)])
        for a, b in zip(h[:-1], h[1:]):
            if P[a][0] - 1e-9 <= B <= P[b][0]:
                w = (B - P[a][0]) / max(P[b][0] - P[a][0], 1e-12)
                return (1 - w) * P[a][1] + w * P[b][1], self._mix([(a, 1 - w), (b, w)])
        k = h[0]
        return P[k][1], self._mix([(k, 1.0)])

    def _mix(self, ws):
        g = lambda arrs: sum(w * arrs[k] for k, w in ws) if arrs is not None else None
        return dict(tp=g(self.t), bits=g(self.b), fp=g(self.f), tps=g(self.s), ws=ws)


def lagrange_choices(value, bits, I, menu, lams=LAMS):
    """argmax_{m in menu} value[i, m] - lam * bits[i, m] for every lam (value: [I_all, M] array)."""
    I = np.asarray(I)
    menu = np.asarray(menu)
    V, Bt = value[I][:, menu], bits[I][:, menu]
    out = []
    for lam in lams:
        if np.isinf(lam):
            j = np.argmin(Bt, 1)
        else:
            j = np.argmax(V - lam * Bt, 1)
        out.append(menu[j])
    return out


def fixed_choices(I, menu):
    return [np.full(len(I), m) for m in menu]


def baselines(d, I, train, menu, tp, fp=None, tps=None, mu_tp=None):
    """Content-agnostic families on pool I: F (fixed), MX (fixed-mode mixtures, hull vertices chosen on train),
    RV (mu_m - lam b_i(m)), CB (per-image bit cap, best-mu mode under the cap)."""
    menu = list(menu)
    mu = mu_tp if mu_tp is not None else np.nanmean(tp[train], 0)
    kw = dict(bits=d.bits, tp=tp, ngt=d.ngt, fp=fp, tps=tps, nsmall=d.nsmall)
    fixed = Curve(fixed_choices(I, menu), I, **kw)
    ftr = Curve(fixed_choices(train, menu), train, **kw)
    vert = [menu[k] for k in ftr.h]
    mx = Curve(fixed_choices(I, vert), I, **kw)
    rv = Curve(lagrange_choices(np.broadcast_to(mu, d.bits.shape), d.bits, I, menu), I, **kw)
    caps = np.unique(np.quantile(d.bits[np.asarray(I)][:, menu], np.linspace(0, 1, 400)))
    order = sorted(menu, key=lambda m: -mu[m])
    cb = []
    Bi = d.bits[np.asarray(I)]
    for c in caps:
        ch = np.full(len(I), -1)
        for m in order:
            ok = (ch < 0) & (Bi[:, m] <= c)
            ch[ok] = m
        cheapest = np.array(menu)[np.argmin(Bi[:, menu], 1)]
        ch[ch < 0] = cheapest[ch < 0]
        cb.append(ch)
    cb = Curve(cb, I, **kw)
    return dict(F=fixed, MX=mx, RV=rv, CB=cb)


def fixed_at(curve, B):
    """F: best fixed mode with mean bits <= B (no mixing)."""
    best, k = -np.inf, None
    for j, (b, r) in enumerate(curve.pts):
        if b <= B + 1e-9 and r > best:
            best, k = r, j
    return (best, curve._mix([(k, 1.0)])) if k is not None else (np.nan, None)


def bl_at(base, B):
    """BL = max over F / MX / RV / CB at mean bits B -> (recall, name, mix)."""
    cands = {"F": fixed_at(base["F"], B)}
    for k in ("MX", "RV", "CB"):
        cands[k] = base[k].at(B)
    name = max((k for k in cands if np.isfinite(cands[k][0])), key=lambda k: cands[k][0])
    return cands[name][0], name, cands[name][1], {k: v[0] for k, v in cands.items()}
