"""Review #45 (A-T1): task-utility headroom in the network simulator.

Per-image, per-mode detector utility = TP_i(m) of the A5 detector (YOLO11s, conf >= 0.25, IoU >= 0.5, class-aware;
runs/uav_semcom/a6l1/fidelity/per_image_val.jsonl). Summed over the required workload and divided by the number of
ground-truth objects it is the full-workload recall (an image delivered late or dropped contributes 0 TP).

Utility sources (what a controller is allowed to know about the CURRENT image's task value):
  mean : mu_m = mean TP of mode m over the TRAIN pool images (causal, deployable: no per-image knowledge)
  gt   : TP_i(m) of this image (NON-DEPLOYABLE diagnostic oracle; uses ground truth)
  flip : TP_i(m) from the flipped-image re-measurement (G-L2 0.4; runs/uav_semcom/g_l2/flip for train/val images,
         runs/uav_semcom/at1/flip for test images). An oracle with a second, independent-ish detector measurement:
         select with one measurement, score with the other (cross-check against selection on detector noise).
         Modes without a flip measurement (the 3 caption-generative modes) fall back to the original measurement.
Channel information is the same as for every other controller (causal; the MPC's online-calibrated rate model).

Controllers (mode rules; the slot scheduler and movement are fixed by the caller):
  rv:SRC:LAM  argmax_m U(m) - LAM * b_hat(m) [Mbit]  (b_hat = probe-predicted payload; payload-aware rule, G-L2 RV)
  tb:SRC:TAU  backlog-budgeted rule (mode_heur.budget_mode) with value U(m) if predicted on time, else 0
  TaskMPC     the A-E8 calibrated selective MPC with the objective "sum of on-time TP" instead of the A5 utility
"""
import json
import math

import numpy as np

import a5_common as C
import a9_mpc as M
import policies as P

# the L1 measurement: identical TP to runs/uav_semcom/a5/fidelity on every common row, but complete for the 3
# caption-generative modes (the A5 file has them for 100 of 548 images only)
FID = C.REPO / "runs" / "uav_semcom" / "a6l1" / "fidelity" / "per_image_val.jsonl"
FLIP = [C.REPO / "runs" / "uav_semcom" / "g_l2" / "flip" / "per_image_flip.jsonl",
        C.REPO / "runs" / "uav_semcom" / "at1" / "flip" / "per_image_flip_test.jsonl"]
SOURCES = ("mean", "gt", "flip")
_T = {}


def _rows(paths):
    out = {}
    for p in paths:
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                out[(r["image"], r["mode"])] = r
    return out


def tables(table, pools):
    """TP matrices (table rows x modes) for 'gt' and 'flip', n_gt per row, mu (train mean). NaN = not measured."""
    key = id(table)
    if key not in _T:
        fid, flip = _rows([FID]), _rows(FLIP)
        n, nm = len(table.images), len(C.MODES)
        gt, fl, ngt = np.full((n, nm), np.nan), np.full((n, nm), np.nan), np.full(n, np.nan)
        for r, name in enumerate(table.images):
            o = fid.get((name, "original"))
            if o is not None:
                ngt[r] = o["n_gt"]
            for m, mode in enumerate(C.MODES):
                x = fid.get((name, mode))
                if x is not None:
                    gt[r, m] = x["tp"]
                y = flip.get((name, mode))
                fl[r, m] = y["tp"] if y is not None else gt[r, m]
        tr = np.asarray(pools["train"])
        mu = np.nanmean(gt[tr], 0)
        _T[key] = dict(gt=gt, flip=fl, ngt=ngt, mu=mu,
                       flip_measured=np.array([any((name, mode) in flip for name in table.images) for mode in C.MODES]))
    return _T[key]


class Utility:
    def __init__(self, src, table, pools):
        assert src in SOURCES, src
        self.src, self.t = src, tables(table, pools)

    def __call__(self, env, task):
        if self.src == "mean" or task.img < 0:
            return self.t["mu"]
        u = self.t[self.src][task.img]
        assert not np.isnan(u).any(), f"no {self.src} measurement for table row {task.img}"
        return u

    def mean(self):
        return self.t["mu"]


def rv_rule(util, lam):
    def f(env, k, task):
        u = util(env, task)
        b = env.mode_info(task)[0] / 1e6
        return int(np.argmax(u - lam * b))
    f.__name__ = f"rv:{util.src}:{lam:g}"
    return f


def tb_rule(util, tau):
    """mode_heur.budget_mode with task value: U(m) if predicted to end before the deadline, else 0 (dropped)."""
    def f(env, k, task):
        u, tab = env.uavs[k], env.tab
        rate = max(P.expected_rate(env, k) * P.channel_share(env), 1.0)
        tx_free = env.t
        for tk in u.queue:
            if tk is task:
                break
            if tk.mode >= 0:
                b = tk.rem_bits if tk.ready <= env.t else env.mode_info(tk)[0][tk.mode] - tk.sent_bits
                tx_free = max(tx_free, tk.ready) + max(b, 0.0) / rate
        b_hat = env.mode_info(task)[0]
        val = util(env, task)
        enc_free = max(env.t, u.enc_free)
        best, best_v, early, early_end = None, -math.inf, 0, math.inf
        for m in range(env.n_modes):
            end = max(tx_free, enc_free + tab.enc_s[m]) + b_hat[m] / rate
            if end < early_end:
                early, early_end = m, end
            if end - env.t <= tau:
                v = val[m] if end <= task.t_cap + task.dmax - tab.dec_s[m] else 0.0
                if v > best_v:
                    best, best_v = m, v
        return best if best is not None else early
    f.__name__ = f"tb:{util.src}:{tau:g}"
    return f


def rule(spec, table, pools):
    kind, src, x = spec.split(":")
    util = Utility(src, table, pools)
    return {"rv": rv_rule, "tb": tb_rule}[kind](util, float(x))


class TaskMPC(M.MPCMode):
    """A-E8 selective MPC (same rate models, samples, horizon, rollout, deadline handling) whose per-image value is
    the task utility of the chosen mode if predicted on time, else 0. The current image and the queued images whose
    probe has finished use util(env, task); not-yet-probed and future frames use the train mean mu."""

    def __init__(self, util, *a, **kw):
        super().__init__(*a, **kw)
        self.util = util
        self.__name__ = "task" + self.__name__

    def _value(self, env, tab, G, tx_free, ready, bits, q, task, m):
        end = self._finish(G, tx_free, ready, bits)
        latest = task.t_cap + task.dmax - tab.dec_s[m]
        miss = end > latest
        return np.where(miss, 0.0, q), np.where(miss, np.maximum(latest, np.maximum(tx_free, ready)), end)

    def _ctx(self, env, k, task, rates):
        ctx = super()._ctx(env, k, task, rates)
        ctx["q_hat"] = self.util(env, task)
        return ctx

    def _eval_mode(self, env, ctx, task, m):
        tab, G, ns = env.tab, ctx["G"], ctx["ns"]
        mu = self.util.mean()
        ready = np.full(ns, ctx["enc0"] + tab.enc_s[m])
        v, txf = self._value(env, tab, G, ctx["tx_free"], ready, ctx["b_hat"][m], ctx["q_hat"][m], task, m)
        enc = ready
        for tk in ctx["later"]:
            probed = tk.info_t <= env.t + 1e-9
            bh = env.mode_info(tk)[0] if probed else tab.mean_bits
            uh = self.util(env, tk) if probed else mu
            enc = np.maximum(enc, tk.info_t)
            uu, txf, enc = self._rollout(env, tab, G, txf, enc, tk, bh, uh, ctx.get("g", 1))
            v = v + uu
        for tk in ctx["fut"]:
            enc = np.maximum(enc, tk.t_cap) + env.probe_s()
            uu, txf, enc = self._rollout(env, tab, G, txf, enc, tk, tab.mean_bits, mu, ctx.get("g", 1))
            v = v + uu
        return v


def score(deliv, tab_tp, ngt_by_man, man_rows, ontime=True):
    """full-workload recall of a list of episodes of (manifest index, mode, ok, ontime) using TP table tab_tp
    (indexed by table row; man_rows maps manifest index -> table row)."""
    tp = gt = 0.0
    for ep in deliv:
        for mi, mode, ok, ot in ep:
            r = man_rows[mi]
            gt += ngt_by_man[r]
            if mode >= 0 and (ot if ontime else ok):
                tp += tab_tp[r, mode]
    return tp / gt if gt else float("nan")
