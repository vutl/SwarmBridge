"""A6.0-L1 task reward (pre-registered: reports/archive/uav-semcom-a/uav-semcom-a6-l1.md section 0).

Task value of frame i sent in mode m (measured, a5_fidelity.py over every (image, mode) pair, run_a6l1_gpu.sh):
    v_i(m) = (TP_i(m) - BETA * FP_i(m)) / Vbar,   BETA = 0.5,  conf >= 0.25, IoU >= 0.5, class-aware,
    Vbar = mean over TRAIN images of (TP - BETA FP) of the ORIGINAL frame.
Per-image utility = xi_q * v_i(m) + xi_d * max(dmax - delay, 0) / dmax - 1[Qmin or deadline violated]: only the
quality term of A5 (xi_q * clip((PSNR - Qmin) / q_span)) is replaced; drops, uncaptured frames, energy, time, the
Qmin constraint, the observation and the action space are unchanged. The A5 (PSNR) utility of every delivered frame
is accumulated in parallel, so each episode reports both returns.
Labels enter the REWARD only (train pool when training, val for checkpoint selection, test / test-dev for
evaluation), never the observation. Heuristics may use v_hat_i(m) = a_m + b_m log(probe bits), fitted on train: a
function of the same probe measurement the PPO observation already carries (its per-mode predicted bits / PSNR are
linear in it).
"""
import json
import math

import numpy as np

import a5_common as C
import env_survey as SV

BETA = 0.5
FID = C.A5.parent / "a6l1" / "fidelity"
_CACHE = {}


class TaskTable:
    def __init__(self, table, train_idx, beta=BETA, fid_dir=FID):
        rows = {}
        for pool in ("val", "testdev"):
            p = fid_dir / f"per_image_{pool}.jsonl"
            if p.exists():
                for line in open(p):
                    r = json.loads(line)
                    rows[(r["image"], r["mode"])] = r
        I, Mn = len(table.images), len(table.modes)
        nan = lambda *s: np.full(s, np.nan)
        self.tp, self.fp, self.tps = nan(I, Mn), nan(I, Mn), nan(I, Mn)
        self.tp0, self.fp0, self.ngt, self.nsmall = nan(I), nan(I), nan(I), nan(I)
        for i, im in enumerate(table.images):
            o = rows.get((im, "original"))
            if o is not None:
                self.tp0[i], self.fp0[i] = o["tp"], o["n_det"] - o["tp"]
                self.ngt[i], self.nsmall[i] = o["n_gt"], o["n_small"]
            for j, m in enumerate(table.modes):
                r = rows.get((im, m))
                if r is not None:
                    self.tp[i, j], self.fp[i, j], self.tps[i, j] = r["tp"], r["n_det"] - r["tp"], r["tp_small"]
        self.beta = beta
        tr = np.asarray(train_idx)
        assert not np.isnan(self.tp[tr]).any() and not np.isnan(self.tp0[tr]).any(), "task fidelity missing (train)"
        self.vbar = float(np.mean(self.tp0[tr] - beta * self.fp0[tr]))
        self.v = (self.tp - beta * self.fp) / self.vbar
        self.probe_j = table.probe_j
        x = np.log(table.bits[tr, table.probe_j])
        X = np.stack([np.ones_like(x), x], 1)
        self.coef = np.linalg.lstsq(X, self.v[tr], rcond=None)[0]
        self.v_mean = self.v[tr].mean(0)
        self.bits = table.bits

    def vhat(self, img):
        return np.array([1.0, math.log(self.bits[img, self.probe_j])]) @ self.coef

    def covered(self, idx):
        idx = np.asarray(idx)
        return not (np.isnan(self.tp[idx]).any() or np.isnan(self.tp0[idx]).any())


def task_table(table):
    key = id(table)
    if key not in _CACHE:
        split = {e["name"]: e["split"] for e in C.manifest()["images"]}
        train = [i for i, im in enumerate(table.images) if split.get(im) == "train"]
        _CACHE[key] = (table, TaskTable(table, train))      # (keep the table alive: id() stays unique)
    return _CACHE[key][1]


class TaskSurveyEnv(SV.UavSurveyEnv):
    """A5 survey env with the task reward (see module docstring)."""

    def __init__(self, table, img_pool, cfg=None, seed=0):
        self.tt = task_table(table)
        super().__init__(table, img_pool, cfg, seed)

    def reset(self, seed=None):
        self.sum_task, self.sum_psnr = 0.0, 0.0
        return super().reset(seed)

    def task_value_hat(self, img):
        return self.tt.vhat(img)

    def task_value_mean(self):
        return self.tt.v_mean

    def _finish(self, task, t_end):
        c = self.c
        q = self.tab.psnr[task.img, task.mode]
        delay = t_end + self.tab.dec_s[task.mode] - task.t_cap
        u_psnr, ok = self.utility(task, q, delay)
        u = c["xi_q"] * float(self.tt.v[task.img, task.mode]) + c["xi_d"] * max(task.dmax - delay, 0) / task.dmax
        u_task = u - (0 if ok else 1)
        task.util, task.ok, task.delay, task.done = u_task, ok, delay, True
        self.finished.append(task)
        self.sum_task += u_task
        self.sum_psnr += u_psnr
        return u_task

    def metrics(self):
        m = super().metrics()
        tt = self.tt
        planned = self.task_img.ravel()
        ok = [tk for tk in self.finished if tk.ok]
        g = lambda a, idx: float(np.sum(a[idx])) if len(idx) else 0.0
        oi = np.array([tk.img for tk in ok], int)
        om = np.array([tk.mode for tk in ok], int)
        tp = float(tt.tp[oi, om].sum()) if len(ok) else 0.0
        fp = float(tt.fp[oi, om].sum()) if len(ok) else 0.0
        tps = float(tt.tps[oi, om].sum()) if len(ok) else 0.0
        ngt, nsm = g(tt.ngt, planned), g(tt.nsmall, planned)
        tp0, fp0 = g(tt.tp0, planned), g(tt.fp0, planned)
        m.update(
            psnr_minus_task=self.sum_psnr - self.sum_task,     # A5 (PSNR) return = task return + this
            recall=tp / max(ngt, 1.0), recall_small=tps / max(nsm, 1.0),
            fp=fp, fp_per_deliv=fp / max(len(ok), 1), precision=tp / max(tp + fp, 1.0),
            task_score=(tp - BETA * fp) / max(tp0 - BETA * fp0, 1.0),
            task_score_b0=tp / max(tp0, 1.0), task_score_b1=(tp - fp) / max(tp0 - fp0, 1.0),
            recall_rel_orig=tp / max(tp0, 1.0),
            bits_sent=float(sum(tk.sent_bits for tk in self.tasks)),
            bits_deliv=float(sum(tk.bits for tk in ok)),
            n_planned=int(len(planned)), n_ok=len(ok),
        )
        return m
