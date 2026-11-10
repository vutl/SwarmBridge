"""R1b-0a object-time metrics on the BS output (class-agnostic, IoU >= 0.5).

Per replay tick f: Hungarian match of BS output boxes (predicted to t_f) with GT objects (classes 1..10, score 1).
Outputs whose area lies > 50 % inside an ignored region (class 0 / 11) and are unmatched are discarded (not FP).
  object-time P / R / F1      : summed TP / FP / FN over all ticks
  IDF1                         : global bipartite GT-id <-> output-id assignment maximising co-occurring IoU>=0.5 ticks
  false-track seconds / min    : FP ticks x tick length, per minute of clip
  zero-match tracks            : output ids never matched, count and mean duration (s)
  first-detection latency      : per GT track visible >= min_vis ticks: (first matched tick - first visible tick) x
                                 tick length; never-matched tracks are censored and reported as a fraction
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from r1b_track import iou_matrix, match


def ioa_ignored(boxes, ign):
    if len(boxes) == 0 or len(ign) == 0:
        return np.zeros(len(boxes))
    b = np.asarray(boxes, float)[:, None, :]; g = np.asarray(ign, float)[None, :, :]
    iw = np.clip(np.minimum(b[..., 2], g[..., 2]) - np.maximum(b[..., 0], g[..., 0]), 0, None)
    ih = np.clip(np.minimum(b[..., 3], g[..., 3]) - np.maximum(b[..., 1], g[..., 1]), 0, None)
    area = np.maximum((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]), 1e-9)
    return (iw * ih / area).sum(1)


class Scorer:
    def __init__(self, tick_s=0.1, iou=0.5, min_vis=3):
        self.tick_s, self.iou, self.min_vis = tick_s, iou, min_vis
        self.tp = self.fp = self.fn = 0
        self.n_ticks = 0
        self.co = defaultdict(int)                 # (gid, pid) -> ticks with IoU >= thr
        self.n_gt = self.n_pred = 0
        self.first_vis, self.first_hit = {}, {}
        self.vis_count = defaultdict(int)
        self.pred_ticks, self.pred_hits = defaultdict(int), defaultdict(int)

    def add(self, f, gt, ign, out):
        """gt: list of (gid, box); ign: list of boxes; out: list of (pid, box)."""
        self.n_ticks += 1
        gb = [g[1] for g in gt]; ob = [o[1] for o in out]
        pairs, uo, _ = match(ob, gb, self.iou)          # pairs: (output idx, gt idx); uo: unmatched outputs
        if uo and len(ign):
            drop = set(np.array(uo)[ioa_ignored([ob[i] for i in uo], ign) > 0.5].tolist())
        else:
            drop = set()
        keep = [i for i in range(len(out)) if i not in drop]
        self.tp += len(pairs); self.fp += len(uo) - len(drop); self.fn += len(gt) - len(pairs)
        self.n_gt += len(gt); self.n_pred += len(keep)
        M = iou_matrix([ob[i] for i in keep], gb)
        for a, i in enumerate(keep):
            pid = out[i][0]
            self.pred_ticks[pid] += 1
            for j in np.nonzero(M[a] >= self.iou)[0] if M.size else []:
                self.co[(gt[j][0], pid)] += 1
        for i, j, _ in pairs:
            self.pred_hits[out[i][0]] += 1
            g = gt[j][0]
            if g not in self.first_hit:
                self.first_hit[g] = f
        for g, _ in gt:
            self.vis_count[g] += 1
            self.first_vis.setdefault(g, f)

    def result(self):
        P = self.tp / max(self.tp + self.fp, 1); R = self.tp / max(self.tp + self.fn, 1)
        F1 = 2 * P * R / max(P + R, 1e-12)
        gids = sorted({g for g, _ in self.co}); pids = sorted({p for _, p in self.co})
        idtp = 0
        if gids and pids:
            gi = {g: k for k, g in enumerate(gids)}; pi = {p: k for k, p in enumerate(pids)}
            C = np.zeros((len(gids), len(pids)))
            for (g, p), v in self.co.items():
                C[gi[g], pi[p]] = v
            r, c = linear_sum_assignment(-C)
            idtp = C[r, c].sum()
        IDF1 = 2 * idtp / max(self.n_gt + self.n_pred, 1)
        minutes = self.n_ticks * self.tick_s / 60
        zero = [p for p, n in self.pred_ticks.items() if self.pred_hits.get(p, 0) == 0]
        elig = [g for g, n in self.vis_count.items() if n >= self.min_vis]
        lat = [(self.first_hit[g] - self.first_vis[g]) * self.tick_s for g in elig if g in self.first_hit]
        return dict(P=100 * P, R=100 * R, F1=100 * F1, IDF1=100 * IDF1, tp=self.tp, fp=self.fp, fn=self.fn,
                    false_track_s_per_min=self.fp * self.tick_s / max(minutes, 1e-9),
                    zero_match_tracks=len(zero),
                    zero_match_mean_s=float(np.mean([self.pred_ticks[p] for p in zero]) * self.tick_s) if zero else 0.0,
                    first_det_median_s=float(np.median(lat)) if lat else None,
                    first_det_p90_s=float(np.percentile(lat, 90)) if lat else None,
                    never_detected_frac=1 - len(lat) / max(len(elig), 1), n_gt_tracks=len(elig), ticks=self.n_ticks)
