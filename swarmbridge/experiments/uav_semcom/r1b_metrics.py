"""R1b-0a object-time metrics on the BS output.

PRIMARY = class-aware: a TP needs the same class (VisDrone 1..10) AND IoU >= 0.5. A right box with the wrong class
is one FP and one FN.  SECONDARY = class-agnostic (localisation only), keys suffixed `_ag`.
Per replay tick f, per mode: Hungarian match (per class in the class-aware mode). Outputs left unmatched whose area
lies > 50 % inside ignored regions (class 0 / 11) are discarded (not FP) - decided separately per mode.
  object-time P / R / F1   : summed TP / FP / FN over ticks
  IDF1                      : class-aware = per-class bipartite GT-id <-> output-id on co-occurring (same class,
                              IoU >= thr) ticks, pooled 2*sum IDTP / (n_gt + n_pred); equals py-motmetrics run per
                              class and pooled (checked in test_r1b). Agnostic = one bipartite over all.
  false-track s / min       : FP ticks x tick length per minute
  zero-match tracks         : output ids never matched; count and mean duration
  first-detection latency   : per GT track visible >= min_vis ticks (class-aware match), never-matched = censored
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


class _Mode:
    def __init__(self, aware):
        self.aware = aware
        self.tp = self.fp = self.fn = 0
        self.co = defaultdict(int)                 # (class or 0, gid, pid) -> ticks
        self.n_gt = self.n_pred = 0
        self.first_vis, self.first_hit = {}, {}
        self.vis_count = defaultdict(int)
        self.pred_ticks, self.pred_hits = defaultdict(int), defaultdict(int)

    def add(self, f, gt, ign, out, thr):
        groups = sorted({g[2] for g in gt} | {o[2] for o in out}) if self.aware else [None]
        pairs = []
        for c in groups:
            gi = [j for j, g in enumerate(gt) if c is None or g[2] == c]
            oi = [i for i, o in enumerate(out) if c is None or o[2] == c]
            p, _, _ = match([out[i][1] for i in oi], [gt[j][1] for j in gi], thr)
            pairs += [(oi[a], gi[b]) for a, b, _ in p]
        mo = {i for i, _ in pairs}
        uo = [i for i in range(len(out)) if i not in mo]
        drop = set()
        if uo and len(ign):
            drop = set(np.array(uo)[ioa_ignored([out[i][1] for i in uo], ign) > 0.5].tolist())
        keep = [i for i in range(len(out)) if i not in drop]
        self.tp += len(pairs); self.fp += len(uo) - len(drop); self.fn += len(gt) - len(pairs)
        self.n_gt += len(gt); self.n_pred += len(keep)
        M = iou_matrix([out[i][1] for i in keep], [g[1] for g in gt])
        for a, i in enumerate(keep):
            pid, _, pc = out[i]
            self.pred_ticks[pid] += 1
            if M.size:
                for j in np.nonzero(M[a] >= thr)[0]:
                    if not self.aware or gt[j][2] == pc:
                        self.co[(pc if self.aware else 0, gt[j][0], pid)] += 1
        for i, j in pairs:
            self.pred_hits[out[i][0]] += 1
            self.first_hit.setdefault(gt[j][0], f)
        for g in gt:
            self.vis_count[g[0]] += 1
            self.first_vis.setdefault(g[0], f)

    def idtp(self):
        tot = 0.0
        for c in sorted({k[0] for k in self.co}):
            items = {(g, p): v for (cc, g, p), v in self.co.items() if cc == c}
            gids = sorted({g for g, _ in items}); pids = sorted({p for _, p in items})
            gi = {g: k for k, g in enumerate(gids)}; pi = {p: k for k, p in enumerate(pids)}
            C = np.zeros((len(gids), len(pids)))
            for (g, p), v in items.items():
                C[gi[g], pi[p]] = v
            r, cc_ = linear_sum_assignment(-C)
            tot += C[r, cc_].sum()
        return tot

    def result(self, tick_s, n_ticks, min_vis):
        P = self.tp / max(self.tp + self.fp, 1); R = self.tp / max(self.tp + self.fn, 1)
        F1 = 2 * P * R / max(P + R, 1e-12)
        idtp = self.idtp()
        IDF1 = 2 * idtp / max(self.n_gt + self.n_pred, 1)
        minutes = n_ticks * tick_s / 60
        zero = [p for p in self.pred_ticks if self.pred_hits.get(p, 0) == 0]
        elig = [g for g, n in self.vis_count.items() if n >= min_vis]
        lat = [(self.first_hit[g] - self.first_vis[g]) * tick_s for g in elig if g in self.first_hit]
        return dict(P=100 * P, R=100 * R, F1=100 * F1, IDF1=100 * IDF1, tp=self.tp, fp=self.fp, fn=self.fn,
                    idtp=idtp, n_gt=self.n_gt, n_pred=self.n_pred,
                    false_track_s_per_min=self.fp * tick_s / max(minutes, 1e-9), zero_match_tracks=len(zero),
                    zero_match_mean_s=float(np.mean([self.pred_ticks[p] for p in zero]) * tick_s) if zero else 0.0,
                    first_det_median_s=float(np.median(lat)) if lat else None,
                    first_det_p90_s=float(np.percentile(lat, 90)) if lat else None,
                    never_detected_frac=1 - len(lat) / max(len(elig), 1), n_gt_tracks=len(elig))


class Scorer:
    def __init__(self, tick_s=0.1, iou=0.5, min_vis=3):
        self.tick_s, self.iou, self.min_vis = tick_s, iou, min_vis
        self.n_ticks = 0
        self.ca, self.ag = _Mode(True), _Mode(False)

    def add(self, f, gt, ign, out):
        """gt: list of (gid, box, cls); ign: list of boxes; out: list of (pid, box, cls)."""
        self.n_ticks += 1
        self.ca.add(f, gt, ign, out, self.iou)
        self.ag.add(f, gt, ign, out, self.iou)

    @property
    def fp(self):
        return self.ca.fp

    def result(self):
        r = self.ca.result(self.tick_s, self.n_ticks, self.min_vis)
        a = self.ag.result(self.tick_s, self.n_ticks, self.min_vis)
        r.update({k + "_ag": a[k] for k in ("P", "R", "F1", "IDF1", "tp", "fp", "fn", "first_det_median_s",
                                              "never_detected_frac")})
        r["ticks"] = self.n_ticks
        return r
