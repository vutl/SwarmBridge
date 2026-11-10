"""R1b-0a tracker: one deterministic class used for the BS tracker, the UAV local tracker and the A0/A1 virtual replica.

Track state: box (x1, y1, x2, y2) at reference tick t_ref, centre velocity (px / tick), cls, conf, hits, misses.
Association: Hungarian on IoU (class-agnostic), gate IoU >= iou_gate, two stages (high-conf dets first, then low).
Update: alpha-beta on centre (alpha on position, beta on velocity), size smoothed with alpha.
Observation kinds: 'full' (a whole frame was observed: unmatched tracks count a miss), 'partial' (tokens: no misses).
Death: misses >= max_miss (full observations only), or no observation for > max_age ticks, or centre outside frame.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    """a: (n, 4), b: (m, 4) xyxy -> (n, m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float)[:, None, :]; b = np.asarray(b, float)[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    ua = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def match(a, b, thr):
    """Hungarian on IoU; returns (pairs [(i, j, iou)], unmatched_a, unmatched_b)."""
    M = iou_matrix(a, b)
    if M.size == 0:
        return [], list(range(len(a))), list(range(len(b)))
    r, c = linear_sum_assignment(-M)
    pairs = [(i, j, M[i, j]) for i, j in zip(r, c) if M[i, j] >= thr]
    ma = {i for i, _, _ in pairs}; mb = {j for _, j, _ in pairs}
    return pairs, [i for i in range(len(a)) if i not in ma], [j for j in range(len(b)) if j not in mb]


class Track:
    __slots__ = ("tid", "box", "vel", "t_ref", "cls", "conf", "hits", "misses", "t_obs", "born")

    def __init__(self, tid, box, cls, conf, t):
        self.tid, self.box, self.vel, self.t_ref = tid, np.asarray(box, float), np.zeros(2), t
        self.cls, self.conf, self.hits, self.misses, self.t_obs, self.born = int(cls), float(conf), 1, 0, t, t

    def at(self, t):
        d = (t - self.t_ref) * self.vel
        return self.box + np.array([d[0], d[1], d[0], d[1]])


class Tracker:
    def __init__(self, W, H, iou_gate=0.3, conf_high=0.5, conf_new=0.4, conf_confirm=0.6, alpha=0.7, beta=0.3,
                 max_miss=2, max_age=50, id_mod=1024):
        self.W, self.H = W, H
        self.iou_gate, self.conf_high, self.conf_new, self.conf_confirm = iou_gate, conf_high, conf_new, conf_confirm
        self.alpha, self.beta, self.max_miss, self.max_age, self.id_mod = alpha, beta, max_miss, max_age, id_mod
        self.tracks = []
        self.next_id = 0

    def _new_id(self):
        live = {t.tid for t in self.tracks}
        for _ in range(self.id_mod):
            i = self.next_id
            self.next_id = (self.next_id + 1) % self.id_mod
            if i not in live:
                return i
        raise RuntimeError("no free track id")

    def confirmed(self, t):
        return [tr for tr in self.tracks if tr.hits >= 2 or tr.conf >= self.conf_confirm]

    def boxes(self, t, confirmed_only=True):
        trs = self.confirmed(t) if confirmed_only else self.tracks
        return [(tr.tid, tr.at(t), tr.cls, tr.conf) for tr in trs]

    def _update_track(self, tr, box, cls, conf, t):
        pred = tr.at(t)
        c_p = (pred[:2] + pred[2:]) / 2; c_o = (np.asarray(box[:2]) + np.asarray(box[2:])) / 2
        dt = max(t - tr.t_obs, 1)
        res = c_o - c_p
        c_new = c_p + self.alpha * res
        tr.vel = tr.vel + self.beta * res / dt
        wh = (1 - self.alpha) * (pred[2:] - pred[:2]) + self.alpha * (np.asarray(box[2:]) - np.asarray(box[:2]))
        tr.box = np.concatenate([c_new - wh / 2, c_new + wh / 2]); tr.t_ref = t
        tr.cls, tr.conf = int(cls), max(float(conf), 0.5 * tr.conf + 0.5 * float(conf))
        tr.hits += 1; tr.misses = 0; tr.t_obs = t

    def update(self, dets, t, kind="full"):
        """dets: list of (box xyxy, cls, conf). Returns {det index: tid} for matched/new tracks."""
        assign = {}
        hi = [i for i, d in enumerate(dets) if d[2] >= self.conf_high]
        lo = [i for i, d in enumerate(dets) if d[2] < self.conf_high]
        free = list(range(len(self.tracks)))
        for group in (hi, lo):
            if not group or not free:
                continue
            P = [self.tracks[k].at(t) for k in free]
            pairs, _, _ = match(P, [dets[i][0] for i in group], self.iou_gate)
            used = set()
            for a_, b_, _ in pairs:
                k = free[a_]; i = group[b_]
                self._update_track(self.tracks[k], *dets[i], t)
                assign[i] = self.tracks[k].tid; used.add(k)
            free = [k for k in free if k not in used]
        if kind == "full":
            for k in free:
                self.tracks[k].misses += 1
        for i in hi + lo:
            if i not in assign and dets[i][2] >= self.conf_new:
                tr = Track(self._new_id(), dets[i][0], dets[i][1], dets[i][2], t)
                self.tracks.append(tr); assign[i] = tr.tid
        self.prune(t)
        return assign

    def prune(self, t):
        keep = []
        for tr in self.tracks:
            b = tr.at(t); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if tr.misses >= self.max_miss or t - tr.t_obs > self.max_age or not (0 <= cx < self.W and 0 <= cy < self.H):
                continue
            keep.append(tr)
        removed = len(self.tracks) - len(keep)
        self.tracks = keep
        return removed
