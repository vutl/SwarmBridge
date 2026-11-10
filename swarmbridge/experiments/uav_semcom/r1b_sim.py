"""R1b-0a event simulator (one UAV, actions SKIP / TOK / FRAME, motion variant M0) for the feedback-information ablation.

Causality: one event heap in ms. The UAV only sees (a) its own frame / detections at its decision time and (b) downlink
messages whose arrival event has fired. Only condition D reads the BS tracker directly (oracle, not deployable).
Compute is NOT free (protocol R1b-0a sec "compute"): 'serial' = one onboard worker doing detect -> track/policy ->
encode; a frame captured while the worker is busy is dropped (newest-frame-only). 'pipelined' (sensitivity) = detector
and encoder on separate resources. Packet ready time, uplink queueing, BS compute and the deadline all include compute.
"""
import heapq
import math
from dataclasses import dataclass, field

import numpy as np

import r1b_proto as P
from r1b_metrics import Scorer
from r1b_track import Tracker, iou_matrix, match

CONDS = ("A0", "A1", "B", "C0", "C1", "D")


@dataclass
class Clip:
    name: str
    W: int
    H: int
    gt: list          # per tick: list of (gid, box)
    ign: list         # per tick: list of boxes
    det_n: list       # per tick: list of (box, cls, conf)   UAV YOLO11n, conf >= 0.05
    det_s: list       # per tick: list of (box, cls, conf)   BS YOLO11s on the decoded FRAME image
    frame_bits: list  # per tick: AVIF payload bits of FRAME


@dataclass
class Cfg:
    cond: str = "C1"
    tick_ms: float = 100.0
    R_ul: float = 400e3            # bit/s
    DL_cap: float = 64e3           # bit/s
    d_ul: float = 10.0             # ms propagation
    d_dl: float = 100.0            # ms one-way downlink delay (scenario)
    p_ul: float = 0.01
    p_dl: float = 0.01
    ge: tuple = None               # Gilbert-Elliott downlink (p_gb, p_bg, loss_bad) or None
    p_dup: float = 0.0             # downlink duplicate prob
    jitter_dl: float = 0.0         # downlink extra delay U[0, jitter] ms (reordering)
    D_max: float = 300.0
    mtu_bits: int = 1200 * 8
    app_hdr: int = 96
    tok_hdr: int = 48
    tok_box: int = 52
    ack_bits: int = 32
    hint_hdr: int = 24
    hint_box: int = 52
    compute: str = "serial"
    c_det: float = 26.0            # ms YOLO11n (CPU proxy, measured)
    c_trk: float = 2.0             # ms local tracker + policy
    c_enc: float = 100.0           # ms AVIF encode (measured, locked)
    c_bs_frame: float = 20.0       # ms decode + YOLO11s (GPU)
    c_bs_tok: float = 1.0
    # policy (one class for every condition)
    th_T: float = 1.0
    th_F: float = 4.0
    N_max: int = 50
    w_new: float = 1.0
    w_lost: float = 1.0
    w_drift: float = 1.0
    w_unc: float = 0.0
    k_lost: int = 3
    bucket_s: float = 1.0
    tok_max: int = 64
    # protocol
    N_ref: int = 20
    W_hold: int = 10
    unacked: str = "conservative"  # A1 / C: content sent but not confirmed is assumed NOT at the BS
    frame_bits0: float = 2e5       # initial FRAME size estimate (train mean, locked)
    seed: int = 0
    trk: dict = field(default_factory=dict)


class Link:
    """FIFO serialising link with loss model; returns (arrival ms or None if lost, tx end ms)."""

    def __init__(self, rate, delay, p, rng, ge=None, p_dup=0.0, jitter=0.0):
        self.rate, self.delay, self.p, self.rng, self.ge = rate, delay, p, rng, ge
        self.p_dup, self.jitter = p_dup, jitter
        self.busy = 0.0
        self.bad = False

    def _lost(self, n_pk):
        if self.ge is None:
            return self.rng.random() < 1 - (1 - self.p) ** n_pk
        p_gb, p_bg, lb = self.ge
        lost = False
        for _ in range(n_pk):
            self.bad = (self.rng.random() < p_gb) if not self.bad else (self.rng.random() >= p_bg)
            lost |= self.rng.random() < (lb if self.bad else 0.0)
        return lost

    def send(self, now, bits, n_pk):
        start = max(now, self.busy)
        end = start + 1000.0 * bits / self.rate
        self.busy = end
        arr = end + self.delay + (self.rng.uniform(0, self.jitter) if self.jitter else 0.0)
        out = [] if self._lost(n_pk) else [arr]
        if out and self.p_dup and self.rng.random() < self.p_dup:
            out.append(arr + self.rng.uniform(0, max(self.jitter, 1.0) + 200.0))
        return out, end


def n_packets(bits, mtu):
    return max(1, math.ceil(bits / mtu))


def run_clip(clip, cfg, trace=False):
    rng = np.random.default_rng(cfg.seed)
    T = cfg.tick_ms
    W, H = clip.W, clip.H
    n = len(clip.gt)
    tk = dict(cfg.trk)
    bs = Tracker(W, H, **tk)
    local = Tracker(W, H, **tk)
    virt = Tracker(W, H, **tk)            # A0 / A1 replica of the BS tracker
    pub = P.Publisher(versioned=(cfg.cond == "C1"))
    rep = P.Replica(versioned=(cfg.cond == "C1"), w_hold=cfg.W_hold)
    ul = Link(cfg.R_ul, cfg.d_ul, cfg.p_ul, rng)
    dl = Link(cfg.DL_cap, cfg.d_dl, cfg.p_dl, rng, ge=cfg.ge, p_dup=cfg.p_dup, jitter=cfg.jitter_dl)
    ev, seq = [], [0]

    def push(t, kind, data=None):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, data))

    S = dict(ul_bits=0, dl_bits=0, n_frame=0, n_tok=0, n_skip=0, dropped_compute=0, ul_lost=0, ul_late=0,
             dl_msgs=0, dl_lost=0, snapshots=0, desync_ticks=0, uav_ms=0.0, bs_ms=0.0, belief_f1=[])
    scorer = Scorer(tick_s=T / 1000)
    bucket = [cfg.R_ul * cfg.bucket_s, 0.0]          # tokens, last update ms
    worker = dict(busy=0.0, enc_busy=0.0)
    bs_busy = [0.0]
    bs_state = dict(dirty=False, acks=[], last_proc=0, hint_pending=False, snap_due=False, snap_req=False,
                    last_snap_tick=0, changed=set())
    sent_obs = {}                                      # capture tick -> (obs, kind) for A0/A1
    acked_upto = [-1]
    hint = dict(boxes=[], fp=-1, follow={})            # B
    local_hist = {}                                    # tick -> {tid: box}
    miss = {}                                          # \hat T id -> consecutive unmatched ticks
    est_frame = [float(cfg.frame_bits0)]               # train-set mean, never this clip's own sizes
    last_frame = [-10 ** 9]
    actions = []

    # ---------------- BS side
    def canonical(t_state):
        """Tracks not observed since the last publication keep the exact integer prediction of the published
        state (no re-quantisation drift, no spurious update ops); observed ones are re-quantised."""
        prev = pub.current_at(t_state)
        st = {}
        for tr in bs.confirmed(t_state):
            if tr.tid in prev and tr.tid not in bs_state["changed"]:
                st[tr.tid] = prev[tr.tid]
            else:
                st[tr.tid] = P.quantise(tr.at(t_state), tr.vel, tr.cls, tr.conf, W, H)
        return st

    def try_send(now):
        if dl.busy > now + 1e-9 or cfg.cond in ("A0", "D"):
            return
        bits, payload = None, None
        f_now = int(now // T)
        if cfg.cond == "A1":
            if bs_state["acks"]:
                payload = ("ack", list(bs_state["acks"])); bits = cfg.ack_bits * len(bs_state["acks"])
                bs_state["acks"] = []
        elif cfg.cond == "B":
            if bs_state["hint_pending"]:
                fp = bs_state["last_proc"]
                boxes = [b for _, b, _, _ in bs.boxes(fp)]
                payload = ("hint", fp, boxes); bits = cfg.hint_hdr + cfg.hint_box * len(boxes)
                bs_state["hint_pending"] = False
        else:  # C0 / C1
            snap = bs_state["snap_due"] or bs_state["snap_req"]
            if bs_state["dirty"] or snap:
                t_state = f_now
                m = pub.make(canonical(t_state), t_state, bs_state["last_proc"], snapshot=snap)
                bs_state["dirty"] = False; bs_state["changed"] = set()
                if snap:
                    bs_state["snap_due"] = bs_state["snap_req"] = False; bs_state["last_snap_tick"] = f_now
                    S["snapshots"] += 1
                if m is not None:
                    payload = ("state", m); bits = m.bits
        if payload is None:
            return
        tot = bits + cfg.app_hdr * n_packets(bits, cfg.mtu_bits)
        arrs, end = dl.send(now, tot, n_packets(tot, cfg.mtu_bits))
        S["dl_bits"] += tot; S["dl_msgs"] += 1
        if not arrs:
            S["dl_lost"] += 1
        for a in arrs:
            push(a, "DL_RX", payload)
        push(end, "DL_IDLE")

    # ---------------- UAV belief \hat T_f (list of (id, box))
    def belief(f, now):
        c = cfg.cond
        if c == "D":
            return [(tid, b) for tid, b, _, _ in bs.boxes(f)]
        if c in ("A0", "A1"):
            return [(tid, b) for tid, b, _, _ in virt.boxes(f)]
        if c == "B":
            out = []
            for k, b in enumerate(hint["boxes"]):
                lt = hint["follow"].get(k)
                tr = next((x for x in local.tracks if x.tid == lt), None) if lt is not None else None
                out.append((("h", hint["fp"], k), tr.at(f) if tr is not None else np.asarray(b)))
            return out
        st = rep.view(f)
        return [(tid, np.array(P.dequant_box(v, W, H))) for tid, v in st.items()]

    def apply_dl(payload, now):
        kind = payload[0]
        if kind == "ack":
            for c_ in sorted(payload[1]):
                if c_ > acked_upto[0] and c_ in sent_obs:
                    obs, k = sent_obs.pop(c_)
                    virt.update(obs, c_, kind=k)
                    acked_upto[0] = c_
        elif kind == "hint":
            _, fp, boxes = payload
            if fp < hint["fp"]:
                return
            hist = local_hist.get(fp, {})
            ids = list(hist); lb = [hist[i] for i in ids]
            pairs, _, _ = match(boxes, lb, 0.3)
            hint.update(boxes=boxes, fp=fp, follow={i: ids[j] for i, j, _ in pairs})
        else:
            rep.receive(payload[1], int(now // T))

    # ---------------- UAV decision at tick f
    def decide(f, now):
        dets = clip.det_n[f]
        local.update([d for d in dets if d[2] >= 0.05], f, kind="full")
        local_hist[f] = {tr.tid: tr.at(f) for tr in local.tracks}
        local_hist.pop(f - 100, None)
        L = [(tr.tid, tr.at(f), tr.cls, tr.conf) for tr in local.confirmed(f) if tr.t_obs == f]
        if cfg.cond in ("C0", "C1"):
            rep.tick(f)
            if rep.desync:
                S["desync_ticks"] += 1
        if cfg.cond in ("A0", "A1"):
            virt.prune(f)
        Th = belief(f, now)
        lb = [x[1] for x in L]; tb = [x[1] for x in Th]
        pairs, uL, uT = match(lb, tb, 0.3)
        new = len(uL); drift = sum(1 for _, _, v in pairs if v < 0.5)
        mt = {Th[j][0] for _, j, _ in pairs}
        for i_, _ in Th:
            miss[i_] = 0 if i_ in mt else miss.get(i_, 0) + 1
        live = {i_ for i_, _ in Th}
        for k in [k for k in miss if k not in live]:
            del miss[k]
        lost = sum(1 for i_, _ in Th if miss[i_] >= cfg.k_lost)
        unc = 0.0
        low = [d for d in dets if 0.05 <= d[2] < 0.25]
        if low and cfg.w_unc:
            M = iou_matrix([d[0] for d in low], tb) if tb else np.zeros((len(low), 0))
            for i_, d in enumerate(low):
                if M.shape[1] == 0 or M[i_].max() < 0.3:
                    b = d[0]; unc += (b[2] - b[0]) * (b[3] - b[1]) / (W * H) * 100
        g = cfg.w_new * new + cfg.w_lost * lost + cfg.w_drift * drift + cfg.w_unc * unc
        # diagnostics: belief vs true BS state
        true = [b for _, b, _, _ in bs.boxes(f)]
        if true or tb:
            pp, _, _ = match(tb, true, 0.5)
            S["belief_f1"].append(2 * len(pp) / max(len(tb) + len(true), 1))
        # token bucket
        bucket[0] = min(cfg.R_ul * cfg.bucket_s, bucket[0] + cfg.R_ul * (now - bucket[1]) / 1000.0); bucket[1] = now
        sync = P.UL_SYNC if cfg.cond == "C1" else 0
        dev = [L[i] for i in uL] + [L[i] for i, _, v in pairs if v < 0.5]
        dev = dev[:cfg.tok_max]
        tok_bits = cfg.tok_hdr + cfg.tok_box * len(dev) + sync
        fr_bits = est_frame[0] + sync
        forced = f - last_frame[0] >= cfg.N_max

        def meets_deadline(bits, ready):
            # local information only: own uplink backlog and nominal rate
            tot = bits + cfg.app_hdr * n_packets(bits, cfg.mtu_bits)
            return max(ready, ul.busy) + 1000.0 * tot / cfg.R_ul + cfg.d_ul - f * T <= cfg.D_max

        enc_ready = now + cfg.c_enc if cfg.compute == "serial" else max(now, worker["enc_busy"]) + cfg.c_enc
        act = "SKIP"
        if (g >= cfg.th_F or forced) and bucket[0] >= fr_bits and meets_deadline(fr_bits, enc_ready):
            act = "FRAME"
        elif g >= cfg.th_T and dev and bucket[0] >= tok_bits and meets_deadline(tok_bits, now):
            act = "TOK"
        actions.append(act)
        if act == "SKIP":
            S["n_skip"] += 1
            return now
        if act == "FRAME":
            S["n_frame"] += 1; last_frame[0] = f
            pay = clip.frame_bits[f] + sync
            est_frame[0] = 0.8 * est_frame[0] + 0.2 * clip.frame_bits[f]
            if cfg.compute == "serial":
                ready = now + cfg.c_enc
            else:
                ready = max(now, worker["enc_busy"]) + cfg.c_enc; worker["enc_busy"] = ready
            obs_bs = ("frame", clip.det_s[f])
            obs_virt = ([d for d in dets if d[2] >= 0.05], "full")
        else:
            S["n_tok"] += 1
            pay = tok_bits
            ready = now
            obs_bs = ("tok", [(x[1], x[2], x[3]) for x in dev])
            obs_virt = ([(x[1], x[2], x[3]) for x in dev], "partial")
        tot = pay + cfg.app_hdr * n_packets(pay, cfg.mtu_bits)
        bucket[0] -= tot
        arrs, _ = ul.send(ready, tot, n_packets(tot, cfg.mtu_bits))
        S["ul_bits"] += tot
        hdr = rep.header() if cfg.cond == "C1" else None
        if not arrs:
            S["ul_lost"] += 1
        else:
            push(arrs[0], "BS_RX", (f, obs_bs, hdr))
        if cfg.cond == "A0":
            virt.update(obs_virt[0], f, kind=obs_virt[1])
        elif cfg.cond == "A1":
            sent_obs[f] = obs_virt
        return ready

    # ---------------- main loop
    for f in range(n):
        push(f * T, "TICK", f)
    while ev:
        now, _, kind, data = heapq.heappop(ev)
        if kind == "TICK":
            f = data
            bs.prune(f)
            if cfg.cond in ("C0", "C1"):
                if pub.state and set(pub.current_at(f)) != {t.tid for t in bs.confirmed(f)}:
                    bs_state["dirty"] = True
                if f - bs_state["last_snap_tick"] >= cfg.N_ref:
                    bs_state["snap_due"] = True
                try_send(now)
            scorer.add(f, clip.gt[f], clip.ign[f], [((tr.tid, tr.born), tr.at(f)) for tr in bs.confirmed(f)])
            if cfg.compute == "serial" and worker["busy"] > now + 1e-9:
                S["dropped_compute"] += 1; actions.append("DROP")
                continue
            t_dec = now + cfg.c_det + cfg.c_trk
            worker["busy"] = t_dec
            push(t_dec, "DECIDE", f)
        elif kind == "DECIDE":
            f = data
            end = decide(f, now)
            S["uav_ms"] += cfg.c_det + cfg.c_trk + (end - now)
            if cfg.compute == "serial":
                worker["busy"] = end
        elif kind == "BS_RX":
            f_cap, obs, hdr = data
            if now - f_cap * T > cfg.D_max:
                S["ul_late"] += 1
                continue
            c = cfg.c_bs_frame if obs[0] == "frame" else cfg.c_bs_tok
            start = max(now, bs_busy[0]); bs_busy[0] = start + c; S["bs_ms"] += c
            push(start + c, "BS_PROC", (f_cap, obs, hdr))
        elif kind == "BS_PROC":
            f_cap, obs, hdr = data
            asg = bs.update(obs[1], f_cap, kind="full" if obs[0] == "frame" else "partial")
            bs_state["changed"] |= set(asg.values())
            bs_state["last_proc"] = max(bs_state["last_proc"], f_cap)
            bs_state["dirty"] = True; bs_state["hint_pending"] = True
            bs_state["acks"].append(f_cap)
            if hdr is not None and pub.check_uplink(*hdr):
                bs_state["snap_req"] = True
            try_send(now)
        elif kind == "DL_IDLE":
            try_send(now)
        elif kind == "DL_RX":
            apply_dl(data, now)
    R = scorer.result()
    dur = n * T / 1000
    R.update(clip=clip.name, cond=cfg.cond, ul_kbps=S["ul_bits"] / dur / 1e3, dl_kbps=S["dl_bits"] / dur / 1e3,
             n_frame=S["n_frame"], n_tok=S["n_tok"], n_skip=S["n_skip"], dropped_compute=S["dropped_compute"],
             ul_lost=S["ul_lost"], ul_late=S["ul_late"], dl_msgs=S["dl_msgs"], dl_lost=S["dl_lost"],
             snapshots=S["snapshots"], desync_ticks=S["desync_ticks"], uav_ms_per_tick=S["uav_ms"] / n,
             bs_ms_per_tick=S["bs_ms"] / n, belief_f1=100 * float(np.mean(S["belief_f1"])) if S["belief_f1"] else None)
    if trace:
        R["actions"] = actions
    return R
