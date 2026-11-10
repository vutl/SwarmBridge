"""R1b-0a event simulator (one UAV, actions SKIP / TOK / FRAME, motion variant M0) for the feedback-information ablation.

Causality: one event heap in ms. The UAV only sees (a) its own frame / detections at its decision time and (b) downlink
messages whose arrival event has fired, decoded from their wire bytes (r1b_proto). Only D reads the BS tracker.
Compute is NOT free: 'serial' (primary) = one onboard worker: detect -> track/policy -> encode; a frame captured while
the worker is busy is dropped (newest-frame-only). 'pipelined' (sensitivity) = detector and encoder on separate
resources. Packet ready time, uplink queueing, BS compute and the deadline all include compute.
Channel randomness is EXOGENOUS and PAIRED across conditions: every draw is a keyed hash of (seed, stream, tick,
message index, packet index), so the fate of the uplink packet captured at tick f does not depend on how many random
numbers other traffic consumed; Gilbert-Elliott states are a per-10 ms slot chain independent of traffic.
"""
import hashlib
import heapq
import math
import struct
from dataclasses import dataclass, field

import numpy as np

import r1b_proto as P
from r1b_metrics import Scorer
from r1b_track import Tracker, iou_matrix, match

CONDS = ("A0", "A1", "B", "Bp", "C0", "C1", "D")
UL_LOSS, DL_LOSS, DL_JIT, DL_DUP, DL_DUPD, GE_CHAIN, GE_LOSS = range(1, 8)


def ku(*key):
    """Deterministic U[0, 1) from an integer key."""
    h = hashlib.blake2b(struct.pack(f"<{len(key)}q", *key), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0 ** 64


@dataclass
class Clip:
    name: str
    W: int
    H: int
    gt: list          # per tick: list of (gid, box, cls)
    ign: list         # per tick: list of boxes
    det_n: list       # per tick: list of (box, cls, conf)   UAV YOLO11n, conf >= 0.05
    det_s: list       # per tick: list of (box, cls, conf)   BS YOLO11s on the decoded FRAME image
    frame_bits: list  # per tick: AVIF payload bits of FRAME


@dataclass
class Cfg:
    cond: str = "C1"
    tick_ms: float = 100.0
    R_ul: float = 1.75e6
    DL_cap: float = 64e3
    d_ul: float = 10.0
    d_dl: float = 100.0
    p_ul: float = 0.01
    p_dl: float = 0.01
    ge: tuple = None               # downlink Gilbert-Elliott (p_gb, p_bg, loss_bad) per 10 ms slot, or None
    p_dup: float = 0.0
    jitter_dl: float = 0.0
    D_max: float = 300.0
    mtu_bits: int = 1200 * 8
    app_hdr: int = 96
    tok_hdr: int = 48
    tok_box: int = 52
    compute: str = "serial"
    c_det: float = 29.0
    c_trk: float = 2.0
    c_enc: float = 92.0
    c_bs_frame: float = 20.0
    c_bs_tok: float = 1.0
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
    N_ref: int = 20
    W_hold: int = 10
    frame_bits0: float = 1.5e5
    seed: int = 0
    trk: dict = field(default_factory=dict)


class Link:
    """FIFO serialising link; keyed (paired) loss / jitter / duplicate draws."""

    def __init__(self, rate, delay, p, seed, loss_stream, ge=None, p_dup=0.0, jitter=0.0, mtu=9600):
        self.rate, self.delay, self.p, self.seed, self.ls = rate, delay, p, seed, loss_stream
        self.ge, self.p_dup, self.jitter, self.mtu = ge, p_dup, jitter, mtu
        self.busy = 0.0
        self._ge = []                      # per 10 ms slot: bad state (exogenous chain, extended on demand)

    def _bad(self, slot):
        p_gb, p_bg, _ = self.ge
        while len(self._ge) <= slot:
            prev = self._ge[-1] if self._ge else False
            u = ku(self.seed, GE_CHAIN, len(self._ge))
            self._ge.append((u < p_gb) if not prev else (u >= p_bg))
        return self._ge[slot]

    def send(self, now, bits, n_pk, key):
        start = max(now, self.busy)
        end = start + 1000.0 * bits / self.rate
        self.busy = end
        lost = False
        for j in range(n_pk):
            u = ku(self.seed, self.ls, *key, j)
            if self.ge is None:
                lost |= u < self.p
            else:
                t_j = start + 1000.0 * min(bits, (j + 1) * self.mtu) / self.rate
                lost |= self._bad(int(t_j // 10)) and u < self.ge[2]
        arr = end + self.delay + (ku(self.seed, DL_JIT, *key) * self.jitter if self.jitter else 0.0)
        out = [] if lost else [arr]
        if out and self.p_dup and ku(self.seed, DL_DUP, *key) < self.p_dup:
            out.append(arr + ku(self.seed, DL_DUPD, *key) * (self.jitter + 200.0))
        return out, end, start


def n_packets(bits, mtu):
    return max(1, math.ceil(bits / mtu))


def run_clip(clip, cfg, trace=False):
    T = cfg.tick_ms
    W, H = clip.W, clip.H
    n = len(clip.gt)
    cond = cfg.cond
    tk = dict(cfg.trk)
    bs, local, virt = Tracker(W, H, **tk), Tracker(W, H, **tk), Tracker(W, H, **tk)
    versioned = cond == "C1"
    pub = P.Publisher(versioned=versioned)
    rep = P.Replica(versioned=versioned, w_hold=cfg.W_hold)
    ul = Link(cfg.R_ul, cfg.d_ul, cfg.p_ul, cfg.seed, UL_LOSS, mtu=cfg.mtu_bits)
    dl = Link(cfg.DL_cap, cfg.d_dl, cfg.p_dl, cfg.seed, DL_LOSS, ge=cfg.ge, p_dup=cfg.p_dup, jitter=cfg.jitter_dl,
              mtu=cfg.mtu_bits)
    ev, seq = [], [0]

    def push(t, kind, data=None):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, data))

    S = dict(ul_bits=0, dl_bits=0, n_frame=0, n_tok=0, n_skip=0, dropped_compute=0, ul_lost=0, ul_late=0,
             dl_msgs=0, dl_lost=0, snapshots=0, desync_ticks=0, uav_ms=0.0, bs_ms=0.0, belief_f1=[], oos=0,
             ghost_ticks=0, ghost_uav_absent=0, ghost_known_absent=0, decided=0)
    ul_log, dl_sent_in_tick, pending_ul = [], {}, []
    scorer = Scorer(tick_s=T / 1000)
    bucket = [cfg.R_ul * cfg.bucket_s, 0.0]
    worker = dict(busy=0.0, enc_busy=0.0)
    bs_busy = [0.0]
    bs_state = dict(dirty=False, acks=[], last_proc=0, hint_pending=False, snap_due=False, snap_req=False,
                    last_snap_tick=0, changed=set(), newest_obs=-1)
    sent_obs = {}
    acked_upto = [-1]
    hint = dict(items=[], fp=-1, follow={})
    local_hist = {}
    miss = {}
    est_frame = [float(cfg.frame_bits0)]
    last_frame = [-10 ** 9]
    actions = []

    # ---------------- BS side
    def canonical(t_state):
        prev = pub.current_at(t_state)
        st = {}
        for tr in bs.confirmed(t_state):
            if tr.tid in prev and tr.tid not in bs_state["changed"]:
                v = prev[tr.tid]
            else:
                v = P.quantise(tr.at(t_state), tr.vel, tr.cls, tr.conf, W, H)
            if P.representable(v):          # unrepresentable (left the frame range) -> deleted from the state
                st[tr.tid] = v
        return st

    def try_send(now):
        if dl.busy > now + 1e-9 or cond in ("A0", "D"):
            return
        f_now = int(now // T)
        wire, kind = None, None
        if cond == "A1":
            if bs_state["acks"]:
                nb, wire = P.pack_ack(bs_state["acks"]); kind = "ack"; bs_state["acks"] = []
        elif cond in ("B", "Bp"):
            if bs_state["hint_pending"]:
                fp = bs_state["last_proc"]
                items = []
                for tr in bs.confirmed(fp):
                    c, cx, cy, w, h, _, _, cf = P.quantise(tr.at(fp), (0, 0), tr.cls, tr.conf, W, H)
                    items.append((c, cx, cy, w, h, cf))
                nb, wire = P.pack_hint(fp, items, plus=(cond == "Bp")); kind = "hint"
                bs_state["hint_pending"] = False
        else:
            snap = bs_state["snap_due"] or bs_state["snap_req"]
            if bs_state["dirty"] or snap:
                m = pub.make(canonical(f_now), f_now, bs_state["last_proc"], snapshot=snap)
                bs_state["dirty"] = False; bs_state["changed"] = set()
                if snap:
                    bs_state["snap_due"] = bs_state["snap_req"] = False; bs_state["last_snap_tick"] = f_now
                    S["snapshots"] += 1
                if m is not None:
                    wire, kind = m.wire, "state"
        if wire is None:
            return
        bits = 8 * len(wire)
        tot = bits + cfg.app_hdr * n_packets(bits, cfg.mtu_bits)
        k = dl_sent_in_tick.get(f_now, 0); dl_sent_in_tick[f_now] = k + 1
        arrs, end, _ = dl.send(now, tot, n_packets(tot, cfg.mtu_bits), (f_now, k))
        S["dl_bits"] += tot; S["dl_msgs"] += 1
        if not arrs:
            S["dl_lost"] += 1
        for a in arrs:
            push(a, "DL_RX", (kind, wire))
        push(end, "DL_IDLE")

    # ---------------- UAV belief \hat T_f: list of (id, box, cls or None)
    def belief(f):
        if cond == "D":
            return [(tr.tid, tr.at(f), tr.cls) for tr in bs.confirmed(f)]
        if cond in ("A0", "A1"):
            return [(tr.tid, tr.at(f), tr.cls) for tr in virt.confirmed(f)]
        if cond in ("B", "Bp"):
            out = []
            for k, (c, b) in enumerate(hint["items"]):
                lt = hint["follow"].get(k)
                tr = next((x for x in local.tracks if x.tid == lt), None) if lt is not None else None
                out.append((("h", hint["fp"], k), tr.at(f) if tr is not None else np.asarray(b),
                            c if cond == "Bp" else None))
            return out
        st = rep.view(f)
        return [(tid, np.array(P.dequant_box(v, W, H)), v[0]) for tid, v in st.items()]

    def apply_dl(kind, wire, now):
        f_now = int(now // T)
        if kind == "ack":
            for c16 in sorted(P.unpack_ack(wire)):
                c_ = P.unwrap(c16, f_now)
                if c_ > acked_upto[0] and c_ in sent_obs:
                    obs, k = sent_obs.pop(c_)
                    virt.update(obs, c_, kind=k)
                    acked_upto[0] = c_
        elif kind == "hint":
            fp16, items = P.unpack_hint(wire, plus=(cond == "Bp"))
            fp = P.unwrap(fp16, f_now)
            if fp < hint["fp"]:
                return
            boxes = [(c, np.array(P.dequant_box((c, cx, cy, w, h), W, H))) for c, cx, cy, w, h, _ in items]
            hist = local_hist.get(fp, {})
            ids = list(hist)
            pairs, _, _ = match([b for _, b in boxes], [hist[i] for i in ids], 0.3)
            hint.update(items=boxes, fp=fp, follow={i: ids[j] for i, j, _ in pairs})
        else:
            rep.receive(P.unpack_state_msg(wire, versioned), f_now)

    # ---------------- UAV decision at tick f
    def decide(f, now):
        S["decided"] += 1
        dets = clip.det_n[f]
        local.update([d for d in dets if d[2] >= 0.05], f, kind="full")
        local_hist[f] = {tr.tid: tr.at(f) for tr in local.tracks}
        local_hist.pop(f - 100, None)
        L = [(tr.tid, tr.at(f), tr.cls, tr.conf) for tr in local.confirmed(f) if tr.t_obs == f]
        if cond in ("C0", "C1"):
            rep.tick(f)
            if rep.desync:
                S["desync_ticks"] += 1
        if cond in ("A0", "A1"):
            virt.prune(f)
        Th = belief(f)
        lb = [x[1] for x in L]; tb = [x[1] for x in Th]
        pairs, uL, uT = match(lb, tb, 0.3)
        mism = [(i, j, v) for i, j, v in pairs if v < 0.5 or (Th[j][2] is not None and Th[j][2] != L[i][2])]
        new = len(uL); drift = len(mism)
        mt = {Th[j][0] for _, j, _ in pairs}
        for i_, _, _ in Th:
            miss[i_] = 0 if i_ in mt else miss.get(i_, 0) + 1
        live = {x[0] for x in Th}
        for k in [k for k in miss if k not in live]:
            del miss[k]
        lost = sum(1 for x in Th if miss[x[0]] >= cfg.k_lost)
        unc = 0.0
        if cfg.w_unc:
            low = [d for d in dets if 0.05 <= d[2] < 0.25]
            if low:
                M = iou_matrix([d[0] for d in low], tb) if tb else np.zeros((len(low), 0))
                for i_, d in enumerate(low):
                    if M.shape[1] == 0 or M[i_].max() < 0.3:
                        b = d[0]; unc += (b[2] - b[0]) * (b[3] - b[1]) / (W * H) * 100
        g = cfg.w_new * new + cfg.w_lost * lost + cfg.w_drift * drift + cfg.w_unc * unc
        # ---- diagnostics only (never used for decisions): belief vs true BS state, BS ghost tracks
        trs = bs.confirmed(f)
        true = [tr.at(f) for tr in trs]
        if true or tb:
            pp, _, _ = match(tb, true, 0.5)
            S["belief_f1"].append(2 * len(pp) / max(len(tb) + len(true), 1))
        if trs:
            gt = clip.gt[f]
            gh = []
            for c in {tr.cls for tr in trs}:
                ti = [k for k, tr in enumerate(trs) if tr.cls == c]
                gi = [g_[1] for g_ in gt if g_[2] == c]
                _, ut, _ = match([true[k] for k in ti], gi, 0.5)
                gh += [ti[k] for k in ut]
            if gh:
                ub = [d[0] for d in dets if d[2] >= 0.25]
                Mu = iou_matrix([true[k] for k in gh], ub) if ub else np.zeros((len(gh), 0))
                Mb = iou_matrix([true[k] for k in gh], tb) if tb else np.zeros((len(gh), 0))
                for a, k in enumerate(gh):
                    absent = Mu.shape[1] == 0 or Mu[a].max() < 0.3
                    known = Mb.shape[1] > 0 and Mb[a].max() >= 0.3
                    S["ghost_ticks"] += 1; S["ghost_uav_absent"] += absent; S["ghost_known_absent"] += absent and known
        # ---- action
        bucket[0] = min(cfg.R_ul * cfg.bucket_s, bucket[0] + cfg.R_ul * (now - bucket[1]) / 1000.0); bucket[1] = now
        sync = P.UL_SYNC if cond == "C1" else 0
        dev = ([L[i] for i in uL] + [L[i] for i, _, _ in mism])[:cfg.tok_max]
        tok_bits = cfg.tok_hdr + cfg.tok_box * len(dev) + sync
        fr_bits = est_frame[0] + sync
        forced = f - last_frame[0] >= cfg.N_max

        def meets_deadline(bits, ready):
            # local information only: own link backlog + own packets still being encoded (not yet queued)
            b = ul.busy
            for r_, bb in sorted(pending_ul):
                if r_ <= ready:
                    b = max(b, r_) + 1000.0 * bb / cfg.R_ul
            tot = bits + cfg.app_hdr * n_packets(bits, cfg.mtu_bits)
            return max(ready, b) + 1000.0 * tot / cfg.R_ul + cfg.d_ul - f * T <= cfg.D_max

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
            ready = enc_ready
            if cfg.compute != "serial":
                worker["enc_busy"] = ready
            obs_bs = ("frame", clip.det_s[f])
            obs_virt = ([d for d in dets if d[2] >= 0.05], "full")
        else:
            S["n_tok"] += 1
            pay = tok_bits
            ready = now
            obs_bs = ("tok", [(x[1], x[2], x[3]) for x in dev])
            obs_virt = ([(x[1], x[2], x[3]) for x in dev], "partial")
        npk = n_packets(pay, cfg.mtu_bits)
        tot = pay + cfg.app_hdr * npk
        bucket[0] -= tot
        pending_ul.append((ready, tot))
        push(ready, "UL_ENQ", (f, act, tot, ready, obs_bs))     # enters the uplink queue when encoded
        if cond == "A0":
            virt.update(obs_virt[0], f, kind=obs_virt[1])
        elif cond == "A1":
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
            if cond in ("C0", "C1"):
                if set(pub.current_at(f)) != set(canonical(f)):
                    bs_state["dirty"] = True
                if f - bs_state["last_snap_tick"] >= cfg.N_ref:
                    bs_state["snap_due"] = True
                try_send(now)
            scorer.add(f, clip.gt[f], clip.ign[f], [((tr.tid, tr.born), tr.at(f), tr.cls) for tr in bs.confirmed(f)])
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
        elif kind == "UL_ENQ":
            f_cap, act, tot, ready, obs_bs = data
            pending_ul.remove((ready, tot))
            arrs, _, start = ul.send(now, tot, n_packets(tot, cfg.mtu_bits), (f_cap, 0))
            S["ul_bits"] += tot
            hdr = rep.header() if cond == "C1" else None
            ul_log.append(dict(f=f_cap, act=act, bits=tot, n_pk=n_packets(tot, cfg.mtu_bits), ready=ready,
                               start=start, lost=not arrs, arr=arrs[0] if arrs else None))
            if not arrs:
                S["ul_lost"] += 1
            else:
                push(arrs[0], "BS_RX", (f_cap, obs_bs, hdr))
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
            if f_cap < bs_state["newest_obs"]:
                S["oos"] += 1
            bs_state["newest_obs"] = max(bs_state["newest_obs"], f_cap)
            bs.update(obs[1], f_cap, kind="full" if obs[0] == "frame" else "partial")
            bs_state["changed"] |= bs.last_changed
            bs_state["last_proc"] = max(bs_state["last_proc"], f_cap)
            bs_state["dirty"] = True; bs_state["hint_pending"] = True
            bs_state["acks"].append(f_cap)
            if hdr is not None and pub.check_uplink(*hdr):
                bs_state["snap_req"] = True
            try_send(now)
        elif kind == "DL_IDLE":
            try_send(now)
        elif kind == "DL_RX":
            apply_dl(data[0], data[1], now)
    R = scorer.result()
    dur = n * T / 1000
    R.update(clip=clip.name, cond=cond, seed=cfg.seed, ul_kbps=S["ul_bits"] / dur / 1e3,
             dl_kbps=S["dl_bits"] / dur / 1e3, ul_bits=S["ul_bits"], dl_bits=S["dl_bits"],
             n_frame=S["n_frame"], n_tok=S["n_tok"], n_skip=S["n_skip"], dropped_compute=S["dropped_compute"],
             ul_lost=S["ul_lost"], ul_late=S["ul_late"], dl_msgs=S["dl_msgs"], dl_lost=S["dl_lost"],
             snapshots=S["snapshots"], desync_ticks=S["desync_ticks"], oos_obs=S["oos"],
             uav_ms_per_tick=S["uav_ms"] / n, bs_ms_per_tick=S["bs_ms"] / n,
             belief_f1=100 * float(np.mean(S["belief_f1"])) if S["belief_f1"] else None,
             ghost_s_per_min=S["ghost_ticks"] * T / 1000 / max(dur / 60, 1e-9),
             ghost_uav_absent_frac=S["ghost_uav_absent"] / max(S["ghost_ticks"], 1),
             ghost_known_absent_frac=S["ghost_known_absent"] / max(S["ghost_ticks"], 1))
    if trace:
        R["actions"] = actions; R["ul_log"] = ul_log
    return R
