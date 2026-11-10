"""R1b-0a unit tests: protocol (design v3 sec 4 tests 1-4) + bit-exact wire format, metrics (class-aware primary,
IDF1 vs py-motmetrics), tracker incl. out-of-sequence observations, simulator causality / hidden-state leakage /
paired channel randomness / compute timing / deadline / budget / header overhead, baselines B and B+, split."""
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

import r1b_proto as P
from r1b_metrics import Scorer
from r1b_sim import Cfg, Clip, Link, run_clip
from r1b_split import check_split
from r1b_track import Tracker

REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- helpers
def rand_track(rng):
    return (rng.randrange(1, 11), rng.randrange(4096), rng.randrange(4096), rng.randrange(1, 4096),
            rng.randrange(1, 4096), rng.randrange(-512, 512), rng.randrange(-512, 512), rng.randrange(8))


def rand_state(rng, n):
    return {rng.randrange(1024): rand_track(rng) for _ in range(n)}


def evolve(rng, st):
    st = dict(st)
    for k in list(st):
        r = rng.random()
        if r < 0.1:
            del st[k]
        elif r < 0.4:
            c, cx, cy, w, h, vx, vy, cf = st[k]
            st[k] = (c, min(max(cx + rng.randrange(-20, 21), 0), 4095), cy, w, h,
                     min(max(vx + rng.randrange(-3, 4), -512), 511), vy, cf)
    for _ in range(rng.randrange(0, 3)):
        k = rng.randrange(1024)
        if k not in st:
            st[k] = rand_track(rng)
    return st


def synth_clip(seed=0, n=120, n_obj=12, W=1344, H=756, noise=2.0, miss=0.1, fp=1, bits=150_000, cls_flip=0.0,
               speed=3.0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([100, 100], [W - 100, H - 100], (n_obj, 2)); vel = rng.normal(0, speed, (n_obj, 2))
    size = rng.uniform(20, 60, (n_obj, 2)); cls = rng.integers(1, 11, n_obj)
    gt, dn, ds = [], [], []
    for f in range(n):
        p = pos + vel * f
        boxes = [(i, np.array([*(p[i] - size[i] / 2), *(p[i] + size[i] / 2)]), int(cls[i])) for i in range(n_obj)
                 if 0 < p[i, 0] < W and 0 < p[i, 1] < H]
        gt.append(boxes)

        def det(sd, mr, nfp):
            out = []
            for _, b, c in boxes:
                if rng.random() > mr:
                    cc = c if rng.random() >= cls_flip else int(rng.integers(1, 11))
                    out.append((b + rng.normal(0, sd, 4), cc, float(rng.uniform(0.5, 0.95))))
            for _ in range(nfp):
                c0 = rng.uniform([0, 0], [W, H]); out.append((np.array([*c0, *(c0 + 30)]), 4, float(rng.uniform(0.05, 0.45))))
            return out
        dn.append(det(noise, miss, fp)); ds.append(det(noise / 2, miss / 2, 0))
    return Clip("synth", W, H, gt, [[] for _ in range(n)], dn, ds, [bits] * n)


# ---------------------------------------------------------------- protocol (design sec 4)
def test_serial_arithmetic_wraparound():
    assert P.s_newer(1, 65535) and not P.s_newer(65535, 1)
    assert P.s_newer(0, 65535) and P.s_diff(2, 65534) == 4 and P.s_diff(65534, 2) == -4
    assert not P.s_newer(5, 5) and P.unwrap(3, 65540) == 65539 and P.unwrap(65535, 65540) == 65535
    pub = P.Publisher(); rep = P.Replica()
    pub.version = 65530; pub.msg_id = 65530
    pub.hist = {65530: (0, {}, P.crc({}, 0))}
    rep.applied = 65530
    rng = random.Random(1); st = {}
    t = 65520
    for _ in range(30):                        # versions and t_state both cross 2^16
        t += 1
        st = evolve(rng, st)
        m = pub.make(st, t, t)
        if m is None:
            continue
        rep.receive(P.unpack_state_msg(m.wire, True), t)
        assert not rep.desync and rep.applied == pub.version and rep.state == pub.state
    assert pub.version < 100


def test_replica_converges_under_loss_dup_reorder():
    """10^5 wire messages, 10% loss, 5% duplicates, random reordering, snapshot every 20 messages + on request."""
    rng = random.Random(7)
    pub = P.Publisher(n_hist=64); rep = P.Replica(w_hold=10)
    st = rand_state(rng, 15)
    inflight = []
    seq = 0; t = 0; checked = 0; n_msgs = 0
    while n_msgs < 100_000:
        t += 1
        st = evolve(rng, st)
        m = pub.make(st, t, t, snapshot=(n_msgs % 20 == 0))
        if m is not None:
            n_msgs += 1
            if rng.random() > 0.10:
                inflight.append((t + rng.randrange(0, 4), seq, m.wire)); seq += 1
                if rng.random() < 0.05:
                    inflight.append((t + rng.randrange(0, 6), seq, m.wire)); seq += 1
        inflight.sort(key=lambda x: (x[0], x[1]))
        while inflight and inflight[0][0] <= t:
            _, _, w = inflight.pop(0)
            mm = P.unpack_state_msg(w, True)
            rep.receive(mm, t)
            if mm.snapshot and rep.applied == mm.version:
                h = pub.hist.get(rep.applied)
                if h is not None:
                    assert not rep.desync and rep.state == h[1] and rep.t_state == h[0]
                    checked += 1
        rep.tick(t)
        if pub.check_uplink(*P.unpack_ul_sync(P.pack_ul_sync(*rep.header())[1])):
            m2 = pub.make(st, t, t, snapshot=True)
            if m2 is not None and rng.random() > 0.10:
                inflight.append((t + 1, seq, m2.wire)); seq += 1
    assert checked > 3000


def test_hash_detects_every_single_unit_deviation():
    rng = random.Random(3)
    for _ in range(3000):
        st = rand_state(rng, rng.randrange(1, 40)); t = rng.randrange(P.MOD)
        k = rng.choice(list(st)); f = rng.randrange(8)
        bad = dict(st); v = list(bad[k]); v[f] += rng.choice((-1, 1)); bad[k] = tuple(v)
        assert P.crc(st, t) != P.crc(bad, t)
        assert P.crc(st, t) != P.crc(st, (t + 1) % P.MOD)
    pub = P.Publisher(); rep = P.Replica()
    m = pub.make(rand_state(rng, 10), 5, 5)
    mm = P.unpack_state_msg(m.wire, True)
    kind, k, v = mm.ops[0]; v = list(v); v[1] = (v[1] + 1) % 4096
    mm.ops[0] = (kind, k, tuple(v))
    rep.receive(mm, 5)
    assert rep.desync


def test_predictor_bit_exact_1000_ticks():
    rng = random.Random(5)
    st = rand_state(rng, 30)
    step = dict(st)
    for _ in range(1000):
        step = P.predict(step, 1)
    assert step == P.predict(st, 1000)
    assert P.crc(step, 1000) == P.crc(P.predict(st, 1000), 1000)


# ---------------------------------------------------------------- bit-exact wire format
@pytest.mark.parametrize("versioned", [True, False])
def test_state_msg_roundtrip_and_exact_size(versioned):
    rng = random.Random(11)
    pub = P.Publisher(versioned=versioned)
    st = {}
    for t in range(1, 400):
        st = evolve(rng, st)
        m = pub.make(st, t, t - 1, snapshot=(t % 37 == 0))
        if m is None:
            continue
        hdr = P.C1_HDR if versioned else P.C0_HDR
        exp = hdr + sum(P.DEL_BITS if k == P.DEL else P.OP_BITS for k, _, _ in m.ops)
        assert m.bits == exp and len(m.wire) == math.ceil(exp / 8)
        d = P.unpack_state_msg(m.wire, versioned)
        assert d.ops == m.ops and d.t_state == t % P.MOD and d.frame_processed == (t - 1) % P.MOD
        assert d.snapshot == m.snapshot and d.msg_id == m.msg_id
        if versioned:
            assert (d.base, d.version, d.h) == (m.base, m.version, m.h)


def test_full_1024_track_snapshot_and_field_extremes():
    rng = random.Random(2)
    st = {k: rand_track(rng) for k in range(1024)}
    st[0] = (15, 4095, 4095, 4095, 4095, 511, 511, 7)
    st[1] = (0, 0, 0, 1, 1, -512, -512, 0)
    pub = P.Publisher()
    m = pub.make(st, 70000, 69999, snapshot=True)
    assert len(m.ops) == 1024 and m.bits == P.C1_HDR + 1024 * P.OP_BITS
    d = P.unpack_state_msg(m.wire, True)
    rep = P.Replica(); rep.receive(d, 70000)
    assert rep.state == st and not rep.desync


@pytest.mark.parametrize("bad", [(16, 1, 1, 1, 1, 0, 0, 0), (1, 4096, 1, 1, 1, 0, 0, 0), (1, -1, 1, 1, 1, 0, 0, 0),
                                 (1, 1, 1, 0, 1, 0, 0, 0), (1, 1, 1, 1, 4096, 0, 0, 0), (1, 1, 1, 1, 1, 512, 0, 0),
                                 (1, 1, 1, 1, 1, 0, -513, 0), (1, 1, 1, 1, 1, 0, 0, 8)])
def test_out_of_range_fields_are_rejected(bad):
    with pytest.raises(ValueError):
        P.Publisher().make({5: bad}, 1, 1)
    m = P.Msg(1, 0, 1, 1, 1, [(P.ADD, 5, bad)], 0, True)
    with pytest.raises(ValueError):
        P.pack_state_msg(m, True)


def test_n_ops_limit_and_bad_ids():
    m = P.Msg(1, 0, 1, 1, 1, [(P.DEL, k % 1024, None) for k in range(P.MAX_OPS + 1)], 0, False)
    with pytest.raises(ValueError):
        P.pack_state_msg(m, True)
    with pytest.raises(ValueError):
        P.pack_state_msg(P.Msg(1, 0, 1, 1, 1, [(P.DEL, 1024, None)], 0, False), True)
    with pytest.raises(ValueError):
        P.unpack_state_msg(P.pack_state_msg(P.Msg(1, 0, 1, 1, 1, [(P.ADD, 3, (1,) * 8)], 0, False), True)[1][:10],
                           True)


def test_hint_ack_ulsync_roundtrip_sizes():
    rng = random.Random(4)
    items = [(rng.randrange(16), rng.randrange(4096), rng.randrange(4096), rng.randrange(1, 4096),
              rng.randrange(1, 4096), rng.randrange(8)) for _ in range(77)]
    for plus in (False, True):
        nb, w = P.pack_hint(65541, items, plus)
        assert nb == P.HINT_HDR + 77 * (P.HINT_BOX_PLUS if plus else P.HINT_BOX)
        fp, out = P.unpack_hint(w, plus)
        assert fp == 65541 % P.MOD
        assert out == (items if plus else [(0, cx, cy, ww, hh, 0) for _, cx, cy, ww, hh, _ in items])
    nb, w = P.pack_ack([3, 70000, 65535])
    assert nb == P.ACK_HDR + 3 * P.ACK_ITEM and P.unpack_ack(w) == [3, 70000 % P.MOD, 65535]
    nb, w = P.pack_ul_sync(65535, 0xFFFFFFFF)
    assert nb == 48 and P.unpack_ul_sync(w) == (65535, 0xFFFFFFFF)


# ---------------------------------------------------------------- metrics
def test_metrics_gt_as_output_is_perfect():
    clip = synth_clip(1)
    sc = Scorer()
    for f in range(len(clip.gt)):
        sc.add(f, clip.gt[f], [], list(clip.gt[f]))
    r = sc.result()
    assert r["F1"] == pytest.approx(100) and r["IDF1"] == pytest.approx(100) and r["fp"] == 0
    assert r["F1_ag"] == pytest.approx(100) and r["first_det_median_s"] == 0 and r["never_detected_frac"] == 0


def test_right_box_wrong_class_is_fp_and_fn():
    sc = Scorer()
    b = np.array([100, 100, 150, 150])
    sc.add(0, [(1, b, 4)], [], [(9, b, 1)])        # GT car, predicted pedestrian, IoU 1
    r = sc.result()
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1) and r["F1"] == 0
    assert (r["tp_ag"], r["fp_ag"], r["fn_ag"]) == (1, 0, 0)


def test_metrics_injected_delay():
    clip = synth_clip(2)
    sc = Scorer()
    k = 7
    for f in range(len(clip.gt)):
        sc.add(f, clip.gt[f], [], list(clip.gt[f]) if f >= k else [])
    r = sc.result()
    assert r["first_det_median_s"] == pytest.approx(k * 0.1)
    assert r["fn"] == sum(len(g) for g in clip.gt[:k])


def test_ignored_regions_do_not_count_as_fp():
    sc = Scorer()
    sc.add(0, [], [np.array([0, 0, 100, 100])], [(1, np.array([10, 10, 40, 40]), 4), (2, np.array([500, 500, 530, 530]), 4)])
    assert sc.fp == 1


def _noisy_tracks(seed):
    rng = np.random.default_rng(seed)
    clip = synth_clip(seed, n=80, n_obj=15, noise=0)
    out = []
    pid_of = {}
    for f, g in enumerate(clip.gt):
        o = []
        for gid, b, c in g:
            if rng.random() < 0.1:
                continue
            if gid not in pid_of or rng.random() < 0.03:          # id switches
                pid_of[gid] = int(rng.integers(0, 10 ** 6))
            cc = c if rng.random() > 0.15 else int(rng.integers(1, 11))
            o.append((pid_of[gid], b + rng.normal(0, 6, 4), cc))
        for _ in range(2):
            c0 = rng.uniform(0, 600, 2); o.append((int(rng.integers(10 ** 7, 10 ** 8)), np.array([*c0, *(c0 + 40)]), 3))
        out.append(o)
    return clip, out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_idf1_matches_py_motmetrics(seed):
    mm = pytest.importorskip("motmetrics")
    clip, out = _noisy_tracks(seed)
    sc = Scorer()
    for f in range(len(clip.gt)):
        sc.add(f, clip.gt[f], [], out[f])
    r = sc.result()

    def xywh(bs):
        return np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in bs]).reshape(-1, 4)

    def run(sel):
        acc = mm.MOTAccumulator(auto_id=True)
        for f in range(len(clip.gt)):
            g = [x for x in clip.gt[f] if sel(x[2])]; o = [x for x in out[f] if sel(x[2])]
            acc.update([x[0] for x in g], [x[0] for x in o],
                       mm.distances.iou_matrix(xywh([x[1] for x in g]), xywh([x[1] for x in o]), max_iou=0.5))
        s = mm.metrics.create().compute(acc, metrics=["idtp", "num_objects", "num_predictions"])
        return float(s["idtp"].iloc[0]), float(s["num_objects"].iloc[0]), float(s["num_predictions"].iloc[0])
    idtp, no, npred = run(lambda c: True)
    assert r["IDF1_ag"] == pytest.approx(100 * 2 * idtp / (no + npred), abs=1e-9)
    tot = np.zeros(3)
    for c in range(1, 11):
        tot += run(lambda x, c=c: x == c)
    assert r["IDF1"] == pytest.approx(100 * 2 * tot[0] / (tot[1] + tot[2]), abs=1e-9)
    assert r["IDF1"] < r["IDF1_ag"]


# ---------------------------------------------------------------- tracker
def test_tracker_on_gt_keeps_ids():
    clip = synth_clip(3, noise=0.0, miss=0.0, fp=0)
    tr = Tracker(clip.W, clip.H)
    sc = Scorer()
    for f in range(len(clip.gt)):
        tr.update([(b, c, 0.9) for _, b, c in clip.gt[f]], f)
        sc.add(f, clip.gt[f], [], [((t.tid, t.born), t.at(f), t.cls) for t in tr.confirmed(f)])
    r = sc.result()
    assert r["F1"] > 99 and r["IDF1"] > 99


def test_tracker_out_of_sequence_observation_does_not_roll_back():
    tr = Tracker(1000, 1000)
    b = np.array([100.0, 100, 140, 140])
    tr.update([(b, 4, 0.9)], 0); tr.update([(b + 10, 4, 0.9)], 10)
    t0 = tr.tracks[0]; box10, ref = t0.box.copy(), t0.t_ref
    tr.update([(b + 5, 4, 0.9)], 8)                       # older capture arrives late
    assert len(tr.tracks) == 1 and tr.tracks[0].t_ref == ref and np.allclose(tr.tracks[0].box, box10)
    assert tr.last_changed == set()


# ---------------------------------------------------------------- simulator: causality, leakage
FAST = dict(R_ul=2e6, th_T=1.0, th_F=3.0, N_max=10, frame_bits0=150_000)


@pytest.mark.parametrize("cond", ["A0", "A1", "B", "Bp", "C0", "C1", "D"])
def test_no_lookahead_truncation(cond):
    clip = synth_clip(4, n=100)
    m = 60
    short = Clip("s", clip.W, clip.H, clip.gt[:m], clip.ign[:m], clip.det_n[:m], clip.det_s[:m], clip.frame_bits[:m])
    a = run_clip(clip, Cfg(cond=cond, **FAST), trace=True)["actions"]
    b = run_clip(short, Cfg(cond=cond, **FAST), trace=True)["actions"]
    assert a[:len(b) - 3] == b[:len(b) - 3]


@pytest.mark.parametrize("cond", ["A0", "A1", "B", "Bp", "C0", "C1"])
def test_no_hidden_bs_state_without_feedback(cond):
    c1 = synth_clip(5, n=80); c2 = synth_clip(5, n=80)
    rng = np.random.default_rng(9)
    c2.det_s = [[(b + rng.normal(0, 30, 4), 2, 0.9) for b, _, _ in d][: max(0, len(d) - 3)] for d in c1.det_s]
    a = run_clip(c1, Cfg(cond=cond, p_dl=1.0, **FAST), trace=True)["actions"]
    b = run_clip(c2, Cfg(cond=cond, p_dl=1.0, **FAST), trace=True)["actions"]
    assert a == b


def test_oracle_does_see_bs_state():
    c1 = synth_clip(5, n=80); c2 = synth_clip(5, n=80)
    c2.det_s = [[] for _ in c1.det_s]
    a = run_clip(c1, Cfg(cond="D", **FAST), trace=True)["actions"]
    b = run_clip(c2, Cfg(cond="D", **FAST), trace=True)["actions"]
    assert a != b


# ---------------------------------------------------------------- paired channel randomness
def test_uplink_losses_are_paired_across_conditions():
    clip = synth_clip(12, n=200, bits=20_000)
    base = dict(R_ul=2e6, th_T=99, th_F=0.0, N_max=1, p_ul=0.3, frame_bits0=20_000, seed=3)   # same action everywhere
    logs = {c: {e["f"]: e for e in run_clip(clip, Cfg(cond=c, **base), trace=True)["ul_log"]} for c in ("A1", "C1", "B")}
    common = 0
    for f in set(logs["A1"]) & set(logs["C1"]) & set(logs["B"]):
        if logs["A1"][f]["n_pk"] == logs["C1"][f]["n_pk"] == logs["B"][f]["n_pk"]:
            assert logs["A1"][f]["lost"] == logs["C1"][f]["lost"] == logs["B"][f]["lost"]
            common += 1
    assert common > 20 and any(e["lost"] for e in logs["A1"].values())


def test_channel_seeds_differ_and_ge_chain_is_exogenous():
    l1 = Link(1e5, 0, 0.0, 7, 2, ge=(0.05, 0.2, 1.0)); l2 = Link(1e5, 0, 0.0, 7, 2, ge=(0.05, 0.2, 1.0))
    a = [l1._bad(s) for s in range(0, 3000, 7)]
    l2._bad(2999)                                          # different query pattern
    assert a == [l2._bad(s) for s in range(0, 3000, 7)] and any(a) and not all(a)
    clip = synth_clip(12, n=150, bits=20_000)
    base = dict(cond="A1", R_ul=2e6, th_T=0.5, th_F=2.0, N_max=8, p_ul=0.3, frame_bits0=20_000)
    l_a = [e["lost"] for e in run_clip(clip, Cfg(seed=1, **base), trace=True)["ul_log"]]
    l_b = [e["lost"] for e in run_clip(clip, Cfg(seed=2, **base), trace=True)["ul_log"]]
    assert l_a != l_b


# ---------------------------------------------------------------- timing, deadline, budget, overhead
def test_serial_compute_drops_next_frame_after_encode():
    clip = synth_clip(6, n=40)
    r = run_clip(clip, Cfg(cond="A0", R_ul=5e6, th_F=0.0, th_T=0.0, c_det=29, c_enc=92, D_max=1000), trace=True)
    acts = r["actions"]
    for i, a in enumerate(acts[:-1]):
        if a == "FRAME":
            assert acts[i + 1] == "DROP"       # 29 + 2 + 92 = 123 ms > 100 ms tick
    assert "DROP" in acts
    rp = run_clip(clip, Cfg(cond="A0", R_ul=5e6, th_F=0.0, th_T=0.0, c_det=29, c_enc=92, compute="pipelined",
                            D_max=1000), trace=True)
    assert "DROP" not in rp["actions"]


def test_exact_header_overhead_and_arrival_time():
    clip = synth_clip(13, n=5, bits=100_000)
    cfg = Cfg(cond="C1", R_ul=1e6, th_F=0.0, th_T=99, N_max=1000, d_ul=10, c_det=29, c_trk=2, c_enc=92, D_max=1000,
              frame_bits0=100_000, p_ul=0.0)
    e = run_clip(clip, cfg, trace=True)["ul_log"][0]
    pay = 100_000 + P.UL_SYNC
    assert e["bits"] == pay + 96 * math.ceil(pay / 9600)
    assert e["ready"] == pytest.approx(29 + 2 + 92)
    assert e["arr"] == pytest.approx(29 + 2 + 92 + 1000 * e["bits"] / 1e6 + 10)


def test_deadline_admission_and_late_drop():
    clip = synth_clip(7, n=40, bits=400_000)
    r = run_clip(clip, Cfg(cond="A0", R_ul=1e6, th_F=0.0, th_T=99.0, D_max=300, frame_bits0=400_000), trace=True)
    assert r["n_frame"] == 0 and r["ul_late"] == 0
    r = run_clip(clip, Cfg(cond="A0", R_ul=1e6, th_F=0.0, th_T=99.0, D_max=300, frame_bits0=50_000), trace=True)
    assert r["n_frame"] > 0 and r["ul_late"] > 0 and r["ul_kbps"] > 0
    for e in r["ul_log"]:
        if e["arr"] is not None and e["arr"] - e["f"] * 100 > 300:
            assert e["bits"] > 0                       # late frames still cost bits


def test_per_action_delivery_counters():
    clip = synth_clip(17, n=150, bits=290_000)
    r = run_clip(clip, Cfg(cond="A1", R_ul=4e6, th_T=0.5, th_F=3.0, N_max=10, p_ul=0.01, frame_bits0=290_000), trace=True)
    log = r["ul_log"]
    assert r["frame_lost"] == sum(e["lost"] for e in log if e["act"] == "FRAME")
    assert r["tok_lost"] == sum(e["lost"] for e in log if e["act"] == "TOK")
    assert r["frame_lost"] + r["tok_lost"] == r["ul_lost"] and r["frame_late"] + r["tok_late"] == r["ul_late"]
    assert 0 <= r["frame_ok_frac"] <= 1 and 0 <= r["tok_ok_frac"] <= 1
    # all-or-nothing: P(FRAME lost) = 1 - 0.99^31 ~ 27 %; tokens fit in one packet (~1 %)
    n_pk = math.ceil((290_000 + 96 * 31) / 9600)
    assert n_pk >= 30


def test_budget_holds_when_real_frame_sizes_differ_from_estimate():
    clip = synth_clip(14, n=200, bits=300_000)
    R = 2e6
    r = run_clip(clip, Cfg(cond="A1", R_ul=R, th_F=0.0, th_T=0.0, frame_bits0=50_000, D_max=2000), trace=True)
    dur = 200 * 0.1
    assert r["ul_bits"] <= R * dur + R * 1.0 + 300_000 + 96 * 40


def test_pipelined_out_of_sequence_at_bs_is_handled():
    clip = synth_clip(15, n=120)
    r = run_clip(clip, Cfg(cond="C1", R_ul=5e6, th_F=3.0, th_T=0.5, N_max=5, c_enc=150, compute="pipelined",
                           D_max=1000, frame_bits0=150_000), trace=True)
    assert r["oos_obs"] > 0 and r["desync_ticks"] == 0


def test_fast_objects_leave_frame_without_unrepresentable_state():
    clip = synth_clip(16, n=150, speed=12.0)
    for cond in ("C0", "C1"):
        r = run_clip(clip, Cfg(cond=cond, **FAST))
        assert r["F1"] >= 0


def test_ideal_channel_frames_every_tick_track_well():
    clip = synth_clip(8, n=100, noise=0.5, miss=0.0, fp=0)
    r = run_clip(clip, Cfg(cond="D", R_ul=1e9, th_F=0.0, d_dl=0, d_ul=0, p_ul=0, p_dl=0, c_enc=1, c_bs_frame=1,
                           frame_bits0=150_000))
    assert r["R"] > 95 and r["P"] > 95


# ---------------------------------------------------------------- baselines / diagnostics
def test_b_plus_carries_class_and_costs_more():
    clip = synth_clip(10, n=150, bits=40_000, cls_flip=0.2)
    base = dict(R_ul=1e6, th_T=1.0, th_F=4.0, N_max=30, p_dl=0.0, p_ul=0.0, frame_bits0=40_000)
    b = run_clip(clip, Cfg(cond="B", **base)); bp = run_clip(clip, Cfg(cond="Bp", **base))
    assert b["dl_msgs"] > 0 and bp["dl_msgs"] > 0          # per-box cost difference is checked at the wire level


def test_c1_belief_beats_a1_with_good_downlink_and_diagnostics():
    clip = synth_clip(10, n=150, bits=40_000)
    base = dict(R_ul=1e6, th_T=1.0, th_F=4.0, N_max=30, p_dl=0.0, p_ul=0.0, frame_bits0=40_000)
    a1 = run_clip(clip, Cfg(cond="A1", **base)); c1 = run_clip(clip, Cfg(cond="C1", **base))
    assert c1["belief_f1"] >= a1["belief_f1"] - 1e-9 and c1["desync_ticks"] == 0
    for r in (a1, c1):
        assert 0 <= r["ghost_uav_absent_frac"] <= 1 and 0 <= r["ghost_known_absent_frac"] <= r["ghost_uav_absent_frac"]


# ---------------------------------------------------------------- H1 decision rule (protocol sec 8)
def _rows(levels, f1, sources=("s%d" % i for i in range(7)), seeds=(0, 1)):
    """f1[(cond, level, source)] -> F1; build rows with tp/fp/fn giving exactly that F1 (P = R)."""
    rows = []
    for lv in levels:
        for c in ("A1", "B", "Bp", "C1", "D", "A0", "C0"):
            for s in [f"s{i}" for i in range(7)]:
                for sd in seeds:
                    v = f1(c, lv, s)
                    tp = int(round(10 * v)); fn = fp = 1000 - tp
                    rows.append(dict(cond=c, level_R=lv, clip=s, seed=sd, tp=tp, fp=fp, fn=fn, F1=100 * tp / 1000))
    return rows


def _verdict(f1):
    from r1b_run import h1_rule, verdict
    levels = [("1x", 1.0), ("2x", 2.0), ("4x", 4.0)]
    V = verdict(_rows([1.0, 2.0, 4.0], f1), levels, [f"s{i}" for i in range(7)], [0, 1], np.random.default_rng(0))
    for ln, _ in levels:
        V[ln]["d_C1_hint"] = V[ln]["F1"]["C1"] - max(V[ln]["F1"]["B"], V[ln]["F1"]["Bp"])
    return h1_rule(V, [ln for ln, _ in levels]), V


def test_h1_rule_pass_fail_inconclusive():
    base = {"A1": 30, "B": 29, "Bp": 30, "C1": 30, "D": 30, "A0": 28, "C0": 30}
    # PASS: C1 +4 over A1 at all levels in 6/7 sources, beats hints, D above
    (h, wins), V = _verdict(lambda c, lv, s: base[c] + (4.5 if c == "C1" and s != "s0" else 0) + (6 if c == "D" else 0))
    assert h == "PASS" and len(wins) == 3 and V["1x"]["n_sources_pos"] == 6
    # FAIL no headroom: D - A1 < 2 everywhere
    (h, _), _ = _verdict(lambda c, lv, s: base[c] + (1 if c == "D" else 0))
    assert h.startswith("FAIL (no headroom")
    # FAIL costed unusable: D +5, C1 +0.5
    (h, _), _ = _verdict(lambda c, lv, s: base[c] + (5 if c == "D" else 0) + (0.5 if c == "C1" else 0))
    assert h.startswith("FAIL (costed")
    # INCONCLUSIVE: C1 +1.5 (between +1 and +2), D +5
    (h, _), _ = _verdict(lambda c, lv, s: base[c] + (5 if c == "D" else 0) + (1.5 if c == "C1" else 0))
    assert h == "INCONCLUSIVE"
    # +3 pooled but only 4/7 sources positive -> no win -> INCONCLUSIVE
    (h, wins), _ = _verdict(lambda c, lv, s: base[c] + (5 if c == "D" else 0)
                            + ((7 if s in ("s0", "s1", "s2", "s3") else -1) if c == "C1" else 0))
    assert wins == [] and h == "INCONCLUSIVE"
    # C1 wins but loses to the strongest hint baseline -> not PASS
    (h, _), _ = _verdict(lambda c, lv, s: base[c] + (4 if c == "C1" else 0) + (6 if c in ("D", "Bp") else 0))
    assert h == "INCONCLUSIVE"


# ---------------------------------------------------------------- split
def test_split_v1():
    p = REPO / "runs/uav_semcom/r1b/split_v1.json"
    a = REPO / "runs/uav_semcom/vid_audit/vid_audit.json"
    if not p.exists() or not a.exists():
        pytest.skip("split / audit json not present")
    S = json.loads(p.read_text()); A = json.loads(a.read_text())
    assert check_split(S, sorted(A["splits"]["train"]["clips"]), sorted(A["splits"]["val"]["clips"])) == []
    assert S["gate"] == ["uav0000020_00406_v", "uav0000138_00000_v", "uav0000244_01440_v", "uav0000278_00001_v",
                         "uav0000289_00001_v", "uav0000295_02300_v", "uav0000316_01288_v"]
    assert S["quarantine"] == ["uav0000289_06922_v"] and len(S["tune"]) == 48
    assert S["aux_val"] == ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000268_05773_v"]
    assert S["vid_internal_confirmed_total"] == 0
