"""R1b-0a unit tests (design v3 sec 4 tests 1-4, checklist items 8-9, causal timing, split)."""
import json
import random
from pathlib import Path

import numpy as np
import pytest

import r1b_proto as P
from r1b_metrics import Scorer
from r1b_sim import Cfg, Clip, run_clip
from r1b_split import check_split
from r1b_track import Tracker

REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- helpers
def rand_state(rng, n):
    return {rng.randrange(1024): (rng.randrange(1, 11), rng.randrange(4096), rng.randrange(4096), rng.randrange(1, 400),
                                  rng.randrange(1, 400), rng.randrange(-512, 512), rng.randrange(-512, 512),
                                  rng.randrange(8)) for _ in range(n)}


def evolve(rng, st):
    st = dict(st)
    for k in list(st):
        r = rng.random()
        if r < 0.1:
            del st[k]
        elif r < 0.4:
            c, cx, cy, w, h, vx, vy, cf = st[k]
            st[k] = (c, cx + rng.randrange(-20, 21), cy + rng.randrange(-20, 21), w, h, vx + rng.randrange(-3, 4), vy, cf)
    for _ in range(rng.randrange(0, 3)):
        k = rng.randrange(1024)
        if k not in st:
            st[k] = list(rand_state(rng, 1).values())[0]
    return st


def synth_clip(seed=0, n=120, n_obj=12, W=1344, H=756, noise=2.0, miss=0.1, fp=1, bits=150_000):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([100, 100], [W - 100, H - 100], (n_obj, 2)); vel = rng.normal(0, 3, (n_obj, 2))
    size = rng.uniform(20, 60, (n_obj, 2))
    gt, dn, ds = [], [], []
    for f in range(n):
        p = pos + vel * f
        boxes = [(i, np.array([*(p[i] - size[i] / 2), *(p[i] + size[i] / 2)])) for i in range(n_obj)
                 if 0 < p[i, 0] < W and 0 < p[i, 1] < H]
        gt.append(boxes)
        def det(sd, mr, nfp):
            out = [(b + rng.normal(0, sd, 4), 4, float(rng.uniform(0.5, 0.95))) for _, b in boxes if rng.random() > mr]
            for _ in range(nfp):
                c = rng.uniform([0, 0], [W, H]); out.append((np.array([*c, *(c + 30)]), 4, float(rng.uniform(0.05, 0.45))))
            return out
        dn.append(det(noise, miss, fp)); ds.append(det(noise / 2, miss / 2, 0))
    return Clip("synth", W, H, gt, [[] for _ in range(n)], dn, ds, [bits] * n)


# ---------------------------------------------------------------- protocol tests (design sec 4)
def test_serial_arithmetic_wraparound():
    assert P.s_newer(1, 65535) and not P.s_newer(65535, 1)
    assert P.s_newer(0, 65535) and P.s_diff(2, 65534) == 4 and P.s_diff(65534, 2) == -4
    assert not P.s_newer(5, 5)
    pub = P.Publisher(); rep = P.Replica()
    pub.version = 65530; pub.msg_id = 65530
    pub.hist = {65530: (0, {}, P.crc({}, 0))}
    rep.applied = 65530
    rng = random.Random(1); st = {}
    t = 65520
    for _ in range(30):                        # versions and t_state both cross 2^16
        t = (t + 1) % P.MOD
        st = evolve(rng, st)
        m = pub.make(st, t, t)
        if m is None:
            continue
        rep.receive(m, t)
        assert not rep.desync and rep.applied == pub.version
        assert rep.state == pub.state
    assert pub.version < 100                   # wrapped


def test_replica_converges_under_loss_dup_reorder():
    """10^5 messages, 10% loss, 5% duplicates, random reordering, snapshot every 20 messages + on request."""
    rng = random.Random(7)
    pub = P.Publisher(n_hist=64); rep = P.Replica(w_hold=10)
    st = rand_state(rng, 15)
    inflight = []                              # (arrival tick, seq, msg)
    seq = 0; t = 0; checked = 0; n_msgs = 0
    while n_msgs < 100_000:
        t += 1
        st = evolve(rng, st)
        snap = (n_msgs % 20 == 0)
        m = pub.make(st, t % P.MOD, t % P.MOD, snapshot=snap)
        if m is not None:
            n_msgs += 1
            if rng.random() > 0.10:
                inflight.append((t + rng.randrange(0, 4), seq, m)); seq += 1
                if rng.random() < 0.05:
                    inflight.append((t + rng.randrange(0, 6), seq, m)); seq += 1
        inflight.sort(key=lambda x: (x[0], x[1]))
        while inflight and inflight[0][0] <= t:
            _, _, mm = inflight.pop(0)
            rep.receive(mm, t)
            if mm.snapshot and rep.applied == mm.version:
                h = pub.hist.get(rep.applied)
                if h is not None:
                    assert not rep.desync and rep.state == h[1] and rep.t_state == h[0]
                    checked += 1
        rep.tick(t)
        if pub.check_uplink(*rep.header()):    # uplink header -> snapshot request
            m2 = pub.make(st, t % P.MOD, t % P.MOD, snapshot=True)
            if m2 is not None and rng.random() > 0.10:
                inflight.append((t + 1, seq, m2)); seq += 1
    assert checked > 3000
    # whenever not desynced, the replica equals the published state of its version
    h = pub.hist.get(rep.applied)
    if h is not None and not rep.desync:
        assert rep.state == h[1]


def test_hash_detects_every_single_unit_deviation():
    rng = random.Random(3)
    for _ in range(3000):
        st = rand_state(rng, rng.randrange(1, 40)); t = rng.randrange(P.MOD)
        k = rng.choice(list(st)); f = rng.randrange(8)
        bad = dict(st); v = list(bad[k]); v[f] += rng.choice((-1, 1)); bad[k] = tuple(v)
        assert P.crc(st, t) != P.crc(bad, t)
        assert P.crc(st, t) != P.crc(st, (t + 1) % P.MOD)        # same content, other time
    # corrupted op on the wire -> replica marks desync
    pub = P.Publisher(); rep = P.Replica()
    st = rand_state(rng, 10); m = pub.make(st, 5, 5)
    k = m.ops[0][1]; v = list(m.ops[0][2]); v[1] += 1
    m.ops[0] = (m.ops[0][0], k, tuple(v))
    rep.receive(m, 5)
    assert rep.desync


def test_predictor_bit_exact_1000_ticks():
    rng = random.Random(5)
    st = rand_state(rng, 30)
    step = dict(st)
    for _ in range(1000):
        step = P.predict(step, 1)
    assert step == P.predict(st, 1000)
    assert P.crc(step, 1000) == P.crc(P.predict(st, 1000), 1000)


# ---------------------------------------------------------------- metric tests (checklist 8)
def test_metrics_gt_as_output_is_perfect():
    clip = synth_clip(1)
    sc = Scorer()
    for f in range(len(clip.gt)):
        sc.add(f, clip.gt[f], [], [(g, b) for g, b in clip.gt[f]])
    r = sc.result()
    assert r["F1"] == pytest.approx(100) and r["IDF1"] == pytest.approx(100) and r["fp"] == 0
    assert r["first_det_median_s"] == 0 and r["never_detected_frac"] == 0


def test_metrics_injected_delay():
    clip = synth_clip(2)
    sc = Scorer()
    k = 7
    for f in range(len(clip.gt)):
        sc.add(f, clip.gt[f], [], [(g, b) for g, b in clip.gt[f]] if f >= k else [])
    r = sc.result()
    assert r["first_det_median_s"] == pytest.approx(k * 0.1)
    assert r["fn"] == sum(len(g) for g in clip.gt[:k])


def test_ignored_regions_do_not_count_as_fp():
    sc = Scorer()
    sc.add(0, [], [np.array([0, 0, 100, 100])], [(1, np.array([10, 10, 40, 40])), (2, np.array([500, 500, 530, 530]))])
    assert sc.fp == 1


def test_tracker_on_gt_keeps_ids():
    clip = synth_clip(3, noise=0.0, miss=0.0, fp=0)
    tr = Tracker(clip.W, clip.H)
    sc = Scorer()
    for f in range(len(clip.gt)):
        tr.update([(b, 4, 0.9) for _, b in clip.gt[f]], f)
        sc.add(f, clip.gt[f], [], [((t.tid, t.born), t.at(f)) for t in tr.confirmed(f)])
    r = sc.result()
    assert r["F1"] > 99 and r["IDF1"] > 99


# ---------------------------------------------------------------- simulator: causality, timing, leakage
FAST = dict(R_ul=2e6, th_T=1.0, th_F=3.0, N_max=10)


@pytest.mark.parametrize("cond", ["A0", "A1", "B", "C0", "C1", "D"])
def test_no_lookahead_truncation(cond):
    """Actions at ticks < m are identical whether or not the clip continues after m."""
    clip = synth_clip(4, n=100)
    m = 60
    short = Clip("s", clip.W, clip.H, clip.gt[:m], clip.ign[:m], clip.det_n[:m], clip.det_s[:m], clip.frame_bits[:m])
    a = run_clip(clip, Cfg(cond=cond, **FAST), trace=True)["actions"]
    b = run_clip(short, Cfg(cond=cond, **FAST), trace=True)["actions"]
    assert a[:len(b) - 3] == b[:len(b) - 3]


@pytest.mark.parametrize("cond", ["A0", "A1", "B", "C0", "C1"])
def test_no_hidden_bs_state_without_feedback(cond):
    """With every downlink message lost, UAV actions cannot depend on what the BS detected (only D may)."""
    c1 = synth_clip(5, n=80)
    c2 = synth_clip(5, n=80)
    rng = np.random.default_rng(9)
    c2.det_s = [[(b + rng.normal(0, 30, 4), 2, 0.9) for b, _, _ in d][: max(0, len(d) - 3)] for d in c1.det_s]
    cfg = dict(cond=cond, p_dl=1.0, **FAST)
    a = run_clip(c1, Cfg(**cfg), trace=True)["actions"]
    b = run_clip(c2, Cfg(**cfg), trace=True)["actions"]
    assert a == b


def test_oracle_does_see_bs_state():
    c1 = synth_clip(5, n=80); c2 = synth_clip(5, n=80)
    c2.det_s = [[] for _ in c1.det_s]
    a = run_clip(c1, Cfg(cond="D", **FAST), trace=True)["actions"]
    b = run_clip(c2, Cfg(cond="D", **FAST), trace=True)["actions"]
    assert a != b


def test_serial_compute_drops_next_frame_after_encode():
    clip = synth_clip(6, n=40)
    r = run_clip(clip, Cfg(cond="A0", R_ul=5e6, th_F=0.0, th_T=0.0, c_det=26, c_enc=150, compute="serial",
                           D_max=1000), trace=True)
    acts = r["actions"]
    assert "DROP" in acts
    for i, a in enumerate(acts[:-1]):
        if a == "FRAME":
            assert acts[i + 1] == "DROP"       # 26 + 2 + 150 ms > 100 ms tick
    rp = run_clip(clip, Cfg(cond="A0", R_ul=5e6, th_F=0.0, th_T=0.0, c_det=26, c_enc=150, compute="pipelined",
                            D_max=1000), trace=True)
    assert "DROP" not in rp["actions"]


def test_deadline_drops_late_frames_but_counts_bits():
    clip = synth_clip(7, n=40, bits=400_000)
    # admission control (local backlog) refuses frames that cannot make the deadline
    r = run_clip(clip, Cfg(cond="A0", R_ul=1e6, th_F=0.0, th_T=99.0, D_max=300, frame_bits0=400_000), trace=True)
    assert r["n_frame"] == 0 and r["ul_late"] == 0
    # if the size estimate is wrong (estimate 50 kbit, real 400 kbit), late frames are dropped at the BS, bits counted
    r = run_clip(clip, Cfg(cond="A0", R_ul=1e6, th_F=0.0, th_T=99.0, D_max=300, frame_bits0=50_000), trace=True)
    assert r["n_frame"] > 0 and r["ul_late"] > 0 and r["ul_kbps"] > 0


def test_ideal_channel_frames_every_tick_track_well():
    clip = synth_clip(8, n=100, noise=0.5, miss=0.0, fp=0)
    r = run_clip(clip, Cfg(cond="D", R_ul=1e9, th_F=0.0, d_dl=0, d_ul=0, p_ul=0, p_dl=0, c_enc=1, c_bs_frame=1))
    assert r["R"] > 95 and r["P"] > 95


def test_c1_belief_beats_a1_with_good_downlink():
    """Diagnostic sanity: with a clean downlink, the C1 replica tracks the true BS state better than the A1 guess."""
    clip = synth_clip(10, n=150, bits=40_000)
    base = dict(R_ul=4e5, th_T=1.0, th_F=4.0, N_max=30, p_dl=0.0, p_ul=0.0, frame_bits0=40_000)
    a1 = run_clip(clip, Cfg(cond="A1", **base)); c1 = run_clip(clip, Cfg(cond="C1", **base))
    assert c1["belief_f1"] >= a1["belief_f1"] - 1e-9
    assert c1["desync_ticks"] == 0


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
