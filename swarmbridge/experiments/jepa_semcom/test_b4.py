"""Tests for B-E8 (b4_roi.py): x265 adapter bit accounting / causality / effect of offsets, causal importance maps,
rank allocation."""
import tempfile

import numpy as np
import pytest

import b3_video as B3
import b4_roi as R


def _frames(T=10, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (R.H_IMG + 60, R.W_IMG + 60, 3), dtype=np.uint8)
    base = (base.astype(np.float32) * 0.3 + 90).astype(np.uint8)
    out = []
    for t in range(T):
        f = base[3 * t % 60: 3 * t % 60 + R.H_IMG, 2 * t % 60: 2 * t % 60 + R.W_IMG].copy()
        f[50 + 5 * t: 90 + 5 * t, 100 + 8 * t: 160 + 8 * t] = 230
        out.append(f)
    return np.stack(out)


def test_adapter_bits_and_match_with_ffmpeg_x265():
    fr = _frames()
    with tempfile.TemporaryDirectory() as tmp:
        d, tot, pk, _ = R.code_roi(fr, 32, None, tmp)
        d3, tot3, _, _ = B3.code_video(fr, "hevc", 32, tmp)
    assert d.shape == fr.shape and tot == pk.sum() and pk[0] > np.median(pk[1:])
    assert abs(tot - tot3) <= 0.03 * tot3                        # same encoder setting as the B-E7 ffmpeg path
    assert np.mean(np.abs(d.astype(float) - d3)) < 0.5


def test_adapter_is_causal_in_frames_and_offsets():
    fr = _frames()
    q = np.zeros((len(fr), R.BH, R.BW), np.float32)
    q[:, :5, :10] = -6
    q[:, 5:, :] = 2
    t0 = 5
    fr2, q2 = fr.copy(), q.copy()
    fr2[t0 + 1:] = 255 - fr2[t0 + 1:]
    q2[t0 + 1:] = -q2[t0 + 1:]
    with tempfile.TemporaryDirectory() as tmp:
        d1, _, p1, _ = R.code_roi(fr, 34, q, tmp)
        d2, _, p2, _ = R.code_roi(fr2, 34, q2, tmp)
    assert np.array_equal(d1[:t0 + 1], d2[:t0 + 1]) and np.array_equal(p1[:t0 + 1], p2[:t0 + 1])
    assert not np.array_equal(d1[t0 + 1:], d2[t0 + 1:])


def test_negative_offsets_improve_the_region():
    fr = _frames()
    q = np.zeros((len(fr), R.BH, R.BW), np.float32)
    q[:, 2:8, 2:12] = -8
    with tempfile.TemporaryDirectory() as tmp:
        du, tu, _, _ = R.code_roi(fr, 38, None, tmp)
        dr, tr, _, _ = R.code_roi(fr, 38, q, tmp)
    sl = (slice(None), slice(32, 128), slice(32, 192))
    err = lambda d: np.mean((d[sl].astype(float) - fr[sl]) ** 2)
    assert err(dr) < 0.9 * err(du) and tr > tu         # (real VisDrone frames: -12 at CRF 38 ~ CRF 26 in the region)


def test_offsets_rank_allocation():
    imp = np.zeros((3, R.BH, R.BW), np.float32)
    imp[1, 0, :5] = 0.9                      # 5 eligible blocks
    imp[2] = np.linspace(0, 1, R.BH * R.BW).reshape(R.BH, R.BW)
    q = R.offsets(imp, 6.0, 0.2)
    n = R.BH * R.BW
    assert np.all(q[0] == 0)                                  # nothing eligible -> no offsets
    assert (q[1] == -6).sum() == 5 and np.all(q[1, 0, :5] == -6) and abs(q[1].mean()) < 1e-5
    k = int(round(0.2 * n))
    assert (q[2] == -6).sum() == k and np.all(q[2].reshape(-1)[-k:] == -6) and abs(q[2].mean()) < 1e-5


def test_global_shift_and_importance_causal():
    fr = _frames(8)
    g = R.gray(fr.astype(np.float32))
    a = g[0]
    b = R.shift_img(a, 3, -5)
    assert R.global_shift(a, b) == (3, -5)
    rng = np.random.default_rng(1)
    sc = rng.normal(size=(len(fr), R.GH * R.GW, 4)).astype(np.float32)
    t0 = 4
    fr2, sc2 = fr.copy(), sc.copy()
    fr2[t0 + 1:] = 255 - fr2[t0 + 1:]
    sc2[t0 + 1:] += 3
    sh1, g1 = R.shifts(fr)
    sh2, g2 = R.shifts(fr2)
    assert sh1[:t0 + 1] == sh2[:t0 + 1]
    assert np.array_equal(R.imp_motion(fr, sh1, g1)[:t0 + 1], R.imp_motion(fr2, sh2, g2)[:t0 + 1])
    assert np.array_equal(R.imp_taskev(sc, sh1)[:t0 + 1], R.imp_taskev(sc2, sh2)[:t0 + 1])
    assert np.array_equal(R.imp_task(sc)[:t0 + 1], R.imp_task(sc2)[:t0 + 1])
    assert np.array_equal(R.imp_event(sc, sh1)[:t0 + 1], R.imp_event(sc2, sh2)[:t0 + 1])


def test_event_map_sees_class_swap_and_quota_is_disjoint():
    T, N = 2, R.GH * R.GW
    sc = np.full((T, N, 4), -6.0, np.float32)
    sc[0, 100, 0] = 2.2                      # car 0.9 ...
    sc[1, 100, 1] = 2.2                      # ... -> person 0.9 at the same patch (no camera motion)
    sh = [(0, 0), (0, 0)]
    ev_old = R.imp_taskev(sc, sh)[1] - R.imp_task(sc)[1]
    assert ev_old.max() < 1e-6                                # old map: no change signal
    e = R.imp_event(sc, sh)
    assert np.all(e[0] == 0) and e[1].max() > 0.8             # per-class map: ~0.9 at the swapped patch
    base = np.zeros((1, R.BH, R.BW), np.float32)
    base[0, 0, :10] = 0.9
    ev = np.zeros((1, R.BH, R.BW), np.float32)
    ev[0, 0, 5:20] = 0.5                                      # overlaps the task set on 5 blocks
    st = {}
    q = R.offsets_quota(base, ev, 4.0, 10 / (R.BH * R.BW), 8 / (R.BH * R.BW), st)
    assert (q[0] == -4).sum() == 18 and np.all(q[0, 0, :18] == -4) and abs(q[0].mean()) < 1e-5
    assert abs(st["top_event_overlap_with_task"] - 5 / 8) < 1e-9 and abs(st["event_share_of_selected"] - 8 / 18) < 1e-9
    assert R.cfg_key("evq", 4.0, 0.15, 0.05) == "evq_A4_f0.15_q0.05" and R.cfg_key("task", 4.0, 0.15) == "task_A4_f0.15"
