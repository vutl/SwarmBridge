"""Synthetic tests for B-E1 (b1_pipeline.py)."""
import numpy as np
import pytest

import b1_pipeline as P

N, GH, GW = P.N_TOK, P.GH, P.GW


@pytest.mark.parametrize("k", [0, 1, 3, 7])
def test_eg_round_trip(k):
    v = np.r_[np.random.default_rng(k).integers(-3000, 3000, 500), 0, 1, -1]
    bits = P.eg_encode(v, k)
    assert len(bits) == P.eg_len(v, k)
    out, end = P.eg_decode(bits, 0, len(v), k)
    assert np.array_equal(out, v) and end == len(bits)


@pytest.mark.parametrize("kk,mv", [(0, None), (5, None), (300, (2, -1)), (N, (-4, 4))])
def test_message_round_trip(kk, mv):
    rng = np.random.default_rng(kk)
    idx = np.sort(rng.choice(N, kk, replace=False))
    rq = rng.integers(-20, 20, (kk, 32))
    data, nb = P.encode_msg(9, idx, rq, mv)
    f, idx2, rq2, mv2 = P.decode_msg(data, nb)
    assert f == 9 and np.array_equal(idx2, idx) and np.array_equal(rq2, rq) and mv2 == mv


def test_shift_map():
    m = np.arange(N * 2, dtype=float).reshape(N, 2)
    s = P.shift_map(m, 1, 2).reshape(GH, GW, 2)
    g = m.reshape(GH, GW, 2)
    assert np.array_equal(s[5, 5], g[3, 4]) and np.array_equal(s[0, 0], g[0, 0])


def _panning_clip(T=8, d=16, seed=0):
    rng = np.random.default_rng(seed)
    world = rng.normal(size=(GH + 20, GW + 30, d)).astype(np.float32)
    return np.stack([world[t:t + GH, 2 * t:2 * t + GW].reshape(N, d) for t in range(T)])   # camera pans (dx=-2, dy=-1)


def test_motion_compensation_beats_hold_under_panning():
    Z = _panning_clip()
    rh, bh = P.transmit(Z, "hold", 200, 0.25)
    rm, bm = P.transmit(Z, "mc", 200, 0.25)
    eh = np.abs(rh[1:] - Z[1:]).mean()
    em = np.abs(rm[1:] - Z[1:]).mean()
    assert em < eh and bm[1:].mean() < bh[1:].mean()


def test_closed_loop_reconstruction_error_bounded_by_step():
    Z = _panning_clip()
    r, _ = P.transmit(Z, "hold", N, 0.25)                         # every patch every frame
    assert np.abs(r - Z).max() <= 0.125 + 1e-5


def test_completion_mode_stays_synchronised():
    def dummy(cur, mask, prev):            # deterministic stand-in for the trained completer
        out = cur.copy()
        out[~mask] = 0.5 * (cur[~mask] + prev[~mask])
        return out
    Z = _panning_clip()
    r, b = P.transmit(Z, "complete", 50, 0.25, completer=dummy)   # asserts receiver == replica every frame
    assert r.shape == Z.shape and (b[1:] < b[0]).all()
