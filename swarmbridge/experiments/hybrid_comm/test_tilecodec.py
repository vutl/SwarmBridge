"""Unit tests for tilecodec (numpy only): python -I -m pytest test_tilecodec.py (from this directory)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilecodec as tc  # noqa: E402


def _feat(rng, c=80):
    return rng.normal(0, 1, (c, tc.G, tc.G)).astype(np.float32)


def test_tiles_layout_matches_scatter():
    rng = np.random.default_rng(0)
    f = _feat(rng, 4)
    tl = tc.tiles_of(f)
    assert tl.shape == (tc.NT * tc.NT, 4, tc.T, tc.T)
    idx = np.array([0, 33, 1023])
    dense, mask = tc.scatter(idx, tl[idx], c=4)
    for i in idx:
        r, col = divmod(i, tc.NT)
        sl = (slice(None), slice(r * 8, r * 8 + 8), slice(col * 8, col * 8 + 8))
        assert np.array_equal(dense[sl], f[sl])
    assert mask.sum() == 3 * 64


def test_roundtrip_error_and_bytes():
    rng = np.random.default_rng(1)
    f = _feat(rng)
    tl = tc.tiles_of(f)
    idx = rng.permutation(tc.NT * tc.NT)[:64]
    msg = tc.encode(tl[idx], idx, sender=3, t_sense_ms=1200, msg_id=7)
    assert len(msg) == tc.message_bytes(64)
    meta, di, dt = tc.decode(msg)
    assert meta["sender"] == 3 and meta["t_sense_ms"] == 1200 and meta["msg_id"] == 7 and meta["n_planned"] == 64
    assert np.array_equal(di, idx)
    span = tl[idx].max(axis=(0, 2, 3)) - tl[idx].min(axis=(0, 2, 3))
    err = np.abs(dt - tl[idx]).max(axis=(0, 2, 3))
    assert np.all(err <= span / 255 * 0.5 + span * 2e-3 + 1e-3)        # half a step + fp16 parameter rounding


def test_prefix_is_valid_message_and_m2_prefix_of_m3():
    rng = np.random.default_rng(2)
    tl = tc.tiles_of(_feat(rng))
    idx = np.arange(64)
    m3 = tc.encode(tl[idx], idx)
    m2 = tc.prefix(m3, 16)
    assert m3[:len(m2)] == m2 and len(m2) == tc.message_bytes(16)
    _, i2, t2 = tc.decode(m2)
    _, i3, t3 = tc.decode(m3)
    assert np.array_equal(i2, i3[:16]) and np.array_equal(t2, t3[:16])


def test_truncated_message_uses_complete_patches_only():
    rng = np.random.default_rng(3)
    tl = tc.tiles_of(_feat(rng))
    idx = np.arange(10)
    m = tc.encode(tl[idx], idx)
    cut = tc.message_bytes(4) + 100                          # 4 full patches + part of the 5th
    _, i, t = tc.decode(m[:cut])
    assert len(i) == 4 and t.shape[0] == 4
    meta, i, t = tc.decode(m[:tc.header_bytes() - 1])       # header without qparams -> nothing usable
    assert len(i) == 0
    assert tc.decode(m[:5])[0] is None                       # not even a header


def test_constant_channel_and_saturation():
    rng = np.random.default_rng(4)
    f = _feat(rng, 3)
    f[1] = 2.5                                               # constant channel
    f[2, :8, :8] = 1e4                                       # extreme value inside the first tile
    tl = tc.tiles_of(f)
    idx = np.array([0, 1, 2])
    _, _, dt = tc.decode(tc.encode(tl[idx], idx))
    assert np.allclose(dt[:, 1], 2.5, atol=1e-2)
    assert np.all(np.isfinite(dt)) and dt[0, 2].max() > 9e3


def test_tile_scores_pooling():
    s = np.zeros((128, 128), np.float32)
    s[5, 70] = 1.0                                           # half resolution: cell (10, 140) -> tile (1, 17)
    ts = tc.tile_scores(s)
    assert ts.argmax() == 1 * tc.NT + 17 and ts.max() == 1.0
