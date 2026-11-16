"""Tests for the B-E7 video baselines (b3_video.py): round trip, bit accounting, causality of the low-latency setup."""
import tempfile

import numpy as np
import pytest

import b3_video as B3


def _frames(T=12, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (B3.H_IMG + 40, B3.W_IMG + 40, 3), dtype=np.uint8)
    base = (base.astype(np.float32) * 0.3 + 90).astype(np.uint8)
    # smooth-ish moving content: a shifted textured background plus a moving bright block
    out = []
    for t in range(T):
        f = base[2 * t % 40: 2 * t % 40 + B3.H_IMG, t % 40: t % 40 + B3.W_IMG].copy()
        f[50 + 5 * t: 90 + 5 * t, 100 + 8 * t: 160 + 8 * t] = 230
        out.append(f)
    return np.stack(out)


@pytest.mark.parametrize("codec,q", [("hevc", 32), ("av1", 45)])
def test_round_trip_and_bits(codec, q):
    fr = _frames()
    with tempfile.TemporaryDirectory() as tmp:
        dec, tot, pk, _ = B3.code_video(fr, codec, q, tmp)
    assert dec.shape == fr.shape
    assert np.mean(np.abs(dec.astype(float) - fr)) < 20
    assert tot > 0 and abs(pk.sum() - tot) <= 0.02 * tot + 800   # packets cover (almost) the whole stream
    assert pk[0] > np.median(pk[1:])                               # intra first frame, cheaper inter frames


@pytest.mark.parametrize("codec,q", [("hevc", 32), ("av1", 45)])
def test_future_frames_do_not_affect_past(codec, q):
    """low-latency configuration: changing frames > t0 leaves decoded frames <= t0 and their packets unchanged."""
    fr = _frames()
    alt = fr.copy()
    t0 = 6
    alt[t0 + 1:] = 255 - alt[t0 + 1:]
    with tempfile.TemporaryDirectory() as tmp:
        d1, _, p1, _ = B3.code_video(fr, codec, q, tmp)
        d2, _, p2, _ = B3.code_video(alt, codec, q, tmp)
    assert np.array_equal(d1[:t0 + 1], d2[:t0 + 1])
    assert np.array_equal(p1[:t0 + 1], p2[:t0 + 1])
    assert not np.array_equal(d1[t0 + 1:], d2[t0 + 1:])


def test_interp_log_and_ci():
    assert B3.interp_log([1000, 4000], [0.2, 0.4], 2000) == pytest.approx(0.3)
    assert np.isnan(B3.interp_log([1000, 4000], [0.2, 0.4], 500))
    c = B3.ci([0.1, 0.2, 0.3])
    assert c["mean"] == pytest.approx(0.2) and c["lo"] < 0.2 < c["hi"] and c["pos"] == 3
