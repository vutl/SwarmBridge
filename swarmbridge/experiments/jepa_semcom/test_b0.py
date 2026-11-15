"""Synthetic tests for B-E0 (b0_pipeline.py). No dataset record is used."""
import numpy as np
import pytest

import b0_pipeline as B


def _clip(T=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, B.N_TOK, B.DIM)).astype(np.float32)
    for t in range(1, T):                       # mostly static scene, a few changing patches
        x[t] = x[t - 1]
        ch = rng.choice(B.N_TOK, 20, replace=False)
        x[t, ch] += rng.normal(size=(20, B.DIM)) * 3
    return x.astype(np.float16)


@pytest.mark.parametrize("q", [2, 4, 8])
@pytest.mark.parametrize("k", [0, 1, 5, 200, B.N_TOK])
def test_message_round_trip(k, q):
    rng = np.random.default_rng(k + q)
    idx = np.sort(rng.choice(B.N_TOK, k, replace=False))
    tok = rng.normal(size=(k, B.DIM)).astype(np.float32)
    data, nb, idx_tx, dec_tx = B.encode_msg(7, idx, tok, q)
    f, idx_rx, dec_rx = B.decode_msg(data, nb)
    assert f == 7 and np.array_equal(idx_rx, idx) and np.array_equal(dec_rx, dec_tx)
    assert len(data) == (nb + 7) // 8
    # bit count = header + indices + k * (16 + 384 q)
    assert nb >= 32 + k * (16 + B.DIM * q)


def test_index_coding_picks_shorter_form():
    few = B.encode_msg(0, [3, 10, 500], np.zeros((3, B.DIM), np.float32), 4)[1]
    many = B.encode_msg(0, np.arange(0, B.N_TOK, 2), np.zeros((B.N_TOK // 2, B.DIM), np.float32), 4)[1]
    payload = lambda k: k * (16 + B.DIM * 4)
    assert few - payload(3) < 100                       # exp-Golomb deltas for a sparse set
    assert many - payload(B.N_TOK // 2) <= 32 + B.N_TOK  # bitmap for a dense set


def test_quantiser_error_shrinks_with_bits():
    x = np.random.default_rng(1).normal(size=(50, B.DIM)).astype(np.float32)
    e = [np.abs(B.dequantise(*B.quantise(x, q)) - x).mean() for q in (2, 4, 8)]
    assert e[0] > e[1] > e[2]


def test_receiver_synchronised_and_innovation_selected():
    x = _clip()
    rx, bits = B.transmit_clip(x, k=20, q=8)               # asserts sender replica == receiver every frame
    assert bits[0] > bits[1:].max()                         # initial message carries every patch
    # with k = number of changed patches, the receiver tracks the scene up to quantisation error
    err = np.abs(rx[-1].astype(np.float32) - x[-1].astype(np.float32)).mean()
    assert err < 0.05
    rx0, _ = B.transmit_clip(x, k=0, q=8)                   # no innovation: frozen first frame
    err0 = np.abs(rx0[-1].astype(np.float32) - x[-1].astype(np.float32)).mean()
    assert err0 > 5 * err


def test_patch_labels_centre_rule_and_ignore():
    # frame 1092 x 616 -> scale 0.5 to 546 x 308; one car box and an ignored region
    ann = np.array([[1, 1, 100, 100, 56, 56, 1, 4, 0, 0],      # car, covers patch centres near (50..78, 50..78)/14
                    [1, 2, 0, 400, 200, 200, 0, 0, 0, 0]], float)  # ignored region
    lab, valid = B.patch_labels(ann, (616, 1092))
    L = lab.reshape(B.GH, B.GW, 4)
    assert L[..., 1].sum() >= 1 and L[..., [0, 2, 3]].sum() == 0
    assert not valid.reshape(B.GH, B.GW)[15, 2]               # inside the ignored region
