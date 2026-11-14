"""Synthetic tests for RQ1-E E0 (h1_rq1e.py). No dataset record, no image, no model is used."""
import numpy as np
import pytest

import h1_pilot0b as P
import h1_rq1e as E


def test_crop_window_square_min_size_and_inside():
    assert E.crop_window((100, 100, 140, 120), 1920, 1080) == (90, 80, 150, 140)        # 1.5 x 40 = 60
    x0, y0, x1, y1 = E.crop_window((5, 5, 15, 15), 1920, 1080)                          # min 32, shifted inside
    assert (x1 - x0, y1 - y0) == (32, 32) and x0 == 0 and y0 == 0
    x0, y0, x1, y1 = E.crop_window((1900, 1060, 1920, 1080), 1920, 1080)
    assert x1 == 1920 and y1 == 1080 and x1 - x0 == 32


def test_preprocess_shape_and_normalisation():
    img = np.full((100, 100, 3), 255, np.uint8)
    x = E.preprocess(img, (10, 10, 42, 42))
    assert x.shape == (3, 224, 224)
    assert np.allclose(x[:, 0, 0], (1 - E.MEAN) / E.STD, atol=1e-5)


def test_compression_is_deterministic_and_distance_kinds():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 384)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert np.array_equal(E.compress(X, 128), E.compress(X, 128))
    b = E.compress(X, 64)
    assert b.dtype == bool and b.shape == (5, 64)
    assert np.allclose(np.diag(E.distance(b, b)), 0)
    q = E.compress(X, 3072)
    assert q.dtype == np.int8 and np.allclose(np.diag(E.distance(q, q)), 0, atol=1e-4)
    assert np.allclose(np.diag(E.distance(X, X)), 0, atol=1e-6)


def test_candidates_mutual_nn_ratio_and_class():
    D = np.array([[0.1, 0.9, 0.9], [0.9, 0.12, 0.13], [0.9, 0.9, 0.5]])
    c = np.array([1, 1, 1])
    cand = E.candidates(D, c, c)
    assert (0, 0) in cand                       # clear match
    assert (1, 1) not in cand                   # 0.12 vs 0.13 fails the ratio test
    assert E.candidates(D, np.array([1, 1, 2]), np.array([2, 2, 1]))[:1] != [(0, 0)]   # class mismatch blocks (0,0)


def _pts(n, rng):
    return rng.uniform([100, 100], [1700, 950], (n, 2))


def _dets(pts):
    return P.dets([(x - 20, y - 30, x + 20, y) for x, y in pts], [1] * len(pts), [0.9] * len(pts))


def test_register_valid_with_correct_candidates_and_rejects_random_ones():
    rng = np.random.default_rng(1)
    p = _pts(20, rng)
    Hs = np.array([[0.9, -0.2, 100.0], [0.2, 0.9, -40.0], [0, 0, 1.0]])
    q = (np.c_[p, np.ones(20)] @ Hs.T)[:, :2]
    d1, d2 = _dets(p), _dets(q)
    H, info = E.register(d1, d2, [(k, k) for k in range(20)])
    assert info["valid"] and np.allclose(H, Hs, atol=1e-3)
    perm = rng.permutation(20)
    H, info = E.register(d1, d2, [(k, int(perm[k])) for k in range(20)])
    assert not info["valid"]
    H, info = E.register(d1, d2, [(k, k) for k in range(5)])
    assert info["why"] == "too few candidates"


def _row(valid=True, diag=10.0, n_cand=20, n_verif=20, n_correct=18, n_ideal=40):
    return dict(valid=valid, why="" if valid else "x", diag=diag if valid else None, n_cand=n_cand, n_verif=n_verif,
                n_correct=n_correct, n_ideal=n_ideal, n_final=10, n_final_ok=9)


def test_e0_pass_rule_each_criterion():
    assert E.summarize([_row() for _ in range(4)])["pass"]
    assert not E.summarize([_row(), _row(False), _row(False), _row(False)])["pass"]        # valid share 25 %
    assert not E.summarize([_row(diag=60.0) for _ in range(4)])["pass"]                    # diag > 40 px
    assert not E.summarize([_row(diag=150.0)] + [_row() for _ in range(5)])["pass"]        # 1/6 false registrations
    assert not E.summarize([_row(n_correct=14) for _ in range(4)])["pass"]                 # precision 0.70
    assert not E.summarize([_row(n_correct=8, n_verif=8, n_ideal=40) for _ in range(4)])["pass"]   # recall 0.20


def test_decision_and_primary_width():
    ok, bad = dict(**{"pass": True}), dict(**{"pass": False})
    assert E.e0_decision({s: ok for s in range(6)} | {6: bad, 7: bad})["verdict"] == "PASS"
    assert E.e0_decision({s: ok for s in range(5)} | {s: bad for s in range(5, 8)})["verdict"] == "FAIL"
    pw = {64: {s: bad for s in range(8)}, 128: {s: ok for s in range(8)}, 256: {s: ok for s in range(8)},
          3072: {s: ok for s in range(8)}}
    assert E.primary_width(pw) == 128
    assert E.primary_width({m: {s: bad for s in range(8)} for m in E.WIDTHS}) is None


def test_declared_with_source_maps_to_original_boxes():
    det = P.dets([(100, 100, 140, 130), (500, 500, 540, 530), (900, 100, 930, 120)], [1, 1, 1], [0.9, 0.1, 0.8])
    rec = dict(W=1920, H=1080, det=det, det_ms=1.0, cands=dict(P1=[], P2=[]), frame=dict(bits=1, enc_ms=1.0, bs=P.dets()))
    d, src = E.declared_with_source(rec)
    assert len(d["s"]) == 2 and sorted(src.tolist()) == [0, 2]                            # weak detection not declared
