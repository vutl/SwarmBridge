"""Synthetic tests for the RQ1 association feasibility test (h1_rq1_assoc.py). No dataset record is used."""
import numpy as np
import pytest

import h1_pilot0b as P
import h1_rq1_assoc as R

I3 = np.eye(3)


def box(x, y, w=40, h=30):
    return (x, y, x + w, y + h)


def D(*boxes, c=1):
    return P.dets(list(boxes), [c] * len(boxes), [0.9] * len(boxes))


def G(**kw):
    return {int(k[1:]): (*b, 1) for k, b in kw.items()}


def test_ideal_fusion_equals_b1_unique_counts():
    g1 = G(i1=box(100, 100), i2=box(500, 500))
    g2 = G(i1=box(100, 100), i3=box(900, 300))
    d1, d2 = D(box(100, 100), box(500, 500), box(1500, 50)), D(box(100, 100), box(900, 300))
    f = R.fused_counts(d1, d2, g1, g2, R.ideal_pairs(d1, d2, g1, g2))
    u = P.unique_counts(P.view_counts(d1, g1, 0.0), P.view_counts(d2, g2, 0.0), g1, g2)
    assert (f["tp"], f["fp"], f["fn"]) == (u["tp"], u["fp"], u["fn"]) == (3, 1, 0)
    assert f["fp_dup"] == f["fp_assoc"] == 0


def test_no_association_double_counts_shared_objects():
    g = G(i1=box(100, 100))
    d = D(box(100, 100))
    f = R.fused_counts(d, d, g, g, [])
    assert (f["tp"], f["fp_dup"], f["fp"]) == (1, 1, 1)


def test_geometric_association_with_exact_homography_matches_ideal():
    g1 = G(i1=box(100, 100), i2=box(600, 400), i3=box(1200, 700))
    d1 = d2 = D(box(100, 100), box(600, 400), box(1200, 700))
    pairs = R.associate(R.distances(d1, d2, I3))
    assert sorted(pairs) == [(0, 0), (1, 1), (2, 2)]
    assert R.fused_counts(d1, d2, g1, g1, pairs) == R.fused_counts(d1, d2, g1, g1, R.ideal_pairs(d1, d2, g1, g1))


def test_wrong_merge_loses_an_object_and_is_flagged():
    g1, g2 = G(i1=box(100, 100)), G(i2=box(110, 100))       # different objects at almost the same place
    d1, d2 = D(box(100, 100)), D(box(110, 100))
    pairs = R.associate(R.distances(d1, d2, I3))
    assert pairs == [(0, 0)]
    f = R.fused_counts(d1, d2, g1, g2, pairs)
    assert (f["tp"], f["fn"], f["fp_assoc"]) == (1, 1, 1)
    assert R.fused_counts(d1, d2, g1, g2, R.ideal_pairs(d1, d2, g1, g2))["tp"] == 2


def test_class_mismatch_never_merges_and_far_pairs_are_gated():
    d1 = P.dets([box(100, 100)], [1], [0.9])
    d2 = P.dets([box(100, 100)], [2], [0.9])
    assert R.associate(R.distances(d1, d2, I3)) == []
    assert R.associate(R.distances(D(box(100, 100)), D(box(100 + R.MERGE_PX + 5, 100)), I3)) == []
    assert R.associate(R.distances(D(box(100, 100)), D(box(100, 100)), None)) == []


def test_ambiguity_rule():
    # one close candidate -> not ambiguous; one candidate at 30 px -> ambiguous; two candidates within AMB_HI -> ambiguous
    A1, A2 = R.ambiguous(R.distances(D(box(100, 100)), D(box(105, 100)), I3))
    assert A1 == set() and A2 == set()
    A1, A2 = R.ambiguous(R.distances(D(box(100, 100)), D(box(130, 100)), I3))
    assert A1 == {0} and A2 == {0}
    A1, A2 = R.ambiguous(R.distances(D(box(100, 100)), D(box(102, 100), box(160, 100)), I3))
    assert A1 == {0}


def test_oracle_fixes_only_ambiguous_detections():
    # view 1: objects 1 (alone) and 2; view 2: object 2 shifted 30 px and a different object 3 next to it
    g1 = G(i1=box(100, 100), i2=box(600, 400))
    g2 = G(i1=box(100, 100), i2=box(630, 400), i3=box(650, 430))
    d1 = D(box(100, 100), box(600, 400))
    d2 = D(box(100, 100), box(630, 400), box(650, 430))
    Dm = R.distances(d1, d2, I3)
    pairs = R.associate(Dm)
    A1, A2 = R.ambiguous(Dm)
    assert 0 not in A1 and 1 in A1
    ideal = R.ideal_pairs(d1, d2, g1, g2)
    orc = R.oracle_pairs(pairs, ideal, A1, A2)
    assert (0, 0) in orc and sorted(orc) == sorted(ideal)
    assert R.fused_counts(d1, d2, g1, g2, orc) == R.fused_counts(d1, d2, g1, g2, ideal)


def test_homography_needs_min_common_ids_and_recovers_a_shift():
    pts = [(100 + 150 * k, 100 + 80 * (k % 4)) for k in range(R.MIN_COMMON)]
    g1 = {k: (*box(*p), 1) for k, p in enumerate(pts)}
    g2 = {k: (*box(p[0] + 50, p[1] - 20), 1) for k, p in enumerate(pts)}
    H = R.homography(g1, g2)
    assert H is not None
    assert np.allclose(R.project(H, np.array([[300.0, 300.0]])), [[350.0, 280.0]], atol=1.0)
    assert R.homography({k: g1[k] for k in range(R.MIN_COMMON - 1)}, g2) is None


def _rec(boxes, frame_boxes=(), frame_bits=400000, p1_bits=(20000, 30000)):
    det = P.dets(list(boxes), [1] * len(boxes), [0.9] * len(boxes))
    cands = [dict(bits={35: b, 50: b}) for b in p1_bits]
    return dict(W=1920, H=1080, det=det, det_ms=60.0, cands=dict(P1=cands, P2=[]),
                frame=dict(bits=frame_bits, enc_ms=1000.0,
                           bs=P.dets(list(frame_boxes), [1] * len(frame_boxes), [0.8] * len(frame_boxes))))


def _seq(n=6, shift_second_half=0.0):
    """view 2 = view 1 shifted by (50, -20) px; 8 shared objects; in the second half view 2 drifts by an extra amount."""
    pts = [(100 + 180 * k, 150 + 90 * (k % 4)) for k in range(R.MIN_COMMON)]
    out = []
    for t in range(n):
        s = shift_second_half if t >= n // 2 else 0.0
        b1 = [box(*p) for p in pts]
        b2 = [box(p[0] + 50 + s, p[1] - 20) for p in pts]
        g1 = {k: (*b, 1) for k, b in enumerate(b1)}
        g2 = {k: (*b, 1) for k, b in enumerate(b2)}
        out.append((_rec(b1), _rec(b2), g1, g2))
    return out


def test_run_sequence_stale_equals_ideal_without_drift_and_degrades_with_drift():
    a = R.run_sequence(_seq(), frame_view=1)
    m = a["modes"]["TT"]
    assert a["stale_available"] and a["crop_bits"] == pytest.approx(25000)
    assert m["stale"]["f1"] == pytest.approx(m["ideal"]["f1"]) == pytest.approx(m["fresh"]["f1"])
    assert m["none"]["f1"] < m["ideal"]["f1"] and m["none"]["fp_dup"] > 0
    d = R.run_sequence(_seq(shift_second_half=120.0), frame_view=1)["modes"]["TT"]
    assert d["fresh"]["f1"] == pytest.approx(d["ideal"]["f1"])           # fresh calibration follows the drift
    assert d["stale"]["f1"] < d["ideal"]["f1"]                          # stale calibration does not
    assert d["stale_oracle"]["f1"] >= d["stale"]["f1"]


def test_oracle_is_never_free():
    seq = [(dict(r1, cands=dict(P1=[], P2=[])), dict(r2, cands=dict(P1=[], P2=[])), g1, g2) for r1, r2, g1, g2 in _seq()]
    r = R.run_sequence(seq, frame_view=1)
    assert r["crop_bits"] == float("inf")


def test_stale_calibration_uses_first_half_labels_only():
    seq = _seq(shift_second_half=120.0)
    blind_first = [(r1, r2, {}, {}) for r1, r2, _, _ in seq[:3]] + seq[3:]
    r = R.run_sequence(blind_first, frame_view=1)
    assert not r["stale_available"]                                    # no first-half labels -> no calibration
    assert r["modes"]["TT"]["stale"]["f1"] == pytest.approx(r["modes"]["TT"]["none"]["f1"])


def _per(dA, rec, extra=1000.0, fbits=400000.0, n=8):
    out = {}
    for s in range(n):
        d = dA[s] if isinstance(dA, list) else dA
        m = {"ideal": dict(f1=0.70), "stale": dict(f1=0.70 - d), "stale_oracle": dict(f1=0.70 - d + rec * d,
                                                                                      extra_bits_per_epoch=extra)}
        out[s] = dict(frame_bits=fbits, modes={"TT": m})
    return out


def test_gate_rule():
    assert R.gate(_per(0.05, 0.6))["verdict"] == "GO"
    assert R.gate(_per(0.005, 1.0))["verdict"] == "NO-GO"
    assert R.gate(_per(0.05, 0.3))["verdict"] == "INCONCLUSIVE"             # oracle recovers too little
    assert R.gate(_per(0.05, 0.6, extra=50000.0))["verdict"] == "INCONCLUSIVE"   # costs > 10 % of a FRAME
    assert R.gate(_per(0.02, 0.9))["verdict"] == "INCONCLUSIVE"             # loss between 1 and 3 points
    few = [0.24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                      # large mean, 1/8 sequences
    assert R.gate(_per(few, 0.9))["verdict"] == "INCONCLUSIVE"


def test_main_requires_data_use_basis(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["h1_rq1_assoc.py", "p", "i", "b", "o", "--zip", "z"])
    with pytest.raises(SystemExit):
        R.main()
