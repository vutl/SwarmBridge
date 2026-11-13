"""Synthetic tests for RQ1 v2 (h1_rq1_assoc.py). No dataset record is used."""
import inspect

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


# ---------------------------------------------------------------- counting
def test_ideal_fusion_equals_b1_unique_counts():
    g1 = G(i1=box(100, 100), i2=box(500, 500))
    g2 = G(i1=box(100, 100), i3=box(900, 300))
    d1, d2 = D(box(100, 100), box(500, 500), box(1500, 50)), D(box(100, 100), box(900, 300))
    f = R.fused_counts(d1, d2, g1, g2, R.ideal_pairs(d1, d2, g1, g2))
    u = P.unique_counts(P.view_counts(d1, g1, 0.0), P.view_counts(d2, g2, 0.0), g1, g2)
    assert (f["tp"], f["fp"], f["fn"]) == (u["tp"], u["fp"], u["fn"]) == (3, 1, 0)
    assert f["fp_dup"] == f["fp_assoc"] == 0


def test_missed_merge_is_a_duplicate_and_wrong_merge_loses_an_object():
    g = G(i1=box(100, 100))
    f = R.fused_counts(D(box(100, 100)), D(box(100, 100)), g, g, [])
    assert (f["tp"], f["fp_dup"], f["fp"]) == (1, 1, 1)
    g1, g2 = G(i1=box(100, 100)), G(i2=box(110, 100))
    d1, d2 = D(box(100, 100)), D(box(110, 100))
    w = R.fused_counts(d1, d2, g1, g2, R.associate(R.distances(d1, d2, I3)))
    assert (w["tp"], w["fn"], w["fp_assoc"]) == (1, 1, 1)


def test_association_respects_class_and_gate():
    assert R.associate(R.distances(P.dets([box(1, 1)], [1], [.9]), P.dets([box(1, 1)], [2], [.9]), I3)) == []
    assert R.associate(R.distances(D(box(100, 100)), D(box(100 + R.MERGE_PX + 5, 100)), I3)) == []
    assert R.associate(R.distances(D(box(100, 100)), D(box(100, 100)), None)) == []


def test_ambiguity_rule():
    assert R.ambiguous(R.distances(D(box(100, 100)), D(box(105, 100)), I3)) == (set(), set())
    assert R.ambiguous(R.distances(D(box(100, 100)), D(box(130, 100)), I3)) == ({0}, {0})
    assert R.ambiguous(R.distances(D(box(100, 100)), D(box(102, 100), box(160, 100)), I3))[0] == {0}


def test_oracle_fixes_only_ambiguous_detections():
    g1 = G(i1=box(100, 100), i2=box(600, 400))
    g2 = G(i1=box(100, 100), i2=box(630, 400), i3=box(650, 430))
    d1, d2 = D(box(100, 100), box(600, 400)), D(box(100, 100), box(630, 400), box(650, 430))
    Dm = R.distances(d1, d2, I3)
    A1, A2 = R.ambiguous(Dm)
    assert 0 not in A1 and 1 in A1
    ideal = R.ideal_pairs(d1, d2, g1, g2)
    orc = R.oracle_pairs(R.associate(Dm), ideal, A1, A2)
    assert sorted(orc) == sorted(ideal)


# ---------------------------------------------------------------- GT-free registration
def _scene(rng, n=14, A=None, distract=4, jitter=1.5):
    pts = rng.uniform([100, 100], [1700, 950], (n, 2))
    A = np.array([[0.9, -0.25, 140.0], [0.25, 0.9, -60.0], [0, 0, 1.0]]) if A is None else A
    q = R.project(A, pts) + rng.normal(0, jitter, (n, 2))
    b1 = [(x - 20, y - 30, x + 20, y) for x, y in pts]
    b2 = [(x - 20, y - 30, x + 20, y) for x, y in q] + [tuple(rng.uniform(0, 1800, 2)) + (0, 0) for _ in range(distract)]
    b2 = [(b[0], b[1], b[0] + 40, b[1] + 30) if b[2] == 0 else b for b in b2]
    perm = rng.permutation(len(b2))                                    # no correspondence given by the order
    return D(*b1), D(*[b2[k] for k in perm]), A, pts, q


def test_register_free_recovers_an_unknown_transform_without_correspondences():
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(10):
        d1, d2, A, pts, q = _scene(rng)
        H, info = R.register_free(d1, d2)
        if info["valid"]:
            ok += np.median(np.linalg.norm(R.project(H, pts) - q, axis=1)) < 10
    assert ok >= 8


def test_register_free_is_label_free_and_reports_failures():
    assert "g1" not in inspect.signature(R.register_free).parameters
    H, info = R.register_free(D(box(100, 100), box(300, 300)), D(box(120, 90)))
    assert H is None and not info["valid"] and info["why"]
    H, info = R.register_free(P.dets([box(1, 1), box(50, 50)], [1, 1], [.9, .9]),
                              P.dets([box(1, 1), box(50, 50)], [2, 2], [.9, .9]))
    assert H is None and info["why"] == "too few same-class candidates"


def test_register_free_warm_start_uses_previous_homography():
    rng = np.random.default_rng(1)
    d1, d2, A, pts, q = _scene(rng, distract=0)
    H, info = R.register_free(d1, d2, H_prev=A)
    assert info["init"] == "prev" and info["valid"]
    assert np.median(np.linalg.norm(R.project(H, pts) - q, axis=1)) < 5


def test_diag_error_uses_labels_only_as_a_ruler():
    g1 = G(i1=box(100, 100), i2=box(500, 300))
    g2 = G(i1=box(110, 100), i2=box(510, 300))
    assert R.diag_error(I3, g1, g2) == pytest.approx(10.0)
    assert R.diag_error(None, g1, g2) is None


# ---------------------------------------------------------------- records, crops, cap
def _rec(boxes, frame_boxes=(), frame_bits=400000, crops=()):
    det = P.dets(list(boxes), [1] * len(boxes), [0.9] * len(boxes))
    P1 = [dict(win=w, conf=0.1, bits={35: bb, 50: bb}, enc_ms={35: 1.0, 50: 1.0},
               bs={35: P.dets(list(cb), [1] * len(cb), [0.8] * len(cb)), 50: P.dets(list(cb), [1] * len(cb), [0.8] * len(cb))})
          for w, bb, cb in crops]
    return dict(W=1920, H=1080, det=det, det_ms=60.0, cands=dict(P1=P1, P2=[]),
                frame=dict(bits=frame_bits, enc_ms=1000.0,
                           bs=P.dets(list(frame_boxes), [1] * len(frame_boxes), [0.8] * len(frame_boxes))))


def test_covering_crop_and_cap():
    r = _rec([box(100, 100)], crops=[((80, 80, 200, 200), 5000, [box(100, 100)])])
    assert R.covering_crop(r, box(100, 100)) == ("P1", 0)
    assert R.covering_crop(r, box(900, 900)) is None
    big = _rec([box(100, 100)], frame_bits=900000)
    small = _rec([box(100, 100)], frame_bits=300000)
    tok = P.send(small, "T0", R.Q, R.NO_CAP)["ul"]
    cap = 300000 + P.G.msg_overhead(300000) + tok
    s = R.send_under_cap((small, big), ("T", "F"), cap)
    assert not s[1]["frame"] and s[0]["ul"] + s[1]["ul"] <= cap        # the big FRAME does not fit -> tokens
    s = R.send_under_cap((small, big), ("F", "T"), cap)
    assert s[0]["frame"]


def _seq(n=8, drift=0.0, crops=False):
    rng = np.random.default_rng(3)
    pts = rng.uniform([150, 150], [1600, 900], (14, 2))
    A = np.array([[0.95, -0.2, 120.0], [0.2, 0.95, -40.0], [0, 0, 1.0]])
    out = []
    for t in range(n):
        q = R.project(A, pts) + np.array([drift * t, 0.0])
        b1 = [box(x - 20, y - 30) for x, y in pts]
        b2 = [box(x - 20, y - 30) for x, y in q]
        g1 = {k: (*b, 1) for k, b in enumerate(b1)}
        g2 = {k: (*b, 1) for k, b in enumerate(b2)}
        cr = [((0, 0, 1920, 1080), 6000, [])] if crops else []
        out.append((_rec(b1, crops=cr), _rec(b2, crops=cr), g1, g2))
    return out


def test_phase0_on_a_clean_sequence():
    s, rows = R.phase0_sequence(_seq())
    assert len(rows) == 4 and s["valid_share"] == 1.0 and s["registrable"]
    assert s["diag_median_px"] < 5
    assert rows[1]["init"] == "prev"


def test_phase0_uses_first_half_only():
    seq = _seq()
    broken = seq[:4] + [(r1, r2, {}, {}) for r1, r2, _, _ in seq[4:]]
    assert R.phase0_sequence(broken)[0] == R.phase0_sequence(seq)[0]


def test_phase0_decision_rule():
    ok = dict(registrable=True, n_amb=10, cov_any=3, cov_p1=2)
    bad = dict(registrable=False, n_amb=10, cov_any=3, cov_p1=2)
    d = R.phase0_decision({s: ok for s in range(6)} | {6: bad, 7: bad})
    assert d["RQ1A"] == "IDENTIFIABLE" and d["RQ1B"] == "IDENTIFIABLE"
    d = R.phase0_decision({s: ok for s in range(5)} | {s: bad for s in range(5, 8)})
    assert d["RQ1A"].startswith("NOT IDENTIFIABLE") and d["RQ1B"].startswith("NOT IDENTIFIABLE")
    low = dict(registrable=True, n_amb=10, cov_any=1, cov_p1=1)
    d = R.phase0_decision({s: low for s in range(8)})
    assert d["RQ1A"] == "IDENTIFIABLE" and d["RQ1B"].startswith("NOT IDENTIFIABLE")


def test_phase12_free_equals_ideal_on_a_clean_sequence():
    r = R.phase12_sequence(_seq(), frame_view=1)
    m = r["modes"]["TT"]
    assert m["fail_share"] == 0 and m["free"]["f1"] == pytest.approx(m["ideal"]["f1"])
    assert m["none"]["fp_dup"] > 0 and m["none"]["f1"] < m["ideal"]["f1"]
    assert m["real_crop"]["f1"] == pytest.approx(m["free"]["f1"]) and m["real_crop_bits"] == 0.0


def test_phase12_real_crop_is_charged_and_never_sees_labels():
    seq = _seq(crops=True)
    a = R.phase12_sequence(seq, 1)["modes"]["TT"]
    # identical decisions and bits when the labels are removed: only the scores can change
    blind = R.phase12_sequence([(r1, r2, {}, {}) for r1, r2, _, _ in seq], 1)["modes"]["TT"]
    assert blind["real_crop_bits"] == pytest.approx(a["real_crop_bits"]) and blind["fail_share"] == a["fail_share"]


def _per(dA, rec_o=0.6, rec_r=0.1, bits_o=1000.0, bits_r=1000.0, fail=0, n=8):
    out = {}
    for s in range(n):
        d = dA[s] if isinstance(dA, list) else dA
        m = {"ideal": dict(f1=0.7), "free": dict(f1=0.7 - d), "gt_oracle": dict(f1=0.7 - d + rec_o * d),
             "real_crop": dict(f1=0.7 - d + rec_r * d), "gt_oracle_bits": bits_o, "real_crop_bits": bits_r,
             "n_fail": fail, "n_epochs": 10}
        out[s] = dict(frame_bits=400000.0, modes={"TT": m})
    return out


def test_decision_rules():
    assert R.rq1a_decision(_per(0.005))["verdict"] == "NO-GO"
    assert R.rq1a_decision(_per(0.05))["verdict"] == "GO (opportunity)"
    assert R.rq1a_decision(_per(0.02))["verdict"] == "INCONCLUSIVE"
    assert R.rq1a_decision(_per(0.05, fail=6))["verdict"].startswith("INCONCLUSIVE")
    assert R.rq1a_decision(_per([0.24] + [0.0] * 7))["verdict"] == "INCONCLUSIVE"     # 1/8 sequences
    p = _per(0.05)
    assert R.rq1b_decision(p, R.rq1a_decision(p))["verdict"].startswith("OPPORTUNITY ONLY")
    p = _per(0.05, rec_r=0.7)
    assert R.rq1b_decision(p, R.rq1a_decision(p))["verdict"].startswith("REAL-CROP USEFUL")
    p = _per(0.05, rec_r=0.7, bits_r=60000.0)
    assert R.rq1b_decision(p, R.rq1a_decision(p))["verdict"].startswith("OPPORTUNITY ONLY")   # too expensive
    p = _per(0.05, rec_o=0.3)
    assert R.rq1b_decision(p, R.rq1a_decision(p))["verdict"].startswith("NO IDEAL")


def test_cli_requires_data_use_basis(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["h1_rq1_assoc.py", "phase0", "p", "i", "o", "--zip", "z"])
    with pytest.raises(SystemExit):
        R.main()
