"""Synthetic tests for B1 tier 1 (view-aware opportunity): allocations, time split, equal-bit baselines, gate.
No dataset record is used."""
import numpy as np
import pytest

import h1_b1_view as B
import h1_pilot0b as P

A_BOX, B_BOX, C_BOX = (100, 100, 140, 130), (600, 400, 640, 430), (900, 700, 940, 730)


def _rec(tok_boxes, frame_boxes, n_extra_weak=0, frame_bits=400000):
    """onboard: strong tokens on tok_boxes (+ weak junk); BS on the frame: frame_boxes."""
    boxes = list(tok_boxes) + [(1500 + 20 * k, 900, 1510 + 20 * k, 910) for k in range(n_extra_weak)]
    det = P.dets(boxes, [1] * len(boxes), [0.9] * len(tok_boxes) + [0.06] * n_extra_weak)
    return dict(W=1920, H=1080, det=det, det_ms=60.0, cands=dict(P1=[], P2=[]),
                frame=dict(bits=frame_bits, enc_ms=1000.0, bs=P.dets(frame_boxes, [1] * len(frame_boxes),
                                                                    [0.8] * len(frame_boxes))))


def _gt(*ids_boxes):
    return {i: (*b, 1) for i, b in ids_boxes}


def _seq(n=8, flip_second_half=False):
    """objects 1, 2, 3. View 0 tokens see {1}; view 1 tokens see {1}. A FRAME from view 1 reveals {1, 2, 3};
    a FRAME from view 0 reveals only {1}. So FRAME pays on view 1 (index 1) only."""
    out = []
    for t in range(n):
        g0 = _gt((1, A_BOX), (2, B_BOX))
        g1 = _gt((1, A_BOX), (2, B_BOX), (3, C_BOX))
        good = 1 if not (flip_second_half and t >= n // 2) else 0
        r0 = _rec([A_BOX], [A_BOX, B_BOX] if good == 0 else [A_BOX], n_extra_weak=5)
        r1 = _rec([A_BOX], [A_BOX, B_BOX, C_BOX] if good == 1 else [A_BOX])
        out.append((r0, r1, g0, g1))
    return out


def test_allocations_and_unique_scoring():
    T = B.epoch_table(_seq(4))
    tt, f1t, tf2, ff = (B.score(T, a, {}) for a in ("TT", "F1T", "TF2", "FF"))
    assert (tt["tp"], tt["fn"], tt["fp"]) == (4, 8, 0)                 # object 1 once per epoch, of 3 unique objects
    assert tf2["r"] == pytest.approx(1.0) and f1t["r"] == pytest.approx(1 / 3)
    assert ff["f1"] == pytest.approx(tf2["f1"])                        # second FRAME adds nothing here
    assert tf2["fp_dup"] == 0 and tf2["fp_assoc"] == 0


def test_object_seen_in_both_views_counts_once_and_fp_counted():
    r0 = _rec([A_BOX], [A_BOX, (1700, 50, 1740, 80)])                  # frame adds a false box
    r1 = _rec([A_BOX], [A_BOX])
    T = B.epoch_table([(r0, r1, _gt((1, A_BOX)), _gt((1, A_BOX)))])
    s = B.score(T, "FF", {})
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 0)


def test_alt_alternates_and_bits_are_reported():
    T = B.epoch_table(_seq(4))
    assert [B.modes_of("ALT", i, {}) for i in range(4)] == [("F", "T"), ("T", "F"), ("F", "T"), ("T", "F")]
    tok = T[0]["bits"][0]["T"]
    assert B.score(T, "ALT", {})["bits"] == pytest.approx(400000 + P.G.msg_overhead(400000) + tok)
    assert B.score(T, "FF", {})["bits"] == pytest.approx(2 * (400000 + P.G.msg_overhead(400000)))
    assert B.score(T, "TT", {})["bits"] == pytest.approx(2 * tok)


def test_view_selection_uses_first_half_only():
    seq = _seq(8)
    a = B.run_sequence(seq)
    assert a["pref"]["VIEW"] == 1 and a["n_select"] == 4 and a["n_eval"] == 4
    assert a["alloc"]["VIEW"]["f1"] == pytest.approx(a["alloc"]["TF2"]["f1"])
    assert a["alloc"]["VIEW"]["f1"] > a["alloc"]["ALT"]["f1"]
    # corrupting the GT of the SECOND half must not change any selection
    bad = seq[:4] + [(r0, r1, {}, {}) for r0, r1, _, _ in seq[4:]]
    assert B.run_sequence(bad)["pref"] == a["pref"]
    # if the good view flips in the second half, the first-half choice is kept (and loses): no peeking
    f = B.run_sequence(_seq(8, flip_second_half=True))
    assert f["pref"]["VIEW"] == 1 and f["alloc"]["VIEW"]["f1"] < f["alloc"]["F1T"]["f1"]


def test_ndet_choice_is_label_free():
    seq = _seq(8)
    T = B.epoch_table(seq)
    assert B.select(T[:4])["NDET"] == 0                                # view 0 has more onboard detections (junk)
    no_gt = [(r0, r1, {}, {}) for r0, r1, _, _ in seq]
    assert B.select(B.epoch_table(no_gt)[:4])["NDET"] == 0             # identical without any label


def test_gate_rule():
    def seqres(view, tt, alt, ndet):
        z = dict(bits=1.0)
        return dict(alloc={"VIEW": dict(f1=view, **z), "TT": dict(f1=tt, **z), "ALT": dict(f1=alt, **z),
                           "NDET": dict(f1=ndet, **z), "FF": dict(f1=.9, **z), "F1T": dict(f1=.5, **z),
                           "TF2": dict(f1=.5, **z)})
    go = B.gate({s: seqres(.60, .50, .57, .55) for s in range(8)})
    assert go["baseline"] == "ALT" and go["view_minus_baseline"] == pytest.approx(.03) and go["GO"]
    small = B.gate({s: seqres(.58, .50, .57, .55) for s in range(8)})
    assert not small["GO"]                                              # +1 point < 2
    per = {s: seqres(.70 if s < 5 else .50, .50, .55, .55) for s in range(8)}   # big mean, only 5/8 positive
    g = B.gate(per)
    assert g["view_minus_baseline"] >= .02 and g["share_positive"] == pytest.approx(5 / 8) and not g["GO"]
    assert "FF" not in B.EQUAL_BIT_BASELINES                            # double-bit reference never sets the bar


def test_main_requires_data_use_basis(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["h1_b1_view.py", "p", "i", "o", "--zip", "z"])
    with pytest.raises(SystemExit):
        B.main()
