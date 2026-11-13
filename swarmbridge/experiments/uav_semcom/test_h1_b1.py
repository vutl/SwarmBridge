"""Synthetic tests for B1 tier 1 (view-aware opportunity): allocations, time split, joint UL cap (amendment 1),
label-free baselines, gate. No dataset record is used."""
import numpy as np
import pytest

import h1_b1_view as B
import h1_pilot0b as P

A_BOX, B_BOX, C_BOX = (100, 100, 140, 130), (600, 400, 640, 430), (900, 700, 940, 730)


def cost(payload):
    return payload + P.G.msg_overhead(payload)


TOK1 = cost(P.TOK_BITS)                                                  # one strong token, with headers


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


def _seq_sizes(n=8, small=300000, big=900000, big_second_half=None):
    """the reviewer's case: unequal FRAME sizes. View 1's FRAME is 3x larger AND is the informative one."""
    out = []
    for t in range(n):
        g0 = _gt((1, A_BOX), (2, B_BOX))
        g1 = _gt((1, A_BOX), (2, B_BOX), (3, C_BOX))
        fb = big if big_second_half is None or t < n // 2 else big_second_half
        out.append((_rec([A_BOX], [A_BOX], frame_bits=small), _rec([A_BOX], [A_BOX, B_BOX, C_BOX], frame_bits=fb),
                    g0, g1))
    return out


# ---------------------------------------------------------------- allocations and scoring (no cap)
def test_allocations_and_unique_scoring():
    T = B.epoch_table(_seq(4))
    tt, f1t, tf2, ff = (B.score(T, a, {}) for a in ("TT", "F1T", "TF2", "FF"))
    assert (tt["tp"], tt["fn"], tt["fp"]) == (4, 8, 0)                 # object 1 once per epoch, of 3 unique objects
    assert tf2["r"] == pytest.approx(1.0) and f1t["r"] == pytest.approx(1 / 3)
    assert ff["f1"] == pytest.approx(tf2["f1"])                        # second FRAME adds nothing here
    assert tf2["fp_dup"] == 0 and tf2["fp_assoc"] == 0
    assert all(x["feasible"] == 1.0 for x in (tt, f1t, tf2, ff))


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
    assert tok == TOK1
    assert B.score(T, "ALT", {})["bits"] == pytest.approx(cost(400000) + tok)
    assert B.score(T, "FF", {})["bits"] == pytest.approx(2 * cost(400000))
    assert B.score(T, "TT", {})["bits"] == pytest.approx(2 * tok)


# ---------------------------------------------------------------- joint UL cap (amendment 1, review #19)
def test_regression_unequal_frames_spend_different_bits_without_a_cap():
    """the defect found in review #19: 'one FRAME + one TOKEN' is not an equal-bit comparison."""
    T = B.epoch_table(_seq_sizes(4))
    assert B.score(T, "TF2", {})["bits"] > 2.9 * B.score(T, "F1T", {})["bits"]


def test_cap_is_enforced_with_token_fallback():
    T = B.epoch_table(_seq_sizes(4))
    cap = cost(300000) + TOK1                                          # the small FRAME fits, the big one does not
    f1t, tf2, tt = (B.score(T, a, {}, 0, cap) for a in ("F1T", "TF2", "TT"))
    assert f1t["feasible"] == 1.0 and f1t["bits"] == pytest.approx(cap)
    assert tf2["feasible"] == 0.0                                       # every epoch falls back to TOKEN + TOKEN
    assert tf2["bits"] == pytest.approx(tt["bits"]) and tf2["f1"] == pytest.approx(tt["f1"])
    assert max(x["bits_max"] for x in (f1t, tf2, tt)) <= cap


def test_ff_under_a_one_frame_cap_sends_one_frame():
    T = B.epoch_table(_seq(4))
    cap = cost(400000) + TOK1
    ff = B.score(T, "FF", {}, 0, cap)
    assert ff["bits"] == pytest.approx(cap) and ff["feasible"] == 0.0
    assert ff["f1"] == pytest.approx(B.score(T, "F1T", {}, 0, cap)["f1"])   # equal sizes: the first view keeps it


def test_tokens_are_truncated_when_even_tokens_do_not_fit():
    r = _rec([A_BOX, B_BOX, C_BOX], [A_BOX])
    g = _gt((1, A_BOX), (2, B_BOX), (3, C_BOX))
    T = B.epoch_table([(r, r, g, g)])
    full = B.score(T, "TT", {})
    cut = B.score(T, "TT", {}, 0, 2 * TOK1)                             # room for one token per view
    assert full["tp"] == 3 and cut["tp"] < 3 and cut["feasible"] == 0.0
    assert cut["bits_max"] <= 2 * TOK1 and cut["bits"] > 0
    zero = B.score(T, "TF2", {}, 0, 10.0)                               # nothing fits at all
    assert zero["bits"] == 0.0 and zero["tp"] == 0


def test_cap_invariant_property():
    rng = np.random.default_rng(0)
    for _ in range(40):
        seq = _seq_sizes(6, small=int(rng.integers(2000, 600000)), big=int(rng.integers(2000, 1200000)))
        T = B.epoch_table(seq)
        for cap in rng.uniform(0, 1.6e6, 6):
            pref = B.select(T[:3], cap)
            for a in B.ALLOCS:
                s = B.score(T[3:], a, pref, 3, cap)                     # score() raises if any epoch exceeds the cap
                assert s["bits_max"] <= cap + 1e-6


def test_budget_ref_uses_first_half_bits_only_and_no_label():
    seq = _seq_sizes(8)
    a = B.run_sequence(seq)
    assert a["B_ref"] == pytest.approx(cost(900000) + TOK1)
    assert a["levels"]["1"]["B"] == pytest.approx(a["B_ref"]) and a["levels"]["0.5"]["B"] == pytest.approx(a["B_ref"] / 2)
    assert set(a["levels"]) == {B.key(k) for k in B.KAPPAS}
    # second-half FRAMEs twice as large: B_ref unchanged, and the oversized FRAME is not sent (no peeking at sizes)
    grown = B.run_sequence(_seq_sizes(8, big_second_half=1800000))
    assert grown["B_ref"] == pytest.approx(a["B_ref"])
    assert grown["alloc"]["TF2"]["feasible"] == 0.0 and grown["alloc"]["TF2"]["bits_max"] <= grown["B_ref"]
    # no label anywhere: same B_ref, same bits, same label-free choices
    blind = B.run_sequence([(r0, r1, {}, {}) for r0, r1, _, _ in seq])
    assert blind["B_ref"] == pytest.approx(a["B_ref"])
    assert {k: blind["pref"][k] for k in ("NDET", "BIG")} == {k: a["pref"][k] for k in ("NDET", "BIG")}
    for alloc in ("TT", "F1T", "TF2", "ALT", "FF", "NDET", "BIG"):
        assert blind["alloc"][alloc]["bits"] == pytest.approx(a["alloc"][alloc]["bits"])


def test_view_choice_is_made_under_the_cap():
    a = B.run_sequence(_seq_sizes(8))
    full, half = a["levels"]["1"], a["levels"]["0.5"]
    assert full["pref"]["VIEW"] == 1 and full["alloc"]["VIEW"]["f1"] == pytest.approx(1.0)
    # at half the reference cap the informative (big) FRAME never fits: VIEW cannot claim its value
    assert half["alloc"]["TF2"]["feasible"] == 0.0
    assert half["pref"]["VIEW"] == 0
    assert half["alloc"]["VIEW"]["f1"] == pytest.approx(half["alloc"]["TT"]["f1"])
    assert half["alloc"]["VIEW"]["bits_max"] <= half["B"]


def test_achieved_bits_do_not_decrease_with_the_cap_for_fixed_allocations():
    a = B.run_sequence(_seq_sizes(8))
    for alloc in ("TT", "F1T", "TF2", "ALT", "FF"):
        bits = [a["levels"][B.key(k)]["alloc"][alloc]["bits"] for k in B.KAPPAS]
        assert all(y >= x - 1e-9 for x, y in zip(bits, bits[1:]))
    assert a["ff_uncapped"]["bits"] == pytest.approx(cost(300000) + cost(900000))


def test_big_is_a_label_free_spend_more_control_in_the_gate():
    seq = _seq_sizes(8)
    assert B.select(B.epoch_table(seq)[:4])["BIG"] == 1
    assert "BIG" in B.EQUAL_BIT_BASELINES and "FF" not in B.EQUAL_BIT_BASELINES
    g = B.gate({s: B.run_sequence(seq) for s in range(8)})
    # VIEW picks the same (larger, informative) view as BIG: no GO, although it beats the view-agnostic baselines
    assert g["baseline"] == "BIG" and g["view_minus_baseline"] == pytest.approx(0.0) and not g["GO"]
    assert g["view_agnostic"]["passes"] and g["view_agnostic"]["baseline"] == "ALT"
    assert g["label_free_rule_sufficient"]
    assert g["bits_ratio_view_over_baseline"] == pytest.approx(1.0)
    assert set(g["tradeoff"]) == {B.key(k) for k in B.KAPPAS}
    assert g["tradeoff"]["0.5"]["view_minus_baseline"] == pytest.approx(0.0)


# ---------------------------------------------------------------- time split and label-free choices
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


def test_realise_does_not_look_at_labels():
    seq = _seq_sizes(4)
    T, U = B.epoch_table(seq), B.epoch_table([(r0, r1, {}, {}) for r0, r1, _, _ in seq])
    for cap in (None, 10.0, 2 * TOK1, cost(300000) + TOK1, 2e6):
        for req in (("T", "T"), ("F", "T"), ("T", "F"), ("F", "F")):
            assert B.realise(T[0], req, cap)[1:] == B.realise(U[0], req, cap)[1:]


# ---------------------------------------------------------------- gate
def _seqres(view, tt, alt, ndet, big=0.0):
    z = dict(bits=1.0, feasible=1.0)
    al = {"VIEW": dict(f1=view, **z), "TT": dict(f1=tt, **z), "ALT": dict(f1=alt, **z), "NDET": dict(f1=ndet, **z),
          "BIG": dict(f1=big, **z), "FF": dict(f1=.9, **z), "F1T": dict(f1=.5, **z), "TF2": dict(f1=.5, **z)}
    return dict(levels={B.key(k): dict(alloc=al) for k in B.KAPPAS}, ff_uncapped=dict(f1=.9, bits=2.0))


def test_gate_rule():
    go = B.gate({s: _seqres(.60, .50, .57, .55) for s in range(8)})
    assert go["baseline"] == "ALT" and go["view_minus_baseline"] == pytest.approx(.03) and go["GO"]
    assert not go["label_free_rule_sufficient"] and go["kappa"] == 1.0
    small = B.gate({s: _seqres(.58, .50, .57, .55) for s in range(8)})
    assert not small["GO"]                                              # +1 point < 2
    per = {s: _seqres(.70 if s < 5 else .50, .50, .55, .55) for s in range(8)}   # big mean, only 5/8 positive
    g = B.gate(per)
    assert g["view_minus_baseline"] >= .02 and g["share_positive"] == pytest.approx(5 / 8) and not g["GO"]
    six = B.gate({s: _seqres(.70 if s < 6 else .50, .50, .55, .55) for s in range(8)})
    assert six["share_positive"] == pytest.approx(6 / 8) and six["GO"]   # 6 / 8 >= 73 % (registered share rule)
    spend = B.gate({s: _seqres(.60, .50, .57, .55, big=.60) for s in range(8)})
    assert not spend["GO"] and spend["label_free_rule_sufficient"]      # the spend-more control matches VIEW
    # the pre-amendment set {TT, ALT, NDET} is reported next to the gate but never decides
    assert spend["original_baselines"]["passes"] and spend["original_baselines"]["baseline"] == "ALT"
    assert go["original_baselines"]["passes"] and "BIG" not in B.ORIGINAL_BASELINES


def test_gate_reports_cap_utilisation_and_bit_ratios():
    g = B.gate({s: B.run_sequence(_seq_sizes(8)) for s in range(3)})
    cap = cost(900000) + TOK1
    assert g["utilisation"]["TF2"] == pytest.approx(1.0)                # the big FRAME + tokens fill the cap exactly
    assert g["utilisation"]["F1T"] == pytest.approx((cost(300000) + TOK1) / cap)
    assert g["utilisation"]["TT"] == pytest.approx(2 * TOK1 / cap)
    assert g["bits_ratio_view_over_each"]["F1T"] == pytest.approx(cap / (cost(300000) + TOK1))   # same cap, 3x the bits
    assert g["bits_ratio_view_over_each"]["BIG"] == pytest.approx(1.0)
    assert set(g["tradeoff"]["0.5"]) >= {"utilisation", "feasible", "mean_bits", "mean_f1"}
    assert "matched CAP" in g["comparison"]


def test_main_requires_data_use_basis(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["h1_b1_view.py", "p", "i", "o", "--zip", "z"])
    with pytest.raises(SystemExit):
        B.main()
