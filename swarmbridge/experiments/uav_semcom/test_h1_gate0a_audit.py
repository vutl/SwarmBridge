"""Tests for h1_gate0a_audit.py: exact oracles vs brute force, counterexamples to the 0A heuristic S0."""
import itertools
import random

import pytest

import h1_gate0a as G
import h1_gate0a_audit as AU


def brute_tok(A, B, budget):
    best = 0
    for n1 in range(len(A) + 1):
        for n2 in range(len(B) + 1):
            if AU.tok_cost(n1) + AU.tok_cost(n2) > budget:
                continue
            for s1 in itertools.combinations(sorted(A), n1):
                for s2 in itertools.combinations(sorted(B), n2):
                    best = max(best, len(set(s1) | set(s2)))
    return best


def brute_crop(A, B, cA, cB, budget):
    items = [(1, i) for i in A] + [(2, j) for j in B]
    best = 0
    for r in range(len(items) + 1):
        for s in itertools.combinations(items, r):
            pay = {1: sum(cA[i] for v, i in s if v == 1), 2: sum(cB[i] for v, i in s if v == 2)}
            if sum(pay.values()) + sum(G.msg_overhead(p) for p in pay.values()) <= budget:
                best = max(best, len({i for _, i in s}))
    return best


def test_counterexample_tok_header_precharge():
    # review #6: at 250 bits one UAV can send two tokens (48 + 96 + 2*52 = 248), but the 0A heuristic charges
    # both UAV headers first (288 bits) and sends nothing.
    A, B = {1, 2}, {3}
    assert AU.heuristic_s0_tok(A, B, 250) == 0
    assert AU.exact_tok(len(A), len(B), len(A | B), 250) == 2 == brute_tok(A, B, 250)


def test_counterexample_crop_greedy_with_headers():
    # greedy cheapest-first opens two messages (two headers) and loses to sending both items from one UAV
    A, B = {1, 2}, {3}
    cA, cB = {1: 100.0, 2: 100.0}, {3: 90.0}
    budget = 200 + G.MSG_HDR + G.APP_HDR
    assert AU.heuristic_s0_crop(A, B, cA, cB, budget) == 1
    assert AU.exact_crop(A, B, cA, cB, budget) == 2 == brute_crop(A, B, cA, cB, budget)


@pytest.mark.parametrize("seed", range(25))
def test_exact_tok_matches_bruteforce(seed):
    rng = random.Random(seed)
    U = list(range(rng.randint(1, 6)))
    A = {i for i in U if rng.random() < 0.7}; B = {i for i in U if rng.random() < 0.7}
    budget = rng.choice([0, 150, 200, 250, 300, 400, 500, 700])
    assert AU.exact_tok(len(A), len(B), len(A | B), budget) == brute_tok(A, B, budget)


@pytest.mark.parametrize("seed", range(25))
def test_exact_crop_matches_bruteforce(seed):
    rng = random.Random(seed)
    U = list(range(rng.randint(1, 5)))
    A = {i for i in U if rng.random() < 0.7}; B = {i for i in U if rng.random() < 0.7}
    cA = {i: float(rng.choice([50, 80, 120, 4000, 9000])) for i in A}
    cB = {i: float(rng.choice([50, 80, 120, 4000, 9000])) for i in B}
    budget = rng.choice([100, 250, 400, 600, 5000, 12000, 20000])
    e, sel = AU.exact_crop(A, B, cA, cB, budget, return_sel=True)
    assert e == brute_crop(A, B, cA, cB, budget) == len({i for _, i in sel})


def test_exact_never_below_heuristic_examples():
    rng = random.Random(7)
    for _ in range(40):
        U = list(range(rng.randint(1, 6)))
        A = {i for i in U if rng.random() < 0.7}; B = {i for i in U if rng.random() < 0.7}
        b = rng.choice([250, 400, 600, 900])
        assert AU.exact_tok(len(A), len(B), len(A | B), b) >= AU.heuristic_s0_tok(A, B, b) - 1e-9


def test_iou():
    assert AU.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert AU.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert AU.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def _toy():
    # one static object seen by both UAVs for 3 epochs, plus a UAV2-only object that moves far each epoch
    box = (100.0, 100.0, 20.0, 20.0)
    d = {"t": {1: {}, 2: {}}}
    for f in (1, 2, 3):
        d["t"][1][f] = {1: box}
        d["t"][2][f] = {1: box, 2: (100.0 * f, 500.0, 20.0, 20.0)}
    return d


def test_history_ind_resends_cross_view_but_oracle_does_not():
    d = _toy()
    big = 1e6
    cI, U, bitsI = AU._hist_run(d, "TOK", big, "IND", 0.5, random.Random(0), 1, 0.65, 2400.0)
    cS, U2, bitsS = AU._hist_run(d, "TOK", big, "S0", 0.5, random.Random(0), 1, 0.65, 2400.0)
    assert cI == cS == U == 6                                  # everything covered with ample budget
    # IND: epoch1 sends obj1 from both views + obj2 = 3 tokens; epochs 2,3: only the moving obj2 -> 5 tokens
    assert bitsI == 5 * G.TOK_BITS
    # oracle: obj1 once + obj2 each epoch = 4 tokens
    assert bitsS == 4 * G.TOK_BITS


def test_history_budget_limits_coverage():
    d = _toy()
    one_tok = G.TOK_BITS + G.msg_overhead(G.TOK_BITS)
    cS, U, _ = AU._hist_run(d, "TOK", one_tok, "S0", 0.5, random.Random(0), 1, 0.65, 2400.0)
    assert cS <= U and cS >= 3                                  # at least one new object per epoch
