"""Tests for h1_gate0a.py (protocol v2.1 sec 10.7). Run: python -m pytest test_h1_gate0a.py -q"""
import itertools
import math

import pytest

import h1_gate0a as G


def brute_random_cov(A, B, kA, kB):
    """exact expectation by enumerating all uniformly random subsets of size kA from A and kB from B."""
    A, B = sorted(A), sorted(B)
    kA, kB = min(kA, len(A)), min(kB, len(B))
    tot = n = 0
    for sa in itertools.combinations(A, kA):
        for sb in itertools.combinations(B, kB):
            tot += len(set(sa) | set(sb)); n += 1
    return tot / n


@pytest.mark.parametrize("A,B,kA,kB", [({1, 2, 3, 4}, {3, 4, 5}, 2, 1), ({1, 2, 3}, {1, 2, 3}, 1, 2),
                                         ({1, 2, 3, 4, 5}, {6, 7}, 3, 1), ({1, 2}, {2, 3, 4, 5}, 5, 2)])
def test_independent_expectation_matches_bruteforce(A, B, kA, kB):
    bA = kA * G.TOK_BITS + G.msg_overhead(kA * G.TOK_BITS)
    bB = kB * G.TOK_BITS + G.msg_overhead(kB * G.TOK_BITS)
    assert G.tok_capacity(bA) == kA and G.tok_capacity(bB) == kB
    r = G.tok_independent(set(A), set(B), bA, bB)
    assert r["cov"] == pytest.approx(brute_random_cov(A, B, kA, kB))


def test_capacity_counts_headers():
    assert G.tok_capacity(G.MSG_HDR + G.APP_HDR + G.TOK_BITS - 1) == 0
    assert G.tok_capacity(G.MSG_HDR + G.APP_HDR + 3 * G.TOK_BITS) == 3


def brute_coord_cov(candA, candB, K):
    items = [("a", i) for i in sorted(candA)] + [("b", i) for i in sorted(candB)]
    K = min(K, len(items))
    tot = n = 0
    for s in itertools.combinations(items, K):
        tot += len({i for _, i in s}); n += 1
    return tot / n


def test_oracle_coverage_is_min_K_union():
    A, B = {1, 2, 3, 4, 5}, {4, 5, 6, 7}
    for K in range(0, 9):
        hdr = G.msg_overhead(len(A) * G.TOK_BITS) + G.msg_overhead(len(B - A) * G.TOK_BITS)
        r = G.tok_coordinated(A, B, A & B, hdr + K * G.TOK_BITS, 0.0, 1.0)
        assert r["cov"] == pytest.approx(min(K, len(A | B)))
        assert r["lost"] == 0 and r["dup_tok"] == pytest.approx(0.0)


def test_coordinated_with_errors_matches_bruteforce_and_counts_lost():
    A, B = {1, 2, 3}, {2, 3, 4, 5}
    sup = {3, 5}                      # 3 correctly suppressed (UAV1 has it); 5 wrongly suppressed (lost)
    candA, candB = A, B - sup         # candB = {2, 4}: object 2 is a missed duplicate
    for K in range(0, 6):
        hdr = G.msg_overhead(len(candA) * G.TOK_BITS) + G.msg_overhead(len(candB) * G.TOK_BITS)
        r = G.tok_coordinated(A, B, sup, hdr + K * G.TOK_BITS, 0.0, 1.0)
        assert r["cov"] == pytest.approx(brute_coord_cov(candA, candB, K))
        assert r["lost"] == 1


def test_downlink_cost_is_charged_at_rho():
    A, B = {1, 2, 3}, {3, 4}
    hdr = G.msg_overhead(3 * G.TOK_BITS) + G.msg_overhead(1 * G.TOK_BITS)
    budget = hdr + 4 * G.TOK_BITS
    assert G.tok_coordinated(A, B, {3}, budget, 0.0, 1.0)["cov"] == pytest.approx(4)
    assert G.tok_coordinated(A, B, {3}, budget, G.TOK_BITS, 1.0)["cov"] == pytest.approx(3)
    assert G.tok_coordinated(A, B, {3}, budget, G.TOK_BITS, 0.0)["cov"] == pytest.approx(4)


def test_scheme_costs_and_lag_errors():
    box = (100.0, 100.0, 20.0, 20.0)
    prev = ({1, 2, 3}, {2, 3, 9}, {2: box, 3: box, 9: box})
    A, B = {1, 2}, {2, 3, 4}          # 3 was shared at t-1 but UAV1 lost it now; 4 is new
    Bbox = {2: box, 3: box, 4: box}
    sm = {"8x8": set(), "16x9": set()}
    s = G.schemes("q", 10, A, B, Bbox, prev, sm)
    assert s["S0"] == ({2}, 0.0)
    assert s["S1"][0] == {2, 3} and s["S1"][1] == G.TOK_BITS * 2 + G.DL_HDR
    assert s["S2"][0] == {2, 3} and s["S2"][1] == 3 + G.DL_HDR
    assert s["S2b_1"][1] == math.ceil(1.44 * math.log2(100) * 2) + G.DL_HDR
    assert {2, 3} <= s["S2b_1"][0]
    # lag error: object 3 suppressed by UAV2 but not held by UAV1 -> lost
    hdr = G.msg_overhead(2 * G.TOK_BITS) + G.msg_overhead(1 * G.TOK_BITS)
    r = G.tok_coordinated(A, B, s["S2"][0], 1e9, 0.0, 1.0)
    assert r["lost"] == 1 and r["cov"] == 3       # covers 1, 2, 4; misses 3


def test_grid_mask_majority_rule():
    left, right = (10.0, 10.0, 5.0, 5.0), (1900.0, 1000.0, 5.0, 5.0)
    m = G.grid_mask({1: left, 2: left, 3: right}, shared={1, 2}, g=(8, 8))
    assert G.in_mask(left, m, (8, 8)) and not G.in_mask(right, m, (8, 8))
    m2 = G.grid_mask({1: left, 2: left}, shared={1}, g=(8, 8))       # 50% shared -> masked
    assert G.in_mask(left, m2, (8, 8))


def test_crop_greedy_optimal_without_headers(monkeypatch):
    monkeypatch.setattr(G, "MSG_HDR", 0); monkeypatch.setattr(G, "APP_HDR", 0)
    costs = [(5.0, 1, "a"), (3.0, 1, "b"), (8.0, 2, "c"), (2.0, 2, "d"), (6.0, 1, "e")]
    for budget in range(0, 26):
        best = max(len(s) for r in range(len(costs) + 1) for s in itertools.combinations(costs, r)
                   if sum(c for c, _, _ in s) <= budget)
        assert len(G.greedy_items(costs, budget)) == best


def test_crop_oracle_sends_shared_once_from_cheaper_view(monkeypatch):
    monkeypatch.setattr(G, "MSG_HDR", 0); monkeypatch.setattr(G, "APP_HDR", 0)
    A, B = {1, 2}, {2, 3}
    cA, cB = {1: 10.0, 2: 50.0}, {2: 20.0, 3: 10.0}
    r = G.crop_coordinated(A, B, cA, cB, set(), 40.0, 0.0, 1.0, oracle=True)
    assert r["cov"] == 3 and r["dup_bits"] == 0
    ri = G.crop_independent(A, B, cA, cB, 20.0, 20.0)
    assert ri["cov"] == 2


def test_policy_never_reads_ids_beyond_given_sets():
    # coordinated schemes only use previous-epoch sets and current UAV2 boxes; S0 is the only one using A&B now
    box = (0.0, 0.0, 1.0, 1.0)
    s = G.schemes("q", 1, {1}, {1, 2}, {1: box, 2: box}, None, {"8x8": set(), "16x9": set()})
    assert all(v == (set(), 0.0) for k, v in s.items() if k != "S0")


def test_capacity_huge_budget_is_fast_and_capped():
    import time
    t = time.time()
    assert G.tok_capacity(1e12, cap=7) == 7
    r = G.tok_independent({1, 2, 3}, {3, 4}, 5e11, 5e11)
    assert r["cov"] == pytest.approx(4) and time.time() - t < 0.5
