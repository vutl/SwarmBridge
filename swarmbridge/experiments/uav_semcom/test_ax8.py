"""Review #43 (A-X8): calibrated heuristic mode rules and heuristic-seeded MPC in ax2_eval."""
import pytest

import a5_eval as AE
import ax2_eval as X2
import ax8_run as R
import policies as P


@pytest.fixture(scope="module")
def g():
    AE.init(False)
    return AE.G


def ep(combo, seed=3):
    return X2.episode("K5", "K5_la", combo, -1, seed, R.EXTRA, R.JA_MEMBER, "train")[0]


def test_calibration_does_not_touch_rate_free_rules(g):
    assert ep("ja|h=fixed_avif_20")["return"] == ep("ja|hc=fixed_avif_20")["return"]


def test_calibrated_wrapper_restores_expected_rate(g):
    orig = P.expected_rate
    ep("jae|hc=greedy+budget1")
    assert P.expected_rate is orig


def test_calibration_changes_rate_based_rules(g):
    assert ep("jae|h=greedy")["return"] != ep("jae|hc=greedy")["return"]


def test_heuristic_seeded_mpc_runs_within_budget(g):
    m = ep("jae|hc=greedy+mpc")
    assert "switched" in m and m["mode_over50"] < 0.05 and m["ms_int_max_p99"] < 100


def test_integrated_latency_fields(g):
    m = ep("ja|h=greedy")
    assert m["ms_int_max_p99"] >= m["ms_slot_p50"] and m["ms_int_sum_p99"] >= m["ms_int_max_p99"] - 1e-9
