"""Tests for the step-0 Huang-inspired model and simulator: python -m pytest test_step0.py (from this directory)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
import policies as P  # noqa: E402
import sim as S  # noqa: E402

FULL = dict(noise="paper", payload="original", queue=True, visibility=True, interference=True)
NONE = dict(noise="paper", payload="original", queue=False, visibility=False, interference=False)


def test_link_budget_matches_hand_calculation():
    # FSPL(750 km, 25 GHz) = 20 log10(4 pi d f / c) = 177.9 dB; 30 dBm + 40 dBi - 177.9 dB + 130 dBm = 22.1 dB
    snr = 10 * math.log10(M.leo_uav_snr_mean(750e3, M.noise_power("paper")))
    assert abs(snr - 22.1) < 0.1
    assert abs(10 * math.log10(M.noise_power("thermal")) + 30 + 103.1) < 0.1


def test_visibility_angle_and_access_time():
    elev, _ = M.leo_elev_dist(M.PHI_VIS)
    assert abs(np.rad2deg(elev) - 40.0) < 1e-6
    assert 150 < M.access_time() < 300


def test_sce_and_penalty():
    u, ok = M.utility(32.2, 1.0, 10.0, 11.0)
    assert ok and abs(u - (0.5 * (1 - M.norm_q(10.0)) + 0.5 * 10 / 11)) < 1e-9
    u2, ok2 = M.utility(11.26, 1.0, 20.0, 11.0)
    assert not ok2 and u2 < 0


def test_fifo_port_serialises_tasks():
    sc = S.Scenario(1, 4.0, FULL)
    st = S.State(sc)
    S.commit(sc, st, 0, 0, 0)
    end0 = st.rx[0]
    _, _, _, tm = S.outcome(sc, st, 1, 0, 0, actual=True)
    assert tm[0] >= end0           # uplink of task 1 ends after task 0's uplink
    assert end0 > sc.arrival[0]


def test_deterministic_given_seed():
    a = S.run(S.Scenario(3, 1.0, FULL), P.greedy)
    b = S.run(S.Scenario(3, 1.0, FULL), P.greedy)
    assert a == b


def test_without_coupling_hindsight_equals_per_task_best():
    """No shared ports, no visibility, no interference: tasks are independent, so the hindsight optimum is the sum
    of per-task best realised outcomes (greedy is optimal up to not knowing the fading draw)."""
    sc = S.Scenario(5, 2.0, NONE)
    st = S.State(sc)
    per_task = sum(max(S.outcome(sc, st, i, k, n, actual=True)[0] for k, n in P.OPTIONS)
                   for i in range(M.G_TASKS)) / M.G_TASKS
    g = S.run(sc, P.greedy)
    h = P.hindsight(sc, g[2], passes=2)
    assert abs(h[0] - per_task) < 1e-9
    assert h[0] >= g[0] - 1e-12


def test_hindsight_never_worse_than_init():
    sc = S.Scenario(7, 4.0, FULL)
    g = S.run(sc, P.greedy)
    h = P.hindsight(sc, g[2], passes=1)
    assert h[0] >= g[0] - 1e-12
