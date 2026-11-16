"""Tests for the A-X1 pilot: control equivalence before any comparison is read."""
import json

import numpy as np
import pytest

import a5_common as C
import a5_eval as AE
import ax1_pilot as XP
import ax1_sched as XS
import env_ev as V
import env_survey as SV
import mode_heur as MH


@pytest.fixture(scope="module", autouse=True)
def init():
    AE.init(False)


@pytest.mark.parametrize("regime", ["K5", "ref"])
def test_sch_a0_rules_reproduce_the_a5_heuristic(regime):
    """SCH:a0:s0:least on the T:best base is the A5 '+reuse' rule; :none is the base rule (exact returns)."""
    name = AE.tuned(regime, "best")
    for sch, ref_name in (("SCH:a0:s0:least", XS.base_name(regime) + "+reuse"), ("SCH:a0:s0:none", XS.base_name(regime))):
        for seed in (8700, 8701):
            sp, mp = XS.from_name(sch, regime)
            r1 = V.run_episode_ev(SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime)), sp, mp, seed)
            sp2, mp2 = MH.from_name(ref_name, C.MODES)
            r2 = V.run_episode_ev(SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime)), sp2, mp2, seed)
            assert abs(r1["return"] - r2["return"]) < 1e-9, (sch, seed)
    assert name.endswith("+reuse")                       # in K5 / ref T:best is the reuse member


@pytest.mark.parametrize("regime", ["K5", "ref"])
def test_pilot_loop_reproduces_heuristic_episode(regime):
    """tbest|heur through the gym wrapper = run_episode_ev with T:best (same tasks, fading and reward)."""
    sp, mp = MH.from_name(AE.tuned(regime, "best"), C.MODES)
    for seed in (8700, 8702):
        m, _ = XP.episode(regime, "tbest|heur", -1, seed)
        r = V.run_episode_ev(SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime)), sp, mp, seed)
        assert abs(m["return"] - r["return"]) < 1e-6


def test_pilot_ppo_mode_matches_a5_rl_episode():
    """tbest|ppo = the A5 mode-only PPO evaluation path (a5_eval.rl_episode) on the same seed."""
    m, _ = XP.episode("K5", "tbest|ppo", 0, 8700)
    info, _ = AE.rl_episode("K5", "test", XP.mode_path("K5", 0), "mode", AE.tuned("K5", "best"), "PPPP", 8700)
    assert abs(m["return"] - info["return"]) < 1e-6


def test_pilot_joint_hphp_matches_a5_combo():
    """jc|jm = A5 action replacement 'HPHP' (heuristic move + power, PPO channel + mode)."""
    m, _ = XP.episode("K5", "jc|jm", 1, 8701)
    info, _ = AE.rl_episode("K5", "test", XP.joint_path("K5", 1), "joint", AE.tuned("K5", "best"), "HPHP", 8701)
    assert abs(m["return"] - info["return"]) < 1e-6
