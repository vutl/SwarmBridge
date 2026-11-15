"""Tests for the A-E2 mode-only MPC (a9_mpc.py): causality, budget fallback, deterministic-path sanity."""
import copy

import numpy as np
import pytest

import a5_common as C
import a5_eval as AE
import a9_mpc as M
import env_survey as SV
import mode_heur as MH


@pytest.fixture(scope="module")
def env_at_decision():
    AE.init(False)
    sp, _ = MH.from_name(AE.tuned("phy_mcs_blind", "best"), C.MODES)
    core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg("phy_mcs_blind"))
    core.reset(8000)
    for _ in range(40):                                  # advance to a mode decision with a non-empty queue
        core.begin_slot([a[:3] for a in sp(core)])
        while True:
            ev = core.advance()
            if ev[0] != "mode":
                break
            if len(core.uavs[ev[1]].queue) >= 2:
                return core, ev[1], ev[2]
            core.decide(0)
    pytest.skip("no suitable decision found")


def test_future_fading_is_never_read(env_at_decision):
    core, k, tk = env_at_decision
    a = M.MPCMode("env", 8, 4, seed=1)(core, k, tk)
    c2 = copy.deepcopy(core)
    s = c2.slot
    rng = np.random.default_rng(99)
    c2.f_u[s:] = rng.random(c2.f_u[s:].shape)
    c2.f_n[s:] = rng.normal(size=c2.f_n[s:].shape)
    c2.f_e[s:] = rng.exponential(size=c2.f_e[s:].shape)
    tk2 = next(t for t in c2.uavs[k].queue if t.t_cap == tk.t_cap and t.img == tk.img)
    assert M.MPCMode("env", 8, 4, seed=1)(c2, k, tk2) == a


def test_budget_timeout_uses_fallback(env_at_decision):
    core, k, tk = env_at_decision
    mpc = M.MPCMode("env", 64, 8, budget_ms=1e-6, fallback=lambda e, kk, t: 3, seed=0)
    assert mpc(core, k, tk) == 3 and mpc.timeouts == 1


def test_models_differ_under_mcs(env_at_decision):
    core, k, tk = env_at_decision
    p = core.c["p_levels"][-1] * core.expected_gain(core.uavs[k].pos)
    r_env = M.MPCMode("env")._rate(core, np.array([p]))[0]
    r_sh = M.MPCMode("shannon")._rate(core, np.array([p]))[0]
    assert r_sh > r_env > 0          # Shannon overestimates the MCS rate (with BLER)


@pytest.mark.parametrize("regime", ["phy_mcs_blind", "ref", "phy_gap3"])
def test_vectorised_rate_matches_env(regime):
    AE.init(False)
    core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime))
    core.reset(1)
    sig = np.logspace(-16, -9, 50)
    n = core.noise_w()
    ref = np.array([core.link_rate(s, n, 0.0, s / n) for s in sig])
    assert np.allclose(M.MPCMode("env")._rate(core, sig), ref)
