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


def test_incumbent_kept_with_large_margin_and_causal_rollout(env_at_decision):
    core, k, tk = env_at_decision
    mpc = M.MPCMode("env", 16, 4, rollout="causal", margin_z=1e9, seed=0)
    assert mpc(core, k, tk, incumbent=5) == 5 and mpc.switched == 0
    free = M.MPCMode("env", 16, 4, rollout="causal", seed=0)(core, k, tk)
    assert 0 <= free < core.n_modes


def test_calibrator_uses_only_finished_segments():
    AE.init(False)
    sp, _ = MH.from_name(AE.tuned("phy_mcs_blind", "best"), C.MODES)
    core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg("phy_mcs_blind"))
    core.reset(8001)
    cal = M.CalRates()
    for _ in range(60):
        cal.update(core)
        n_seen = cal.seg_i
        assert n_seen <= len(core.segments)            # never reads beyond what has been recorded
        core.begin_slot([a[:3] for a in sp(core)])
        while core.advance()[0] == "mode":
            core.decide(4)
    cal.update(core)
    air = sum(sum(w) for _, w in cal.bins.values())
    assert air > 0 and 0 < cal.rho() < 1.5            # MCS + BLER realised below Shannon on average


def _episode_bias(model, regime="phy_mcs_blind", seed=8000, slot_weight=True):
    """run one heuristic episode with a RateErrTracker for `model`; return the pooled relative bias and airtime."""
    sp, mp = MH.from_name(AE.tuned(regime, "best"), C.MODES)
    cal = M.CalRates() if model == "cal" else None
    trk = M.RateErrTracker(M.MPCMode(model, cal=cal), slot_weight=slot_weight)
    import env_ev as V

    def sp2(env):
        if cal is not None:
            cal.update(env)
        trk.update(env)
        return sp(env)
    core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime))
    V.run_episode_ev(core, sp2, mp, seed)
    a = np.array(list(trk.bins.values())).sum(0)
    return a[0] / a[1] - 1, a[2], trk


def test_err_tracker_correct_model_unbiased_shannon_biased(env_at_decision):
    b_env, air, _ = _episode_bias("env")
    b_sh, _, _ = _episode_bias("shannon")
    assert air > 20
    assert abs(b_env) < 0.06         # MCS model without reuse: observed ~ predicted (fading MC error only)
    assert b_sh < -0.1               # Shannon over-predicts MCS throughput


def test_err_tracker_is_prequential():
    """the prediction attached to a slot's segments is the one made at that slot's start, even if the calibrator has
    changed since (it learns from those very segments before the tracker processes them)."""
    import types
    cal = types.SimpleNamespace(val=1e6)
    cal.mean = lambda env, k: cal.val
    mpc = types.SimpleNamespace(model="cal", cal=cal)
    env = types.SimpleNamespace(c=dict(p_levels=(1.0,), dt=30.0), uavs=[types.SimpleNamespace(pos=None)], tasks=object(),
                                t=0.0, segments=[], expected_gain=lambda pos: 1.0, noise_w=lambda: 1e-3)
    trk = M.RateErrTracker(mpc)
    trk.update(env)                       # slot 0 starts: prediction 1e6
    env.segments.append((0, 0, 0.0, 30.0, 0.0, 0.8e6))
    cal.val = 0.8e6                       # calibrator has learnt from the slot-0 segment
    trk.update(env)                       # same slot: nothing changes
    env.t = 30.0
    trk.update(env)                       # slot 0 ended: segment scored against 1e6, not 0.8e6
    assert abs(trk.bias(env, 0) - (-0.2)) < 1e-9
    assert trk.prev[2] == [0.8e6]


def test_airtime_weighting_is_biased_low():
    """queue emptying in good-fading slots: the airtime-weighted log under-reports the correct model's rate."""
    b_slot, _, _ = _episode_bias("env")
    b_air, _, _ = _episode_bias("env", slot_weight=False)
    assert b_air < b_slot - 0.03
