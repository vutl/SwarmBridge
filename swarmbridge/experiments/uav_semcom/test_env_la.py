"""PHY validation for the review #39 link-adaptation model (env_la), independent of any RL."""
import math

import numpy as np
import pytest

import a5_common as C
import a5_eval as AE
import env_ev as V
import env_la as LA
import env_survey as SV
import mode_heur as MH


def test_bler_waterfall():
    for i in range(len(LA.THR)):
        assert abs(LA.bler(LA.THR[i], i) - 0.1) < 1e-12
        assert LA.bler(LA.THR[i] + 1.0, i) < 0.012                     # ~1 % one dB above the threshold
        assert LA.bler(LA.THR[i] - 1.0, i) > 0.5
        xs = [LA.bler(LA.THR[i] + d, i) for d in np.linspace(-5, 5, 41)]
        assert all(a >= b for a, b in zip(xs, xs[1:]))


def olla_run(gamma_db, sigma, bias, delay, step, n=20000, seed=0):
    """stationary link: true SINR gamma, reports = gamma + bias + sigma z, delayed; returns (BLER, goodput / genie)."""
    rng = np.random.default_rng(seed)
    ad = LA.LinkAdapter(0.1, step, delay, 10.0)
    nack = good = 0
    for b in range(n):
        rep = ad.report(b)
        est = (rep[1] if rep else gamma_db) - ad.off
        i = LA.cqi_for(est)
        ok = rng.random() < 1 - LA.bler(gamma_db, i)
        ad.feedback(not ok)
        ad.meas.append((b, gamma_db + bias + sigma * rng.normal(), 0.0))
        nack += not ok
        good += ok * LA.EFF[i]
    genie = max(LA.EFF[i] * (1 - LA.bler(gamma_db, i)) for i in range(len(LA.THR)))
    return nack / n, good / n / genie


@pytest.mark.parametrize("sigma,bias,delay", [(0.0, 0.0, 1), (1.0, 0.0, 2), (3.0, 0.0, 2), (1.0, 4.0, 10),
                                              (1.0, -2.0, 20)])
def test_olla_converges_to_target_bler(sigma, bias, delay):
    for g in (2.0, 8.0, 15.0):
        bl, eff = olla_run(g, sigma, bias, delay, 0.5)
        assert 0.07 < bl < 0.13, (g, bl)
        assert eff > (0.75 if sigma < 3 else 0.55), (g, eff)


def test_without_olla_bias_breaks_bler():
    """sanity: with the OLLA step ~ 0 an optimistic CQI bias gives a high BLER (OLLA is what fixes it)."""
    bl, _ = olla_run(8.0, 1.0, 4.0, 2, 1e-6)
    assert bl > 0.5


@pytest.fixture(scope="module")
def g():
    AE.init(False)
    return AE.G


def core(g, regime, **kw):
    return SV.UavSurveyEnv(g["table"], g["pools"]["test"], C.regime_cfg(regime, **kw))


def test_dispatch_and_old_regimes_untouched(g):
    assert type(core(g, "K5_la")) is LA.LaSurveyEnv
    assert type(core(g, "K5_mcsg")) is SV.UavSurveyEnv
    assert type(core(g, "K5")) is SV.UavSurveyEnv


def run(g, regime, name, seed, **kw):
    e = core(g, regime, **kw)
    sp, mp = MH.from_name(name, C.MODES)
    m = V.run_episode_ev(e, sp, mp, seed)
    m.update(e.la_metrics())
    return m, e


def test_episode_no_reuse_bler_and_determinism(g):
    m1, e1 = run(g, "K5_la", "fixed_mbt2018-mean_q3", 8800)
    m2, _ = run(g, "K5_la", "fixed_mbt2018-mean_q3", 8800)
    assert m1["return"] == m2["return"] and m1["bler"] == m2["bler"]
    assert m1["blocks"] > 500
    assert 0.05 < m1["bler"] < 0.2
    assert m1["shared_block_frac"] == 0.0


def test_crn_and_hidden_params(g):
    _, a = run(g, "K5_la_rand", "fixed_mbt2018-mean_q3", 8801)
    _, b = run(g, "K5_la_rand", "pacedB2+budget1+reuse", 8801)
    assert np.array_equal(a.la_u, b.la_u) and a.la_p == b.la_p           # same exogenous randomness per seed
    _, c = run(g, "K5_la_rand", "fixed_mbt2018-mean_q3", 8802)
    assert a.la_p != c.la_p
    assert 0 <= a.la_p["la_cqi_sigma"] <= 3 and 1 <= a.la_p["la_fb_delay"] <= 20


def test_reuse_raises_bler_when_feedback_is_slow(g):
    """co-channel reuse with stale CQI (delay 20 blocks) fails more often than with fresh CQI (delay 1)."""
    fast, _ = run(g, "K5_la", "pacedB2+budget1+reuse", 8803, la_fb_delay=1)
    slow, _ = run(g, "K5_la", "pacedB2+budget1+reuse", 8803, la_fb_delay=20)
    assert fast["shared_block_frac"] > 0.2
    assert slow["shared_bler"] > fast["shared_bler"]


def test_gym_obs_appends_la_features_only_with_flag(g):
    import gym_env_ev as GE
    a = GE.GymUavSemcomEv(g["table"], g["pools"]["test"], C.regime_cfg("K5_la"), seed=1, core_cls=SV.UavSurveyEnv)
    b = GE.GymUavSemcomEv(g["table"], g["pools"]["test"], C.regime_cfg("K5_la", obs_la=False), seed=1,
                          core_cls=SV.UavSurveyEnv)
    o1, _ = a.reset()
    o2, _ = b.reset()
    assert o1.shape[0] == o2.shape[0] + 5 * LA.N_FEAT
    assert np.allclose(o1[:o2.shape[0]], o2)


def test_gate_identity_when_never_triggered(g):
    a, _ = run(g, "K5_la", "pacedB2+budget1+reuse", 8804)
    b, e = run(g, "K5_la", "pacedB2+budget1+reuse+gate1c10", 8804)       # theta 1: a NACK rate can never exceed it
    assert a["return"] == b["return"] and e._gate["flags"] == 0


def test_gate_backs_off_reuse_under_failures(g):
    a, _ = run(g, "K5_la", "pacedB2+budget1+reuse", 8805, la_fb_delay=20, la_cqi_sigma=3.0)
    b, e = run(g, "K5_la", "pacedB2+budget1+reuse+gate0.15c30", 8805, la_fb_delay=20, la_cqi_sigma=3.0)
    assert e._gate["flags"] > 0
    assert b["shared_block_frac"] < a["shared_block_frac"]


def test_gate_keeps_most_urgent_and_reassigns_free_channels(g):
    import la_heur as LH
    e = core(g, "K5_la")
    sp, _ = MH.from_name("pacedB2+budget1+reuse", C.MODES)
    e.reset(8806)
    st = LH._state(e)
    st["until"] = [10 ** 6] * len(e.uavs)                                   # everybody flagged
    acts = sp(e)
    out = LH.apply_gate(e, acts, 0.5, 10)
    used = [a[1] for a in out if a[1] > 0]
    assert len(used) == len(set(used))                                     # no co-channel reuse left
