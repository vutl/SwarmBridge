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


def test_unit_scale_check_does_not_change_decision(env_at_decision):
    core, k, tk = env_at_decision
    a = M.MPCMode("env", 8, 4, seed=3, rollout="causal", margin_z=1.0)(core, k, tk, incumbent=0)
    m = M.MPCMode("env", 8, 4, seed=3, rollout="causal", margin_z=1.0)
    m.scales = [1.0]
    assert m(core, k, tk, incumbent=0) == a and m.robust_rejects == 0


def _stub_env():
    import types
    return types.SimpleNamespace(c=dict(p_levels=(1.0,), dt=30.0), uavs=[types.SimpleNamespace(pos=None)],
                                 tasks=object(), t=0.0, segments=[], expected_gain=lambda pos: 1.0, noise_w=lambda: 1e-3)


def test_tracker_stats_and_snapshot():
    import types
    cal = types.SimpleNamespace(mean=lambda env, k: 1e6)
    trk = M.RateErrTracker(types.SimpleNamespace(model="cal", cal=cal))
    env = _stub_env()
    obs = [0.8e6, 1.2e6, 0.9e6, 1.1e6]
    trk.update(env)
    for i, r in enumerate(obs):
        env.segments.append((0, 0, env.t, env.t + 30.0, 0.0, r))
        env.t += 30.0
        trk.update(env)
    b, se = trk.stats(env, 0)
    assert abs(b) < 1e-9 and abs(se - np.std(np.array(obs) / 1e6, ddof=1) / 2) < 1e-9
    trk.freeze()
    env2 = _stub_env()                          # new episode: restart from the frozen state, not from scratch
    env2.segments.append((0, 0, 0.0, 30.0, 0.0, 2e6))
    trk.update(env2); env2.t = 30.0; trk.update(env2)
    assert trk.stats(env2, 0)[0] > 0.1
    env3 = _stub_env()
    trk.update(env3)
    assert abs(trk.stats(env3, 0)[0]) < 1e-9   # the 2e6 slot of the previous episode is not carried over


def test_screen_and_candidate_restriction(env_at_decision):
    core, k, tk = env_at_decision
    m = M.MPCMode("env", 8, 4, seed=5, rollout="causal")
    sv = m.screen(core, k, tk, 1)
    assert sv.shape == (core.n_modes,) and m.h == 4
    m.cands = [int(np.argmin(sv))]
    assert m(core, k, tk, incumbent=0) in (0, m.cands[0])


# ---------------------------------------------------------------- A-E8
def test_stacked_groups_equal_separate_evaluations(env_at_decision):
    """group-wise causal rollout: each scale group of a stacked context equals a separate evaluation."""
    core, k, tk = env_at_decision
    m = M.MPCMode("env", 8, 4, seed=11, rollout="causal")
    rates = m._rates(core, k)
    scales = [1.0, 0.6, 1.4]
    cs = m.stacked_ctx(core, k, tk, rates, scales)
    ns = len(rates)
    for mode in range(core.n_modes):
        v = m._eval_mode(core, cs, tk, mode)
        for j, sc in enumerate(scales):
            sep = m._eval_mode(core, m._ctx(core, k, tk, rates if sc == 1.0 else np.maximum(rates * sc, 1.0)), tk, mode)
            assert np.allclose(v[j * ns:(j + 1) * ns], sep)


@pytest.mark.parametrize("cand", ["all", "n2", "np"])
def test_stacked_decision_identical(env_at_decision, cand):
    core, k, tk = env_at_decision
    out = []
    for stack in (False, True):
        m = M.MPCMode("env", 8, 4, seed=21, rollout="causal", margin_z=1.0)
        m.scales = [0.8, 1.2]
        out.append((m.selective(core, k, tk, 0, 0.8, cand=cand, trig="any", stack=stack), m.last_cands, m.last_full))
    assert out[0] == out[1]


def test_selective_n2_equals_shyb_path(env_at_decision):
    core, k, tk = env_at_decision
    a = M.MPCMode("env", 8, 4, seed=31, rollout="causal", margin_z=1.0)
    a.scales = [0.8, 1.2]
    sv = a.screen(core, k, tk, 1)
    order = [m for m in np.argsort(-sv) if m != 0]
    if sv[order[0]] <= sv[0]:
        ref = 0
    else:
        a.cands = [int(m) for m in order[:2]]
        ref = a(core, k, tk, incumbent=0)
    b = M.MPCMode("env", 8, 4, seed=31, rollout="causal", margin_z=1.0)
    b.scales = [0.8, 1.2]
    assert b.selective(core, k, tk, 0, 0.8, cand="n2") == ref


def test_deadline_fallback_and_no_binding(env_at_decision):
    import time
    core, k, tk = env_at_decision
    m = M.MPCMode("env", 8, 4, seed=41, rollout="causal", margin_z=1.0)
    m.scales = [0.8, 1.2]
    dl = M.Deadline(1e-3)                        # nothing fits: incumbent, no planning work started
    dl.start(time.perf_counter())
    assert m.selective(core, k, tk, 5, 0.8, cand="np", stack=True, dl=dl) == 5 and m.last_abort == "skip_screen"
    dl2 = M.Deadline(1e-3)
    dl2.start(time.perf_counter())
    assert m.selective(core, k, tk, 5, 0.8, cand="all", stack=True, dl=dl2) == 5 and m.last_abort == "skip_full"
    ref = M.MPCMode("env", 8, 4, seed=42, rollout="causal", margin_z=1.0)
    ref.scales = [0.8, 1.2]
    big = M.MPCMode("env", 8, 4, seed=42, rollout="causal", margin_z=1.0)
    big.scales = [0.8, 1.2]
    dl3 = M.Deadline(1e9)
    dl3.start(time.perf_counter())
    assert big.selective(core, k, tk, 0, 0.8, cand="np", stack=True, dl=dl3) == \
        ref.selective(core, k, tk, 0, 0.8, cand="np", stack=True) and big.last_abort is None
    assert dl3.obs and all(len(v) > 0 for v in dl3.obs.values())


def test_deadline_estimates_learn_from_measurements():
    dl = M.Deadline(50.0, prior_ms=1.0)
    for _ in range(30):
        dl.record("full1", 4, 8.0)                # 2 ms per unit
    assert abs(dl.unit("full1") - 2.0) < 1e-9 and dl.unit("screen1") == 1.0


def _fb_episode(fb_c, fb_t, seed=8000):
    """one cold phy_mcs_blind episode (heuristic) with a slot-weighted calibrator + tracker; per slot boundary the
    number of observations in the calibrator / tracker."""
    import env_ev as V
    sp0, mp = MH.from_name(AE.tuned("phy_mcs_blind", "best"), C.MODES)
    cal = M.CalRates(slot_weight=True, fb=fb_c)
    mpc = M.MPCMode("cal", 8, 4, cal=cal)
    trk = M.RateErrTracker(mpc, fb=fb_t)
    hist = []

    def sp(env):
        cal.update(env)
        trk.update(env)
        hist.append((sum(len(v[0]) for v in cal.bins.values()), sum(a[5] for a in trk.bins.values())))
        return sp0(env)
    V.run_episode_ev(SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg("phy_mcs_blind")), sp, mp, seed)
    return cal, trk, hist


def test_feedback_zero_delay_zero_loss_is_immediate(env_at_decision):
    c0, t0, h0 = _fb_episode(None, None)
    c1, t1, h1 = _fb_episode(M.Feedback(0, 0.0), M.Feedback(0, 0.0))
    assert h0 == h1 and h0[-1][0] > 50
    assert c0.bins == c1.bins and c0.num == c1.num and t0.bins == t1.bins


def test_feedback_delay_shifts_and_loss_is_shared(env_at_decision):
    _, _, h0 = _fb_episode(None, None)
    _, _, h2 = _fb_episode(M.Feedback(2, 0.0), M.Feedback(2, 0.0))
    assert [x[0] for x in h2[2:]] == [x[0] for x in h0[:-2]]        # reports usable exactly 2 slots later
    c, t, h = _fb_episode(M.Feedback(0, 0.4, seed=5), M.Feedback(0, 0.4, seed=5))
    assert all(a == b for a, b in h)                                  # calibrator and tracker lose the same reports
    assert {k: len(v[0]) for k, v in c.bins.items()} == {k: int(a[5]) for k, a in t.bins.items()}
    frac = c.fb.n_lost / c.fb.n_sent
    assert 0.3 < frac < 0.5 and c.fb.n_used == c.fb.n_sent - c.fb.n_lost
    c2, _, _ = _fb_episode(M.Feedback(0, 0.4, seed=6), M.Feedback(0, 0.4, seed=6))
    assert c2.bins != c.bins                                          # the loss draws depend on the seed


def test_recency_window_drops_oldest():
    """A-E11: with window W the tracker's residual statistics and the calibrator's bin hold only the last W slots."""
    import types
    cal = types.SimpleNamespace(mean=lambda env, k: 1e6)
    trk = M.RateErrTracker(types.SimpleNamespace(model="cal", cal=cal), window=3)
    env = _stub_env()
    obs = [2e6, 2e6, 1e6, 1e6, 1e6]                      # the old (2e6) observations must age out
    trk.update(env)
    for r in obs:
        env.segments.append((0, 0, env.t, env.t + 30.0, 0.0, r))
        env.t += 30.0
        trk.update(env)
    b, se = trk.stats(env, 0)
    assert abs(b) < 1e-9 and se < 1e-9
    cr = M.CalRates(slot_weight=True, window=2)
    for r in (5.0, 6.0, 7.0):
        cr._add(0, r, 30.0, 10.0)
    assert cr.bins[0][0] == [6.0, 7.0] and len(cr.rho_hist) == 3
    big = M.CalRates(slot_weight=True)                   # default: unchanged, keeps everything
    for r in (5.0, 6.0, 7.0):
        big._add(0, r, 30.0, 10.0)
    assert big.bins[0][0] == [5.0, 6.0, 7.0]


def _monitor_run(cal_regime, test_regime, seed=8003):
    """warm-start calibrator from 3 heuristic episodes in cal_regime, then an OnlineMonitor over one test episode."""
    import env_ev as V
    sp, mp = MH.from_name(AE.tuned(test_regime, "best"), C.MODES)
    cal = M.CalRates(slot_weight=True)

    def sp_cal(env):
        cal.update(env)
        return sp(env)
    for s in range(9000, 9010):
        V.run_episode_ev(SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["val"], C.regime_cfg(cal_regime)), sp_cal, mp, s)
    mon = M.OnlineMonitor(cal)
    seen = []

    def sp_test(env):
        mon.update(env)
        seen.append(mon.n)
        return sp(env)
    V.run_episode_ev(SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(test_regime)), sp_test, mp, seed)
    return mon, seen


def test_online_monitor_alarms_on_stale_not_on_matched():
    AE.init(False)
    for seed in (8003, 8004):
        mon, seen = _monitor_run("ref", "phy_mcs_blind", seed)   # Shannon warm start, MCS deployment: over-prediction
        assert mon.alarm_t is not None and mon.alarm_t < 40.0
        assert seen[0] == 0                               # starts empty in the test episode (no historical ratios)
        mon2, _ = _monitor_run("phy_mcs_blind", "phy_mcs_blind", seed)
        assert mon2.alarm_t is None and mon2.n >= mon2.n_min


def test_feedback_delay10_neither_model_uses_early_reports(env_at_decision):
    """review 2026-11-22: with a 10-slot report delay, neither the calibrator nor the error tracker holds a report
    before it is allowed (counts per slot boundary are the immediate counts shifted by exactly 10 slots)."""
    _, _, h0 = _fb_episode(None, None)
    _, _, h10 = _fb_episode(M.Feedback(10, 0.0), M.Feedback(10, 0.0))
    assert all(x == (0, 0) for x in h10[:10])
    assert h10[10:] == h0[:-10]


def test_ax2_eval_feedback_reaches_both_rate_models(env_at_decision, monkeypatch):
    """review 2026-11-22: the +mpcfb arms give the calibrator AND the error tracker the same delayed report schedule."""
    import ax2_eval as X2
    seen = {}
    orig_c, orig_t = M.CalRates.__init__, M.RateErrTracker.__init__

    def cal_init(self, *a, **kw):
        seen["cal"] = kw.get("fb")
        orig_c(self, *a, **kw)

    def trk_init(self, *a, **kw):
        seen["trk"] = kw.get("fb")
        orig_t(self, *a, **kw)

    monkeypatch.setattr(M.CalRates, "__init__", cal_init)
    monkeypatch.setattr(M.RateErrTracker, "__init__", trk_init)
    S = "ja~JA:stat:R:a1:m3@pacedB3"
    X2.episode("K5", "K5_la", f"{S}|h=fixed_mbt2018-mean_q3+mpcfb10l30", -1, 9900,
               dict(la_mode="sched", la_interf="persistent", obs_la=False), None, "test")
    fc, ft = seen["cal"], seen["trk"]
    assert fc is not None and ft is not None and fc is not ft
    assert (fc.delay, fc.loss, fc.seed) == (ft.delay, ft.loss, ft.seed) == (10, 0.3, 31)
    assert fc.n_sent == ft.n_sent > 0 and fc.n_lost == ft.n_lost
