"""Review #42 (A-X7): spatially consistent LoS, causal LoS filter, sounding probe, LoS-only oracle, JA options."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

import a5_common as C
import a5_eval as AE
import ch_filter as CF
import env_ev as V
import env_survey as SV
import ja_sched as JA
import mode_heur as MH

NOREUSE = "fixed_avif_20"


@pytest.fixture(scope="module")
def g():
    AE.init(False)
    return AE.G


def test_bvn_cdf_matches_scipy():
    for h, k, r in [(0.3, -0.7, 0.5), (-1.2, -0.4, 0.9), (1.5, 2.0, -0.3), (0.0, 0.8, 0.2), (-0.5, 0.0, 0.0),
                    (2.5, -2.5, 0.99)]:
        ref = multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([h, k])
        assert abs(CF.bvn_cdf(h, k, r) - ref) < 2e-4, (h, k, r)


def run(g, cfg_extra, seed, name=NOREUSE, hook=None, ja=None):
    e = SV.UavSurveyEnv(g["table"], g["pools"]["test"], C.regime_cfg("K5_la", la_mode="sched",
                                                                       la_interf="persistent", **cfg_extra))
    if hook:
        hook(e)
    sp, mp = MH.from_name(name, C.MODES)
    if ja:
        sp = JA.from_name(ja, sp)
    m = V.run_episode_ev(e, sp, mp, seed)
    return m, e


def test_zero_correlation_is_the_original_env(g):
    a, _ = run(g, {}, 8800)
    b, _ = run(g, dict(ch_corr_m=0.0), 8800)
    assert a["return"] == b["return"]


def _collect(g, corr, seeds):
    rec = []

    def hook(e):
        orig = e.begin_slot

        def bs(acts):
            orig(acts)
            for k in range(len(e.uavs)):
                rec.append((e.slot, k, e._pathgain(e.uavs[k].pos)[0], bool(e.slot_los(k))))
        e.begin_slot = bs
    for s in seeds:
        run(g, dict(ch_corr_m=corr), s, hook=hook)
    return rec


def test_correlated_los_keeps_marginal_and_adds_memory(g):
    seeds = range(8800, 8806)
    r0, r100 = _collect(g, 0.0, seeds), _collect(g, 100.0, seeds)
    for rec in (r0, r100):
        p = np.array([x[2] for x in rec])
        los = np.array([x[3] for x in rec], float)
        assert abs(los.mean() - p.mean()) < 0.04                        # marginal P(LoS) preserved

    def los_after_los(rec):                                            # P(LoS now | LoS in the previous slot)
        d = {}
        for s, k, _, l in rec:
            d.setdefault(k, []).append(l)
        pairs = [b for v in d.values() for a, b in zip(v, v[1:]) if a]
        return np.mean(pairs)
    p_mean = np.mean([x[2] for x in r0])                                # LoS is rare here (~3 %)
    assert los_after_los(r0) < 3 * p_mean + 0.05                        # no memory without correlation
    assert los_after_los(r100) > 0.5                                    # strong memory at 100 m


def test_filter_beats_prior_with_correlation_and_equals_it_without(g):
    dg = []
    for seed in range(8800, 8806):                    # LoS is rare (~3 %): aggregate several episodes
        _, e = run(g, dict(ch_corr_m=100.0), seed, ja="JA:filt:NR:a1:m0")
        dg += e._ja_diag
    pr, us, lt = (np.asarray(x, float) for x in zip(*dg))
    assert ((us - lt) ** 2).mean() < ((pr - lt) ** 2).mean()
    _, e0 = run(g, {}, 8801, ja="JA:filt:NR:a1:m0")
    pr0, us0, _ = (np.asarray(x, float) for x in zip(*e0._ja_diag))
    assert np.allclose(pr0, us0)


def test_sounding_probe_refreshes_reports(g):
    _, e = run(g, dict(ch_corr_m=30.0, la_srs_period=1), 8802)
    assert e.la_st["srs"] > 0
    b = e.block_of(e.t)
    for k, ad in enumerate(e.la):
        if e.uavs[k].t_done >= 0:
            continue
        assert ad.meas and b - ad.meas[-1][0] <= 2 * int(round(e.c["dt"] / e.c["la_tb"]))


def test_los_only_oracle_uses_the_true_state(g):
    _, e = run(g, dict(ch_corr_m=30.0), 8803, ja="JA:los:NR:a1:m0")
    _, us, lt = (np.asarray(x, float) for x in zip(*e._ja_diag))
    assert np.array_equal(us, lt)


def test_member_names_and_latency_log(g):
    d = JA.parse("JA:stat:R:a1:m3:n8")
    assert d == dict(info="stat", reuse=True, alpha=1.0, margin_db=3.0, n_mc=8)
    _, e = run(g, {}, 8804, ja="JA:stat:NR:a1:m0:n8")
    assert len(e._ja_t) > 20 and max(e._ja_t) < 1.0
