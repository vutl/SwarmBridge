"""Review #44 (A-X9): scheduler / movement syntax and the uncalibrated MPC ablation in ax2_eval."""
import pytest

import a5_eval as AE
import ax2_eval as X2

EXTRA = dict(la_mode="sched", la_interf="persistent", obs_la=False)
JA0 = "JA:stat:NR:a1:m0"


@pytest.fixture(scope="module")
def g():
    AE.init(False)
    return AE.G


def ep(combo, seed=3, ja=JA0):
    return X2.episode("K5", "K5_la", combo, -1, seed, EXTRA, ja, "train")[0]


def test_explicit_movement_reproduces_the_old_scheduler_names(g):
    assert ep("ja@pacedB2|h=fixed_mbt2018-mean_q3")["return"] == ep("ja|h=fixed_mbt2018-mean_q3")["return"]
    assert ep("ja@mission|h=fixed_avif_20")["return"] == ep("jae|h=fixed_avif_20")["return"]


def test_inline_ja_member(g):
    assert ep(f"ja~{JA0}@pacedB2|h=fixed_avif_20", ja=None)["return"] == ep("ja@pacedB2|h=fixed_avif_20")["return"]
    assert ep("ja~JA:stat:NR:a0:m0@pacedB2|h=fixed_avif_20", ja=None)["return"] != \
        ep("ja@pacedB2|h=fixed_avif_20")["return"]


def test_uncalibrated_mpc_runs_and_differs(g):
    a = ep("ja@pacedB2|h=fixed_avif_20+mpc")
    b = ep("ja@pacedB2|h=fixed_avif_20+mpcnc")
    assert b["planned"] > 0 and b["gated"] == 0 and b["mode_over50"] < 0.05
    assert a["return"] != b["return"]


def test_needs_ckpt():
    assert X2.needs_ckpt("ja~JA:stat:NR:a1:m0@pacedB2|ppola+mpc")
    assert X2.needs_ckpt("ja@mission|ppo+mpcnc")
    assert not X2.needs_ckpt("ja@mission|h=greedy+mpcnc")


def test_mpc_la_rate_matches_env_link_rate(g):
    import numpy as np
    import a5_common as C
    import a9_mpc as M
    import env_survey as SV
    e = SV.UavSurveyEnv(g["table"], g["pools"]["train"], C.regime_cfg("K5_la", **EXTRA))
    e.reset(3)
    mpc = M.MPCMode("env", 8, 4, None, seed=1, rollout="causal")
    sig = np.logspace(-14, -9, 50)
    ref = [e.link_rate(x, e.noise_w(), 0.0, x / e.noise_w()) for x in sig]
    assert np.allclose(mpc._rate(e, sig), ref)
