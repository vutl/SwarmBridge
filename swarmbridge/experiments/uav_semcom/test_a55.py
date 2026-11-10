"""A5.5 checks (pre-registered, reports/archive/uav-semcom-a/uav-semcom-a5.5.md 0.1): (a) with calibration off the rebuilt heuristics act
exactly as the A5 ones; (b) the self-calibrating heuristic is causal (changing the future does not change the present
action); (c) a sanity check of the calibration. Uses the real A5 mode table (skipped if absent)."""
import copy

import numpy as np
import pytest

import a5_common as C
import a55_heur as H
import env_ev as V
import env_survey as SV
import mode_heur as MH

pytestmark = pytest.mark.skipif(not C.MANIFEST.exists(), reason="A5 data not available")
T = {}


def table():
    if not T:
        T["t"], T["p"] = C.load_table(False)
    return T["t"], T["p"]


def episode(name, regime, seed, use_a55):
    tab, pools = table()
    core = SV.UavSurveyEnv(tab, pools["train"], C.regime_cfg(regime))
    sp, mp = (H.from_name if use_a55 else MH.from_name)(name, C.MODES)
    m = V.run_episode_ev(core, sp, mp, seed)
    return m, list(core.decisions), [(tk.img, tk.mode, tk.ok, tk.delay) for tk in core.finished]


@pytest.mark.parametrize("regime", ["ref", "K5", "phy_mcs_blind"])
@pytest.mark.parametrize("name", ["greedy+budget1", "pacedB1.5+budget1.25+reuse", "pacedB2+budget1",
                                  "pacedB3+budget3+reuse", "greedy+budget0.5+reuse"])
def test_calibration_off_equals_a5(regime, name):
    for seed in (0, 1):
        a = episode(name, regime, seed, False)
        b = episode("a5x:" + name, regime, seed, True)
        assert a[1] == b[1] and a[2] == b[2]
        assert a[0]["return"] == b[0]["return"]


def test_family_names_parse():
    for p in H.PREFIX:
        fam = H.family(p)
        assert len(fam) == 56 and len(set(fam)) == 56
        for n in fam[:3] + fam[-3:]:
            H.from_name(n, C.MODES)


def _run_until(core, sp, mp, seed, n_slots):
    core.reset(seed)
    for _ in range(n_slots):
        core.begin_slot([a[:3] for a in sp(core)])
        while True:
            ev = core.advance()
            if ev[0] == "mode":
                core.decide(mp(core, ev[1], ev[2]))
            else:
                break


@pytest.mark.parametrize("regime", ["K5", "phy_mcs_blind"])
def test_sc_is_causal(regime):
    tab, pools = table()
    name = "sc:pacedB2+budget1+reuse"
    for n_slots in (25, 60):
        core = SV.UavSurveyEnv(tab, pools["train"], C.regime_cfg(regime))
        sp, mp = H.from_name(name, C.MODES)
        _run_until(core, sp, mp, 3, n_slots)
        alt = copy.deepcopy(core)
        rng = np.random.default_rng(99)
        s = core.slot
        alt.f_u[s:] = rng.random(alt.f_u[s:].shape)              # the future (incl. this slot's) fading
        alt.f_n[s:] = rng.normal(size=alt.f_n[s:].shape)
        alt.f_e[s:] = rng.exponential(size=alt.f_e[s:].shape)
        alt.task_img = rng.permutation(alt.task_img.ravel()).reshape(alt.task_img.shape)  # images not captured yet
        a1, a2 = sp(core), sp(alt)
        assert [x[:3] for x in a1] == [x[:3] for x in a2]
        c1, c2 = H.Cal(core, True), H.Cal(alt, True)
        assert (c1.rho, c1.rho_b, c1.s) == (c2.rho, c2.rho_b, c2.s)


def test_calibration_direction():
    """Sanity (not a pre-registered condition): with channel reuse, the interference-BLIND MCS loses co-channel
    blocks that the interference-aware MCS adapts to, so the observed service rate relative to the model is lower
    (rho smaller); calibration factors stay positive and s >= 1."""
    tab, pools = table()
    out = {}
    for regime in ("phy_mcs", "phy_mcs_blind"):
        core = SV.UavSurveyEnv(tab, pools["train"], C.regime_cfg(regime, K=5))
        sp, mp = H.from_name("sc:greedy+budget3+reuse", C.MODES)
        _run_until(core, sp, mp, 5, 80)
        c = H.Cal(core, True)
        assert c.s >= 1.0 and 0 < c.rho and 0 < c.rho_b
        out[regime] = c.rho
    assert out["phy_mcs_blind"] < out["phy_mcs"]
