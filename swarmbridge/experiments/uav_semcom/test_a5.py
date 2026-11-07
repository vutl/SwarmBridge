"""Tests for the A5 additions: PHY abstraction (env.UavSemcomEnv.link_rate), manifest split, regime configs.
Run: python -m pytest -q test_a5.py"""
import json
import math

import numpy as np
import pytest

import env as E
import env_ev as V
import env_survey as S
import mode_heur as MH
import policies as P
from test_env import make_table


def env_with(tmp_path, **cfg):
    return S.UavSurveyEnv(make_table(tmp_path), range(8), dict(K=3, M=2, v_max=20.0, mode_info="predicted", **cfg))


def test_shannon_default_unchanged(tmp_path):
    env = env_with(tmp_path)
    c = env.c
    assert c["phy"] == "shannon"
    for sig, n, i in [(1e-12, 1e-14, 0.0), (3e-13, 2e-14, 5e-14), (1e-15, 1e-14, 1e-13)]:
        assert env.link_rate(sig, n, i, sig / n) == c["B_total"] / c["M"] * math.log2(1 + sig / (n + i))


def test_gap_and_mcs_rates(tmp_path):
    g = env_with(tmp_path, phy="gap", phy_gap_db=3.0)
    m = env_with(tmp_path, phy="mcs")
    sh = env_with(tmp_path)
    W = sh.c["B_total"] / sh.c["M"]
    n = 1e-14
    prev = -1.0
    for snr_db in np.arange(-12, 30, 0.37):
        sig = n * 10 ** (snr_db / 10)
        r_sh, r_g, r_m = (e.link_rate(sig, n, 0.0, sig / n) for e in (sh, g, m))
        assert r_g == pytest.approx(W * math.log2(1 + sig / n / 10 ** 0.3))
        assert r_g <= r_sh and r_m <= r_sh + 1e-9          # practical rates never exceed capacity
        assert r_m >= prev                                   # MCS rate is monotone in SINR
        prev = r_m
        if snr_db < E.CQI_THR_DB[0]:
            assert r_m == 0.0
    sig = n * 10 ** (E.CQI_THR_DB[7] / 10 + 1e-9)               # exactly at CQI 8's threshold
    assert m.link_rate(sig, n, 0.0, sig / n) == pytest.approx(W * E.CQI_EFF[7] * 0.9)


def test_mcs_blind_fails_under_interference(tmp_path):
    b = env_with(tmp_path, phy="mcs_blind")
    m = env_with(tmp_path, phy="mcs")
    n = 1e-14
    sig = n * 10 ** 1.5                                      # 15 dB SNR -> CQI 11 (14.1 dB)
    assert b.link_rate(sig, n, 0.0, sig / n) == m.link_rate(sig, n, 0.0, sig / n) > 0
    interf = n * 0.5                                         # SINR 13.2 dB < 14.1 dB: blind CQI fails, genie drops a level
    assert b.link_rate(sig, n, interf, sig / n) == 0.0
    assert m.link_rate(sig, n, interf, sig / n) > 0


@pytest.mark.parametrize("phy", ["gap", "mcs", "mcs_blind"])
def test_phy_episode_conservation(tmp_path, phy):
    env = env_with(tmp_path, phy=phy, phy_gap_db=3.0, B_total=4e5)
    m = V.run_episode_ev(env, MH.reuse(P.from_name("pacedB1.5")), V.greedy_mode, 3)
    for tk in env.finished:
        if not tk.dropped:
            assert tk.sent_bits == pytest.approx(tk.bits, rel=1e-9)
            assert tk.delay <= tk.dmax + 1e-9
    for (k, ch, t0, t1, interf, r) in env.segments:
        assert r >= 0 and t1 >= t0
    assert len(env.finished) == len(env.tasks)
    assert np.isfinite(m["return"])


def test_mcs_heuristic_rate_consistent(tmp_path):
    """Heuristics predict rates with the same PHY as the env (policies.expected_rate goes through link_rate)."""
    env = env_with(tmp_path, phy="mcs")
    env.reset(0)
    c = env.c
    for k, u in enumerate(env.uavs):
        sig = c["p_levels"][-1] * env.expected_gain(u.pos)
        assert P.expected_rate(env, k) == env.link_rate(sig, env.noise_w(), 0.0, sig / env.noise_w())


def test_split_manifest(tmp_path):
    tab = make_table(tmp_path)
    man = dict(images=[dict(name=f"im{i}", split=("train" if i < 4 else "val" if i < 6 else "test")) for i in range(7)])
    p = tmp_path / "man.json"
    p.write_text(json.dumps(man))
    pools = tab.split_manifest(str(p), probe_mode="m0")
    assert list(pools["train"]) == [0, 1, 2, 3] and list(pools["val"]) == [4, 5] and list(pools["test"]) == [6]
    assert list(pools["none"]) == [7]
    assert np.allclose(tab.mean_bits, tab.bits[:4].mean(0))    # knowledge fitted on train only


def test_regimes_one_dimensional():
    import a5_common as C
    ref = C.regime_cfg("ref")
    for name in C.REGIMES:
        cfg = C.regime_cfg(name)
        changed = {k for k in set(cfg) | set(ref) if cfg.get(k) != ref.get(k)}
        assert changed == set(C.REGIMES[name]) - {k for k, v in C.REGIMES[name].items() if ref.get(k) == v}
    assert C.regime_cfg("ref@oracle")["mode_info"] == "oracle"
