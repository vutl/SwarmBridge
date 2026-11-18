"""Review #45 (A-T1): task-utility tables, rules, task MPC and the full-workload recall."""
import json

import numpy as np
import pytest

import a5_common as C
import a5_eval as AE
import at1_task as AT
import ax2_eval as X2
from ax2_summary import recall

EXTRA = dict(la_mode="sched", la_interf="persistent", obs_la=False)
S = "ja~JA:stat:R:a1:m3@pacedB3"


@pytest.fixture(scope="module")
def g():
    AE.init(False)
    return AE.G


def ep(combo, seed=3, ext=True):
    X2.DELIV_EXT = ext
    try:
        return X2.episode("K5", "K5_la", combo, -1, seed, EXTRA, None, "train")
    finally:
        X2.DELIV_EXT = False


def test_tables_match_fidelity(g):
    t = AT.tables(g["table"], g["pools"])
    fid = AT._rows([AT.FID])
    r = int(g["pools"]["train"][0])
    name = g["table"].images[r]
    assert t["gt"][r, C.MODES.index("mbt2018-mean_q3")] == fid[(name, "mbt2018-mean_q3")]["tp"]
    assert t["ngt"][r] == fid[(name, "original")]["n_gt"]
    assert np.allclose(t["mu"], t["gt"][np.asarray(g["pools"]["train"])].mean(0))
    for p in ("train", "test"):
        assert not np.isnan(t["gt"][np.asarray(g["pools"][p])]).any()


def test_extended_deliveries_and_recall(g):
    m, d = ep(f"{S}|h=fixed_avif_20")
    assert all(len(x) == 4 for x in d)
    assert all(ot >= ok for _, _, ok, ot in d)                 # ok = on time AND PSNR >= Qmin
    t = AT.tables(g["table"], g["pools"])
    rows = {int(mi): r for r, mi in enumerate(g["man_idx"]) if mi >= 0}
    per = {}
    for line in open(AT.FID):
        r = json.loads(line)
        per[(r["image"], r["mode"])] = r
    r_ok = AT.score([d], t["gt"], t["ngt"], rows, ontime=False)
    assert abs(r_ok - recall([[x[:3] for x in d]], per, C.manifest()["images"])) < 1e-12
    assert AT.score([d], t["gt"], t["ngt"], rows, ontime=True) >= r_ok


def test_rules(g):
    import types
    u = AT.Utility("gt", g["table"], g["pools"])
    env = types.SimpleNamespace(mode_info=lambda task: (np.linspace(1e5, 2e6, len(C.MODES)), None))
    task = types.SimpleNamespace(img=int(g["pools"]["train"][5]))
    assert AT.rv_rule(u, 0)(env, 0, task) == int(np.argmax(u(env, task)))
    assert AT.rv_rule(u, 1e6)(env, 0, task) == 0                  # cheapest predicted payload


def test_task_mpc_value_drops_late_images(g):
    import a9_mpc as M
    mpc = AT.TaskMPC(AT.Utility("mean", g["table"], g["pools"]), "env", 8, 4, None, seed=1, rollout="causal")
    tab = g["table"]
    import types
    task = types.SimpleNamespace(t_cap=0.0, dmax=5.0)
    G = (0.0, 1.0, np.concatenate([np.zeros((2, 1)), np.cumsum(np.full((2, M.S_SLOTS), 1e6), 1)], 1),
         np.full((2, M.S_SLOTS), 1e6))
    m = C.MODES.index("avif_20")
    v, _ = mpc._value(None, tab, G, np.zeros(2), np.zeros(2), 2e6, 7.0, task, m)      # ends at 2 s: on time
    assert np.allclose(v, 7.0)
    v, _ = mpc._value(None, tab, G, np.zeros(2), np.zeros(2), 9e6, 7.0, task, m)      # ends at 9 s: dropped
    assert np.allclose(v, 0.0)


def test_task_mpc_episode_runs_and_oracle_differs(g):
    a, _ = ep(f"{S}|h=fixed_mbt2018-mean_q3+tmpc=mean")
    b, _ = ep(f"{S}|h=fixed_mbt2018-mean_q3+tmpc=gt")
    assert a["planned"] > 0 and b["planned"] > 0 and a["mode_over50"] < 0.05
    assert a["return"] != b["return"]
    assert not X2.needs_ckpt(f"{S}|h=fixed_mbt2018-mean_q3+tmpc=gt")
