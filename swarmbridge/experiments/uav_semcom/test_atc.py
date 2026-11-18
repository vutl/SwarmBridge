"""Review #46 step C: compute charge, Qmin-constrained rule, test-dev tables, data independence, scorer, bootstrap."""
import types

import numpy as np
import pytest

import a5_common as C
import a5_eval as AE
import at1_task as AT
import atc_predict as PR
import atc_run as R
import atc_summary as SM
import env as E


@pytest.fixture(scope="module")
def g():
    AE.init(True)
    return AE.G


def test_info_extra_charged(g):
    base = E.UavSemcomEnv(g["table"], g["pools"]["train"], dict(mode_info="predicted"), seed=0)
    more = E.UavSemcomEnv(g["table"], g["pools"]["train"], dict(mode_info="predicted", info_extra_s=0.035), seed=0)
    assert abs(more.probe_s() - base.probe_s() - 0.035) < 1e-12
    assert R.cost("ja|t=rv:pC:20") == 0.035 and R.cost("ja|t=rvq:pB:5:1") == 0.010 and R.cost("ja|t=rv:count:5") == 0


def test_testdev_tables(g):
    t = AT.tables(g["table"], g["pools"])
    for p in ("train", "val", "testdev"):
        idx = np.asarray(g["pools"][p])
        assert not np.isnan(t["gt"][idx]).any() and not np.isnan(t["fp"][idx]).any() and (t["fp"][idx] >= 0).all()
        assert (t["ngt"][idx] > 0).all()                                   # no empty frames in any pool
    assert np.allclose(t["mu"], t["gt"][np.asarray(g["pools"]["train"])].mean(0))


def test_rvq_respects_floor(g):
    u = AT.Utility("count", g["table"], g["pools"])
    f = AT.rvq_rule(u, 5.0, 1.0)
    env = E.UavSemcomEnv(g["table"], g["pools"]["train"], dict(mode_info="predicted"), seed=0)
    for img in np.asarray(g["pools"]["train"])[:40]:
        for qmin in (22.0, 26.0, 30.0, 60.0):
            task = types.SimpleNamespace(img=int(img), qmin=qmin)
            m = f(env, 0, task)
            b, q = env.mode_info(task)
            if (q >= qmin + 1.0).any():
                assert q[m] >= qmin + 1.0
            else:
                assert m == int(np.argmax(q))


def test_independence():
    info = PR.check_independence(C.manifest()["images"])
    assert info["testdev"]["videos"] == 49 and info["train"]["frames"] == 304


def test_scorer_matches_definition(g):
    sc = R.Scorer()
    rows_of = {v: k for k, v in sc.rows.items()}
    tr = np.asarray(g["pools"]["train"])[:5]
    m = C.MODES.index("avif_20")
    ep = [(rows_of[int(r)], m, 0, 1) for r in tr] + [(rows_of[int(tr[0])], -1, 0, 0)]
    T = sc.t
    want = sum(T["gt"][r, m] for r in tr) / (sum(T["ngt"][r] for r in tr) + T["ngt"][tr[0]])
    assert abs(sc.ep(ep) - want) < 1e-12
    assert sc.ep(ep, "recall_qmin") == 0.0                               # ok = 0 everywhere
    assert abs(sc.ep(ep, "macro") - np.mean([T["gt"][r, m] / T["ngt"][r] for r in tr] + [0.0])) < 1e-12
    assert abs(sc.ep(ep, "fp") - sum(T["fp"][r, m] for r in tr) / 6) < 1e-12


def test_two_way_bootstrap():
    rng = np.random.default_rng(1)
    gt = rng.integers(50, 100, (20, 7)).astype(float)
    tp = 0.5 * gt
    pt, lo, hi = SM.two_way(tp, gt, boot=500)
    assert abs(pt - 0.5) < 1e-12 and abs(lo - 0.5) < 1e-9 and abs(hi - 0.5) < 1e-9
    pt, lo, hi = SM.two_way(tp + 0.1 * gt, gt, other=(tp, gt), boot=500)
    assert abs(pt - 0.1) < 1e-12 and lo > 0.09
