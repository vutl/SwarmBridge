"""Review #41: atomic block accounting, scheduler-aware / oracle link adaptation, interference assumptions, joint
channel assignment (env_la, ja_sched)."""
import gzip
import json
import math

import numpy as np
import pytest

import a5_common as C
import a5_eval as AE
import env_ev as V
import env_la as LA
import env_survey as SV
import ja_sched as JA
import mode_heur as MH

REUSE = "pacedB2+budget1+reuse"
NOREUSE = "fixed_mbt2018-mean_q3"


@pytest.fixture(scope="module")
def g():
    AE.init(False)
    return AE.G


def run(g, name, seed, regime="K5_la", hook=None, **kw):
    e = SV.UavSurveyEnv(g["table"], g["pools"]["test"], C.regime_cfg(regime, **kw))
    if hook:
        hook(e)
    sp, mp = MH.from_name(name, C.MODES)
    m = V.run_episode_ev(e, sp, mp, seed)
    m.update(e.la_metrics())
    return m, e


# ---------- unit: rollback ----------
def test_rollback_restores_bits_and_requeues_completed_images_in_order(g):
    e = SV.UavSurveyEnv(g["table"], g["pools"]["test"], C.regime_cfg("K5_la"))
    e.reset(8800)
    from env import Task
    t1, t2, t3 = (Task(uav=0, wp=i, tile=0, img=0, t_cap=0.0, dmax=10.0, qmin=0.0, mode=0) for i in range(3))
    t1.bits, t1.rem_bits, t1.sent_bits = 100.0, 0.0, 100.0        # completed in the block (all 100 bits)
    t2.bits, t2.rem_bits, t2.sent_bits = 200.0, 0.0, 200.0        # completed in the block (150 of its bits here)
    t3.bits, t3.rem_bits, t3.sent_bits = 300.0, 250.0, 50.0       # head, partly sent in the block
    e.uavs[0].queue = [t3]
    rec = dict(prov=[[t1, 100.0], [t2, 150.0], [t3, 50.0]], awaiting=[(t1, 0.01), (t2, 0.03)], bits=300.0)
    e._la_rollback(0, rec)
    assert e.uavs[0].queue == [t1, t2, t3]
    assert (t1.rem_bits, t2.rem_bits, t3.rem_bits) == (100.0, 150.0, 300.0)
    assert (t1.sent_bits, t2.sent_bits, t3.sent_bits) == (0.0, 50.0, 0.0)
    assert e.la_st["tasks_reopened"] == 2 and e.la_st["bits_rolled_back"] == 300.0
    assert not t1.done and not t2.done


# ---------- integration: invariants of the atomic accounting ----------
@pytest.mark.parametrize("regime", ["K5_la", "K5_la_bad"])
def test_atomic_invariants_with_mid_block_interference(g, regime):
    commits = []

    def hook(e):
        orig = e._finish

        def fin(tk, t_end):
            commits.append((e.t, t_end))
            return orig(tk, t_end)
        e._finish = fin
    tot_reopened = 0
    for seed in (8800, 8801, 8802):
        commits.clear()
        m, e = run(g, REUSE, seed, regime, hook=hook)
        tot_reopened += m["tasks_reopened"]
        ids = [id(tk) for tk in e.finished]
        assert len(ids) == len(set(ids))                          # no image completes or drops twice
        for tk in e.finished:
            if not tk.dropped:
                assert abs(tk.sent_bits - tk.bits) < 1e-6 * max(tk.bits, 1)   # no double-counted / lost bits
                assert tk.rem_bits == 0.0
        tb = e.c["la_tb"]
        for t_commit, t_end in commits:                            # completion only after its block closed
            end_of_block = (math.floor((t_end - 1e-9) / tb) + 1) * tb
            assert t_commit + 1e-9 >= end_of_block, (t_commit, t_end)
        delivered = sum(tk.bits for tk in e.finished if not tk.dropped)
        assert e.la_st["bits_ok"] + 1e-3 >= delivered               # committed bits cover every delivered image
    if regime == "K5_la":
        assert tot_reopened > 0                                     # the retransmission path is exercised


def test_reopened_image_past_deadline_is_dropped_not_finished(g):
    """an image re-opened by a NACK after its deadline can no longer be met must be dropped exactly once."""
    reopened_late = []

    def hook(e):
        orig = e._la_rollback

        def rb(k, rec):
            tks = [tk for tk, _ in rec["awaiting"]]
            orig(k, rec)
            reopened_late.extend(tks)
        e._la_rollback = rb
    for seed in range(8800, 8806):
        reopened_late.clear()
        _, e = run(g, REUSE, seed, "K5_la_bad", hook=hook)
        for tk in reopened_late:
            assert tk.done and e.finished.count(tk) == 1
            if tk.dropped:
                assert tk.util == -1.0


def test_no_reuse_is_unaffected_by_atomicity(g):
    """without co-channel interference no block flips mid-way, so atomic == old accounting for the return
    (completion times are identical: the time of the last bit)."""
    a, _ = run(g, NOREUSE, 8800, la_atomic=False)
    b, _ = run(g, NOREUSE, 8800)
    assert abs(a["return"] - b["return"]) < 1e-6
    assert b["bits_rolled_back_frac"] == 0.0


def test_old_accounting_reproduces_stored_ax5_rows(g):
    if not C.REPO.joinpath("runs/uav_semcom/ax5/eval.json.gz").exists():
        pytest.skip("A-X5 results not present")
    d = json.load(gzip.open(C.REPO / "runs/uav_semcom/ax5/eval.json.gz", "rt"))
    for spec, fam in (("T:reuse@K5_la_rand", "reuse"), ("T:best@K5_la", "best")):
        reg = spec.split("@")[1]
        r = [x for x in d["results"] if x["env_regime"] == "K5_la" and x["spec"] == spec][0]
        name = AE.tuned(reg, fam)
        for s, row in list(zip(r["seeds"], r["rows"]))[:2]:
            m, _ = run(g, name, s, la_atomic=False)
            assert abs(m["return"] - row["return"]) < 1e-9


# ---------- link adaptation modes ----------
def test_sched_equals_blind_without_interference(g):
    a, _ = run(g, NOREUSE, 8801)
    b, _ = run(g, NOREUSE, 8801, la_mode="sched")
    assert abs(a["return"] - b["return"]) < 1e-6 and a["bler"] == b["bler"]


def test_sched_predicts_shared_blocks_and_oracle_beats_blind(g):
    s, _ = run(g, REUSE, 8802, la_mode="sched", la_interf="persistent")
    assert s["shared_block_frac"] > 0.2 and s["unpred_shared_frac"] < 0.05     # persistent: all predictable
    bl, _ = run(g, REUSE, 8802, la_interf="persistent")
    orc, _ = run(g, REUSE, 8802, la_mode="oracle", la_interf="persistent")
    assert s["shared_bler"] < bl["shared_bler"]
    assert orc["goodput_kbps"] > bl["goodput_kbps"]


def test_persistent_vs_bursty_without_reuse_differs_only_in_padding_energy(g):
    a, _ = run(g, NOREUSE, 8803)
    b, _ = run(g, NOREUSE, 8803, la_interf="persistent")
    assert a["bler"] == b["bler"] and a["ok_rate"] == b["ok_rate"] and a["shared_block_frac"] == 0.0
    assert b["pad_air_s"] >= 0.0 and b["energy_kj"] >= a["energy_kj"] - 1e-9


def test_block_fading_runs_and_differs(g):
    a, _ = run(g, NOREUSE, 8804)
    b, _ = run(g, NOREUSE, 8804, la_fade="block")
    assert a["return"] != b["return"] and 0.0 < b["bler"] < 0.5


def test_non_atomic_only_for_ax5_reproduction(g):
    with pytest.raises(AssertionError):
        SV.UavSurveyEnv(g["table"], g["pools"]["test"], C.regime_cfg("K5_la", la_atomic=False, la_mode="sched"))


# ---------- joint assignment ----------
def ja_run(g, member, seed, **kw):
    e = SV.UavSurveyEnv(g["table"], g["pools"]["test"], C.regime_cfg("K5_la", **kw))
    sp, mp = MH.from_name(NOREUSE, C.MODES)
    m = V.run_episode_ev(e, JA.from_name(member, sp), mp, seed)
    m.update(e.la_metrics())
    return m, e


def test_ja_no_reuse_never_shares_and_enumerates_all(g):
    assert len(JA.assignments(5, 2, True)) == 243 and len(JA.assignments(5, 2, False)) == 1 + 5 + 5 + 20
    m, e = ja_run(g, "JA:stat:NR:a1:m0", 8805)
    assert m["shared_block_frac"] == 0.0
    assert all(x[0] >= x[1] - 1e-9 for x in e._ja_log)


def test_ja_oracle_reads_only_pre_drawn_fading(g):
    """the oracle scheduler must not change the exogenous randomness (common random numbers)."""
    _, a = ja_run(g, "JA:oracle:R:a1:m0", 8806, la_interf="persistent")
    _, b = ja_run(g, "JA:stat:R:a1:m3", 8806, la_interf="persistent")
    assert np.array_equal(a.la_u, b.la_u) and np.array_equal(a.f_e, b.f_e)
