"""Production paths of ix_common (review of e964ad41, item 1): serialised retrained state, L2 lock, evaluation-time
loading with every fitting function disabled, refusal of the stage-B drivers without a verified lock. A sandbox is
built from the fd_v1 profile relabelled as an ix manifest (rl_train -> U_train, rl_val -> U_val, TD -> U_test,
F -> X1); every ix_common path constant is redirected into tmp_path. No new data."""
import json
import shutil

import numpy as np
import pytest

import env as E
import fd_common as FC
import ix_common as IX

LABEL = {"rl_train": "U_train", "rl_val": "U_val", "TD": "U_test", "F": "X1"}


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    man = json.loads(FC.MANIFEST.read_text())
    for e in man["images"]:
        e["split"] = LABEL.get(e["split"], "other")
    mp = tmp_path / "manifest_ix_v1.json"
    mp.write_text(json.dumps(man))
    split = {e["name"]: e["split"] for e in man["images"]}
    prof = {k: tmp_path / f"profile_{k}.jsonl" for k in ("A", "uavdt", "x1")}
    fo = {k: open(p, "w") for k, p in prof.items()}
    stage_of = {"U_train": "A", "U_val": "A", "U_test": "uavdt", "X1": "x1"}
    for line in open(FC.PROFILE):
        s = stage_of.get(split[json.loads(line)["image"]])
        if s:
            fo[s].write(line)
    for f in fo.values():
        f.close()
    shutil.copy(FC.CPU_EST, tmp_path / "cpu_est.json")
    monkeypatch.setattr(IX, "MANIFEST", mp)
    monkeypatch.setattr(IX, "MANIFEST_SHA", IX.sha_text(mp))
    monkeypatch.setattr(IX, "PROFILE", prof)
    for name, fn in (("CPU_EST", "cpu_est.json"), ("STATE_FILE", "state_retrained.json"), ("TUNE_FILE", "tune.json"),
                     ("CKPTS", "ckpts.json"), ("SELECTION", "selection.json"), ("LOCK", "L2_lock.json")):
        monkeypatch.setattr(IX, name, tmp_path / fn)
    return tmp_path


def stage_a_outputs(tmp_path):
    """stand-ins for the stage-A outputs that the lock pins (tune, checkpoints, selection)."""
    IX.fit_retrained()
    (tmp_path / "tune.json").write_text(json.dumps(dict(H={"N5x2": "fixed_avif_20"})))
    sel = {}
    for k in ("ppoix", "qrix", "dsix"):
        for s in range(3):
            p = tmp_path / f"{k}_s{s}.pt"
            p.write_bytes(f"weights {k} {s}".encode())
            sel.setdefault(k, {})[str(s)] = dict(ckpt=str(p), ckpt_sha256=IX.sha_file(p))
    (tmp_path / "selection.json").write_text(json.dumps(dict(selection=sel)))
    (tmp_path / "ckpts.json").write_text(json.dumps({k: [v[str(s)]["ckpt"] for s in range(3)] for k, v in sel.items()}))


def write_lock():
    IX.LOCK.write_text(json.dumps(dict(created="test", items=IX.lock_items())))


def test_state_file_round_trip_is_exact(sandbox):
    stage_a_outputs(sandbox)
    st = IX.load_state(IX.STATE_FILE)
    t, rows, man, sel = IX.build_table([IX.PROFILE["A"]], IX.MANIFEST, IX.SPLITS["A"])
    ref = IX.refit_state(t, rows, sel, "U_train", IX.CPU_EST)
    assert IX.state_hash(st) == IX.state_hash(ref)
    for k in IX.FROZEN_KEYS:
        assert np.array_equal(st[k], ref[k]), k


def test_fit_requires_registered_cpu_estimate(sandbox):
    IX.CPU_EST.unlink()
    with pytest.raises(IX.LockError):
        IX.fit_retrained()
    t, rows, man, sel = IX.build_table([IX.PROFILE["A"]], IX.MANIFEST, IX.SPLITS["A"])
    with pytest.raises(IX.LockError):
        IX.refit_state(t, rows, sel, "U_train", None)


def test_fit_runs_once(sandbox):
    stage_a_outputs(sandbox)
    with pytest.raises(IX.LockError):
        IX.fit_retrained()


def test_stage_a_table_loads_serialised_state(sandbox, monkeypatch):
    stage_a_outputs(sandbox)
    monkeypatch.setattr(IX, "refit_state", lambda *a, **k: pytest.fail("stage A must not refit"))
    monkeypatch.setattr(E.ModeTable, "fit_info", lambda *a, **k: pytest.fail("fit_info called"))
    t, pools, man_idx, st = IX.load_stage_a()
    assert IX.state_hash(IX.state_of(t)) == json.loads(IX.STATE_FILE.read_text())["state_hash"]
    assert set(pools) >= {"U_train", "U_val", "train", "val"} and "U_test" not in pools


def test_eval_loading_with_all_fitting_disabled(sandbox, monkeypatch):
    stage_a_outputs(sandbox)
    write_lock()
    monkeypatch.setattr(IX, "refit_state", lambda *a, **k: pytest.fail("refit_state called at evaluation"))
    monkeypatch.setattr(IX, "fit_retrained", lambda *a, **k: pytest.fail("fit_retrained called at evaluation"))
    monkeypatch.setattr(E.ModeTable, "fit_info", lambda *a, **k: pytest.fail("fit_info called at evaluation"))
    t, pools, rec, man_idx, st, man = IX.load_eval("uavdt", "retrained")
    assert IX.state_hash(st) == json.loads(IX.LOCK.read_text())["items"]["state_retrained_hash"]
    assert set(pools) == {"U_test"}                                   # no train / val rows in an evaluation table
    t2, pools2, _, _, st2, _ = IX.load_eval("uavdt", "frozen")
    assert IX.state_hash(st2) == IX.FROZEN_EXPECT
    t3, pools3, rec3, _, _, _ = IX.load_eval("x1", "frozen")
    assert set(pools3) == {"X1"} and rec3 is None
    with pytest.raises(IX.RefitForbidden):
        t.fit_info(np.arange(5), "avif_5")


@pytest.mark.parametrize("tamper", ["no_lock", "state", "checkpoint", "tune", "profile_A", "code"])
def test_lock_verification_fails_on_any_change(sandbox, monkeypatch, tamper):
    stage_a_outputs(sandbox)
    write_lock()
    IX.verify_lock()
    if tamper == "no_lock":
        IX.LOCK.unlink()
    elif tamper == "state":
        d = json.loads(IX.STATE_FILE.read_text())
        d["arrays"]["mean_bits"][0] *= 1.0000001
        IX.STATE_FILE.write_text(json.dumps(d))
    elif tamper == "checkpoint":
        (sandbox / "qrix_s1.pt").write_bytes(b"changed")
    elif tamper == "tune":
        IX.TUNE_FILE.write_text(json.dumps(dict(H={"N5x2": "greedy"})))
    elif tamper == "profile_A":
        with open(IX.PROFILE["A"], "a") as f:
            f.write("\n")
    else:
        monkeypatch.setattr(IX, "sha_text", lambda p, _o=IX.sha_text: "x" if str(p).endswith("a9_mpc.py") else _o(p))
    with pytest.raises((IX.LockError, json.JSONDecodeError)):
        IX.verify_lock()
    with pytest.raises((IX.LockError, json.JSONDecodeError)):
        IX.load_eval("uavdt", "retrained")


def test_stage_b_drivers_refuse_without_lock(sandbox, monkeypatch):
    import ix_eval as EV
    import ix_profile as XP
    stage_a_outputs(sandbox)
    monkeypatch.setattr(XP.IX, "verify_lock", IX.verify_lock)
    for stage in ("uavdt", "x1"):
        with pytest.raises(IX.LockError):
            XP.run(type("A", (), dict(stage=stage, limit=0))())
    monkeypatch.setattr(EV, "paths", lambda b: (sandbox / f"{b}.json", sandbox / f"{b}.deliv.json.gz"))
    for block in EV.BLOCKS:
        with pytest.raises(IX.LockError):
            EV.run_block(block, 1)
