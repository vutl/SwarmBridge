"""paperA_ix loader checks (pre-registration §3.1 / review of ebf9dcc2 item 3). Uses the fd_v1 profile relabelled as a
paperA_ix manifest (rl_train -> U_train, rl_val -> U_val, TD -> U_test, everything else -> X1), so the frozen and the
refitted paths can be compared with fd_common bit for bit. No new data, profiling or detector is involved."""
import json

import numpy as np
import pytest

import a9_mpc as M
import ax2_eval as X2
import fd_common as FC
import fd_phase2 as P2
import ix_common as IX

LABEL = {"rl_train": "U_train", "rl_val": "U_val", "TD": "U_test"}


@pytest.fixture(scope="module")
def fd():
    t, pools, rec, man_idx = FC.load()
    return dict(t=t, pools=pools, st=IX.state_of(t))


@pytest.fixture(scope="module")
def ixman(tmp_path_factory):
    d = tmp_path_factory.mktemp("ix")
    man = json.loads(FC.MANIFEST.read_text())
    for e in man["images"]:
        e["split"] = LABEL.get(e["split"], "X1")
    p = d / "manifest_ix_test.json"
    p.write_text(json.dumps(man))
    return p


def perturbed_profile(tmp, keep_split, ixman, which):
    """copy of the fd_v1 profile with bits x2, PSNR + 3 dB and enc / dec x 5 on the rows whose (relabelled) split
    satisfies which(split)."""
    split = {e["name"]: e["split"] for e in json.loads(ixman.read_text())["images"]}
    out = tmp / f"profile_{keep_split}.jsonl"
    with open(FC.PROFILE) as fi, open(out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            if r["mode"] != "original" and which(split[r["image"]]):
                r["bytes"] *= 2
                r["psnr"] += 3.0
                r["enc_s"] *= 5
                r["dec_s"] *= 5
            fo.write(json.dumps(r) + "\n")
    return out


def test_frozen_state_is_fd_v1_bit_for_bit(fd, ixman):
    t, pools, rec, man_idx, st = IX.load(FC.PROFILE, ixman, "frozen")
    assert IX.state_hash(st) == IX.state_hash(fd["st"])
    for k in IX.FROZEN_KEYS:
        assert np.array_equal(getattr(t, k), getattr(fd["t"], k)), k
    assert np.array_equal(t.bits, fd["t"].bits) and np.array_equal(t.psnr, fd["t"].psnr)
    assert np.array_equal(pools["U_test"], fd["pools"]["TD"])


def test_pinned_hash_if_set(fd):
    if IX.FROZEN_EXPECT is not None:
        assert IX.state_hash(fd["st"]) == IX.FROZEN_EXPECT


def test_serialised_frozen_file_is_fd_v1(fd):
    st = IX.frozen_state()
    for k in IX.FROZEN_KEYS:
        assert np.array_equal(st[k], fd["st"][k]), k
    assert st["probe_j"] == fd["st"]["probe_j"] and st["modes"] == fd["st"]["modes"]


def test_fit_info_is_disabled(ixman):
    for arm in ("frozen", "retrained"):
        t = IX.load(FC.PROFILE, ixman, arm, cpu_est=False)[0]
        with pytest.raises(IX.RefitForbidden):
            t.fit_info(np.arange(10), "avif_5")


def test_frozen_state_ignores_new_rows(fd, ixman, tmp_path):
    prof = perturbed_profile(tmp_path, "all", ixman, lambda s: True)
    t, _, _, _, st = IX.load(prof, ixman, "frozen")
    assert IX.state_hash(st) == IX.state_hash(fd["st"])      # every row changed, the controller state did not
    assert np.allclose(t.bits, 2 * fd["t"].bits)              # ... but the true outcomes are the new rows
    img = 0
    b, q = t.predict(img)
    assert b[t.probe_j] == t.bits[img, t.probe_j]            # the probe is the new image's own executed encode
    X = np.array([1.0, np.log(t.bits[img, t.probe_j])])
    assert np.allclose(np.delete(b, t.probe_j), np.delete(np.exp(X @ fd["t"].coef_bits), t.probe_j))


def test_retrained_state_uses_u_train_rows_only(ixman, tmp_path):
    base = IX.load(FC.PROFILE, ixman, "retrained", cpu_est=False)[4]
    other = perturbed_profile(tmp_path, "nontrain", ixman, lambda s: s != "U_train")
    assert IX.state_hash(IX.load(other, ixman, "retrained", cpu_est=False)[4]) == IX.state_hash(base)
    train = perturbed_profile(tmp_path, "train", ixman, lambda s: s == "U_train")
    assert IX.state_hash(IX.load(train, ixman, "retrained", cpu_est=False)[4]) != IX.state_hash(base)


def test_refit_path_reproduces_fd_v1_on_rl_train(fd, ixman):
    """the retrained code path, given fd_v1 rl_train as U_train and the fd_v1 CPU estimate, IS the fd_v1 state."""
    st = IX.load(FC.PROFILE, ixman, "retrained", cpu_est=FC.CPU_EST)[4]
    for k in IX.FROZEN_KEYS:
        assert np.allclose(st[k], fd["st"][k], rtol=0, atol=1e-12), k


def test_frozen_arm_has_no_train_pool(ixman):
    pools = IX.load(FC.PROFILE, ixman, "frozen")[1]
    assert "train" not in pools and "val" not in pools
    pools = IX.load(FC.PROFILE, ixman, "retrained", cpu_est=False)[1]
    assert np.array_equal(pools["train"], pools["U_train"])


class _Unbounded(M.Deadline):
    """deterministic budget condition: every planning step is allowed (no wall-clock dependence)."""

    def ok(self, stage, units):
        return True


@pytest.mark.parametrize("suffix", ["", "+mpcng", "+mpc"])
def test_behavioural_equivalence_deterministic_budget(ixman, monkeypatch, suffix):
    monkeypatch.setattr(M, "Deadline", _Unbounded)
    combo = f"{P2.sched()}|h=fixed_avif_20{suffix}"
    X2.DELIV_EXT = True
    out = {}
    for which in ("fd", "ix"):
        if which == "fd":
            FC.init()
            pool = "TD"
        else:
            IX.init(FC.PROFILE, ixman, "frozen")
            pool = "U_test"
        out[which] = [X2.episode("K5", "K5_la", combo, -1, s, dict(P2.LA, obs_la=False), None, pool)
                      for s in (20000, 20001)]
    for (m1, d1), (m2, d2) in zip(out["fd"], out["ix"]):
        assert m1["return"] == m2["return"]
        assert d1 == d2


def test_real_timing_path_runs(ixman):
    """the real 50 ms wall-clock Deadline path on the ix loader: completes, finite return, abort counter present."""
    IX.init(FC.PROFILE, ixman, "frozen")
    X2.DELIV_EXT = True
    m, d = X2.episode("K5", "K5_la", f"{P2.sched()}|h=fixed_avif_20+mpcng", -1, 20002,
                      dict(P2.LA, obs_la=False), None, "U_test")
    assert np.isfinite(m["return"]) and len(d) > 0
