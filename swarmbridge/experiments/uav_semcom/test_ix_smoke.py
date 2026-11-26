"""End-to-end smoke of the stage drivers on the relabelled fd_v1 sandbox (test_ix_lock.sandbox), tiny budgets:
ix_stage tune (2 rules x 2 seeds x 2 settings) -> L2 lock -> ix_eval blocks x1_ref and x2r_rl (2 seeds; the fd_v1
retrained checkpoints stand in for the UAVDT ones) -> ix_analysis on the outputs. Checks the plumbing only."""
import json

import pytest

import ix_analysis as AN
import ix_common as IX
import ix_eval as EV
import ix_stage as ST
from fd_next_summary import Pool
from test_ix_lock import sandbox  # noqa: F401  (fixture)


def test_stage_drivers_end_to_end(sandbox, monkeypatch):  # noqa: F811
    import ax9_run as A9
    IX.fit_retrained()
    # --- tune (tiny)
    monkeypatch.setattr(ST.C, "TRAIN_SEEDS", [0, 1])
    monkeypatch.setattr(A9, "H_FAMILY", ["fixed_avif_20", "fixed_avif_5"])
    monkeypatch.setattr(ST, "CFG", {k: v for k, v in ST.CFG.items() if k in ("N5x2", "B2")})
    ST.tune(2)
    t = json.loads(IX.TUNE_FILE.read_text())
    assert set(t["H"]) == {"N5x2", "B2"} and all(len(v) == 2 for v in t["train_return"].values())
    t["H"] = {c: "fixed_avif_20" for c in EV.CFG}                    # full mapping for the lock / grid
    IX.TUNE_FILE.write_text(json.dumps(t))
    # --- stand-in checkpoints (fd_v1 retrained) and selection
    fd = json.loads((IX.REPO / "runs" / "uav_semcom" / "paperA_fd" / "phase3" / "ckpts.json").read_text())
    ck = {"ppoix": fd["ppofd"], "qrix": fd["qrfd"], "dsix": fd["dsfd"]}
    IX.CKPTS.write_text(json.dumps(ck))
    IX.SELECTION.write_text(json.dumps(dict(selection={k: {str(s): dict(ckpt=p, ckpt_sha256=IX.sha_file(p))
                                                           for s, p in enumerate(v)} for k, v in ck.items()})))
    with pytest.raises(IX.LockError):
        EV.run_block("x1_ref", 2)                                     # no lock yet
    ST.lock()
    # --- stage B blocks (2 seeds each)
    monkeypatch.setattr(EV, "paths", lambda b: (sandbox / "eval" / f"{b}.json", sandbox / "eval" / f"{b}.deliv.json.gz"))
    blocks = dict(EV.BLOCKS)
    blocks["x1_ref"] = blocks["x1_ref"][:3] + (range(30000, 30002),)
    blocks["x2r_rl"] = blocks["x2r_rl"][:3] + (range(32000, 32002),)
    blocks["x2r_ref"] = blocks["x2r_ref"][:3] + (range(32000, 32002),)
    monkeypatch.setattr(EV, "BLOCKS", blocks)
    for b in ("x1_ref", "x2r_ref", "x2r_rl"):
        EV.run_block(b, 2)
        assert EV.complete(b)
    # --- analysis plumbing
    t_, pools, rec, man_idx, st, man = IX.load_eval("x1", "frozen")
    ctx = (rec, {int(mi): r for r, mi in enumerate(man_idx)}, {k: e["seq"] for k, e in enumerate(man)}, pools)
    P = Pool(*EV.paths("x1_ref"), "X1", *ctx)
    o = AN.analyse(P, "ref", EV.h_star("frozen", "N5x2"), None)
    assert set(o["contrasts"]) == {"H1", "H3b", "H2", "H2b", "H3"} and o["recall"].startswith("unavailable")
    t_, pools, rec, man_idx, st, man = IX.load_eval("uavdt", "retrained")
    ctx = (None, {int(mi): r for r, mi in enumerate(man_idx)}, {k: e["seq"] for k, e in enumerate(man)}, pools)
    P = Pool(*EV.paths("x2r_rl"), "U_test", *ctx)
    Pr = Pool(*EV.paths("x2r_ref"), "U_test", *ctx)
    o = AN.analyse(P, "rl", EV.LEARNED, None, P_ref=Pr, H_ref=EV.h_star("retrained", "N5x2"))
    assert {k for k, v in o["contrasts"].items() if "holm" in v} == {f"R{i}_{x}" for i in (1, 2) for x in EV.LEARNED}
    assert set(o["R5"]) == set(EV.LEARNED)
