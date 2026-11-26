"""ix_analysis on STORED fd_v1 results: the registered contrasts reproduce the fd_v1 summaries, the N family and the
principal-verdict rule behave as registered. No new data is involved."""
import gzip
import json

import numpy as np
import pytest

import fd_common as FD
import ix_analysis as AN
from fd_next_summary import Pool


@pytest.fixture(scope="module")
def ctx():
    t, pools, rec, man_idx = FD.load()
    man = json.loads(FD.MANIFEST.read_text())["images"]
    return rec, {int(mi): r for r, mi in enumerate(man_idx)}, {k: e["seq"] for k, e in enumerate(man)}, pools


def H():
    return json.loads((FD.OUT / "phase2" / "tune.json").read_text())["H"]


def test_ablation_reproduces_fd_v1_td(ctx):
    P = Pool(FD.OUT / "ablation" / "test_TD.json", FD.OUT / "ablation" / "test_TD.deliv.json.gz", "TD", *ctx)
    rec = {"veh": dict(ctx[0], tp_orig=np.full(len(ctx[0]["ngt"]), np.nan))}
    o = AN.analyse(P, "abl", H(), rec)
    ref = json.loads((FD.OUT / "ablation" / "summary_ablation_test.json").read_text())["TD"]["contrasts"]
    for k in ("A1", "A2", "A3", "A4", "A4b", "A5", "A6", "A7", "A8", "A9"):
        assert np.allclose(o["contrasts"][k]["ci"], ref[k]["ci"], atol=1e-12), k
    for k in ("A1", "A2", "A4", "A4b"):
        assert o["contrasts"][k]["holm"]["rejected"] == ref[k]["holm"]["rejected"], k
        assert np.allclose(o["contrasts"][k]["recall"]["veh"]["cluster"], ref[k]["recall"]["cluster"]), k
    # new N family: N2 = A8 contrast, N3 = A7 contrast, N1 = nominal - H*
    assert np.allclose(o["contrasts"]["N2"]["ci"], ref["A8"]["ci"]) and np.allclose(o["contrasts"]["N3"]["ci"], ref["A7"]["ci"])
    d = P.ret("N5x2", f"h={H()}+mpcnc") - P.ret("N5x2", f"h={H()}")
    assert np.isclose(o["contrasts"]["N1"]["ci"][0], d.mean())
    assert set(k for k, v in o["contrasts"].items() if "holm" in v) == {"A1", "A2", "A4", "A4b", "N1", "N2", "N3"}
    assert o["principal"] in ("supported", "partially supported", "not supported")


def test_reference_reproduces_fd_v1_phase2_td(ctx):
    P = Pool(FD.OUT / "phase2" / "test_TD.json", FD.OUT / "phase2" / "test_TD.deliv.json.gz", "TD", *ctx)
    o = AN.analyse(P, "ref", H(), None)
    ref = json.loads((FD.OUT / "phase2" / "summary_test.json").read_text())["TD"]
    for k in ("H1", "H2", "H2b", "H3", "H3b"):
        assert np.allclose(o["contrasts"][k]["ci"], ref["return"][k], atol=1e-12), k
    for k in ("H1", "H3b"):
        assert o["contrasts"][k]["holm"]["rejected"] == ref["holm"][k]["rejected"]
    assert o["recall"].startswith("unavailable")


def _c(n2, a2):
    """n2 / a2: (rejected, positive)."""
    mk = lambda r, s: dict(holm=dict(rejected=r), ci=[1.0 if s else -1.0, 0, 0])
    return {"N2": mk(*n2), "A2": mk(*a2)}


@pytest.mark.parametrize("n2,a2,want", [
    ((True, True), (True, True), "supported"),
    ((True, True), (False, True), "partially supported"),
    ((False, True), (True, True), "partially supported"),
    ((False, True), (False, False), "not supported"),
    ((True, False), (True, True), "not supported"),           # significant negative takes precedence
    ((True, True), (True, False), "not supported"),
    ((False, True), (True, False), "not supported"),
])
def test_principal_verdict_rule(n2, a2, want):
    assert AN.verdict(_c(n2, a2)) == want


def test_utilisation_matches_counts(ctx):
    dl = json.load(gzip.open(FD.OUT / "phase2" / "test_TD.deliv.json.gz", "rt"))
    u = AN.utilisation(dl)
    k = next(k for k in dl if k.endswith("h=fixed_avif_20"))
    assert u[k]["modes"]["avif_20"]["share"] == 1.0 and u[k]["no_mode"] == 0
    tot = sum(len(ep) for ep in dl[k])
    assert u[k]["finished_with_mode"] == tot
