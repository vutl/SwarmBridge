"""A6.0-L1 checks. Need the A5 data and the L1 task-fidelity files (skipped otherwise).
  * the task env changes ONLY the reward: same policy -> same decisions, and its parallel A5 (PSNR) return equals the
    return of the A5 env exactly;
  * the task value is normalised as pre-registered (train mean of the original frame = 1) and the probe predictor is
    a function of the probe payload only;
  * the task metrics of an episode equal a recomputation from its deliveries;
  * policy-level AP50 reproduces COCOeval's AP50 when every pool image is delivered once in one mode;
  * the matching / ignore logic of the fidelity scorer on hand-made boxes.
"""
import json

import numpy as np
import pytest

import a5_common as C
import a5_fidelity as F
import a55_heur as H
import a6l1_task as TK
import env_ev as V
import env_survey as SV
import mode_heur as MH

HAVE = C.MANIFEST.exists() and (TK.FID / "per_image_val.jsonl").exists() and (TK.FID / "coco_val.json").exists()
T = {}


def table():
    if not T:
        T["t"], T["p"] = C.load_table(False)
    return T["t"], T["p"]


def test_match_and_ignore():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    gc = np.array([0, 1])
    det = np.array([[0, 0, 10, 10], [20, 20, 30, 31], [50, 50, 60, 60]], float)
    dc = np.array([0, 0, 1])                                  # 2nd: right box, wrong class
    mg, md = F.match(det, dc, gt, gc)
    assert mg.tolist() == [True, False] and md.tolist() == [True, False, False]
    ign = np.array([[45, 45, 70, 70]], float)
    assert F.in_ignore(det, ign).tolist() == [False, False, True]


@pytest.mark.skipif(not HAVE, reason="L1 fidelity files not available")
def test_task_value_normalisation():
    tab, pools = table()
    tt = TK.task_table(tab)
    tr = pools["train"]
    assert tt.covered(np.concatenate([pools["train"], pools["val"], pools["test"]]))
    assert abs(np.mean((tt.tp0[tr] - TK.BETA * tt.fp0[tr]) / tt.vbar) - 1.0) < 1e-12
    i = int(pools["test"][0])
    x = np.log(tab.bits[i, tab.probe_j])
    assert np.allclose(tt.vhat(i), tt.coef[0] + tt.coef[1] * x)


@pytest.mark.skipif(not HAVE, reason="L1 fidelity files not available")
@pytest.mark.parametrize("regime,name", [("ref", "pacedB1.5+budget1.25+reuse"), ("K5", "pacedB2+budget1+reuse"),
                                         ("phy_mcs_blind", "pacedB2+budget1")])
def test_task_env_changes_only_the_reward(regime, name):
    tab, pools = table()
    for seed in (0, 1):
        a = SV.UavSurveyEnv(tab, pools["train"], C.regime_cfg(regime))
        b = TK.TaskSurveyEnv(tab, pools["train"], C.regime_cfg(regime))
        ma = V.run_episode_ev(a, *MH.from_name(name, C.MODES), seed)
        mb = V.run_episode_ev(b, *MH.from_name(name, C.MODES), seed)
        assert a.decisions == b.decisions
        assert [(t.img, t.mode, t.ok, t.delay) for t in a.finished] == [(t.img, t.mode, t.ok, t.delay) for t in b.finished]
        assert abs(ma["return"] - (mb["return"] + mb["psnr_minus_task"])) < 1e-6
        for k in ("ok_rate", "mean_psnr", "energy_kj", "mission_s"):
            assert ma[k] == mb[k]


@pytest.mark.skipif(not HAVE, reason="L1 fidelity files not available")
def test_task_metrics_recomputed():
    tab, pools = table()
    core = TK.TaskSurveyEnv(tab, pools["train"], C.regime_cfg("K5"))
    m = V.run_episode_ev(core, *H.from_name("tv:pacedB2+budget1+reuse", C.MODES), 2)
    tt = core.tt
    ok = {(t.uav, t.wp): t.mode for t in core.finished if t.ok}
    tp = ngt = 0.0
    K, W = core.task_img.shape[:2]
    for k in range(K):
        for w in range(W):
            i = int(core.task_img[k, w, 0])
            ngt += tt.ngt[i]
            if (k, w) in ok:
                tp += tt.tp[i, ok[(k, w)]]
    assert abs(m["recall"] - tp / ngt) < 1e-12
    assert m["n_ok"] == len(ok) and m["n_planned"] == K * W


@pytest.mark.skipif(not HAVE or not __import__("pathlib").Path("/home/uservutl/work/a6l1/dets_val.npz").exists(),
                    reason="detections not available")
def test_policy_ap_reproduces_cocoeval():
    import a6l1_ap as AP
    b = AP.build("val")
    res = json.loads((TK.FID / "coco_val.json").read_text())["results"]
    n = len(b["images"])
    for mode in ("original", "avif_50", "avif_5", "mbt2018-mean_q3", "thumb128_up", "cap_t2i"):
        ref = res["gen_subset" if mode.startswith("cap_") else "all"][mode]["AP50"]   # gen subset = all frames here
        ap = AP.ap50([("val", i, mode) for i in range(n)], {"val": b})
        assert abs(ap - ref) < 1e-9, (mode, ap, ref)
    # undelivered frames keep their ground truth: delivering nothing gives 0
    assert AP.ap50([("val", i, None) for i in range(n)], {"val": b}) == 0.0
