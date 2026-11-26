"""GPU check of the paperA_ix profiling pipeline (run under the GPU lock): IxPipe.recon reproduces the stored fd_v1
profile rows of fd_v1 rl_train images (same codecs, generator seeds, metrics), and the full stage path (14 modes +
UAVDT recall fields) runs on one UAVDT TRAINING image (stage-A data; no X1 / U_test image is touched)."""
import json

import numpy as np
import pytest
import torch

import a5_common as C
import fd_common as FC
import fd_profile as FP
import ix_common as IX
import ix_profile as XP

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU (run under fd_gpulock_win.py)")


@pytest.fixture(scope="module")
def pipe():
    return XP.IxPipe(recall=True)


def test_recon_reproduces_fd_v1_rows(pipe):
    man = [e for e in json.loads(FC.MANIFEST.read_text())["images"] if e["split"] == "rl_train"][:2]
    stored = {}
    for line in open(FC.PROFILE):
        r = json.loads(line)
        if r["image"] in {e["name"] for e in man}:
            stored[(r["image"], r["mode"])] = r
    pipe.recon(FP.img_path(man[0]), man[0]["gen_seed"])            # warm-up, as the drivers do
    for e in man:
        _, _, rows = pipe.recon(FP.img_path(e), e["gen_seed"])
        for r in rows:
            s = stored[(e["name"], r["mode"])]
            if r["mode"] in FP.CODEC_MODES:
                assert r["bytes"] == s["bytes"], (e["name"], r["mode"])
            assert abs(r["psnr"] - s["psnr"]) < 0.05, (e["name"], r["mode"], r["psnr"], s["psnr"])
            assert r.get("caption") == s.get("caption")


def test_stage_path_on_uavdt_training_image(pipe):
    e = next(e for e in IX.read_manifest() if e["split"] == "U_train")
    gt = XP.read_uavdt_gt(e["seq"], e["frame"])
    rows = pipe.image_ix(XP.img_path(e), e["gen_seed"], gt)
    assert [r["mode"] for r in rows] == list(C.MODES) + ["original"]
    for r in rows:
        for v in IX.RECALL_VARIANTS:
            assert 0 <= r[f"tp_{v}"] <= r[f"n_gt_{v}"]
    assert rows[-1]["n_gt_veh"] == len(gt[0]) - int(XP.AF.in_ignore(gt[0], gt[2]).sum())
    assert all(np.isfinite(r["psnr"]) and r["bytes"] > 0 for r in rows[:-1])
