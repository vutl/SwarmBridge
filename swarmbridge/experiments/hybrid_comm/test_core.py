import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C


def test_geometry_roundtrip_and_known_cell():
    pose = (12, -4, np.pi / 2)
    local = np.array([[0.4, 1.2], [-102, -102]])
    world = C.local_to_world(local, pose)
    assert np.allclose(C.world_to_local(world, pose), local)
    ix, valid = C.cells(world, pose)
    assert np.array_equal(ix[0], [128, 129]) and valid.all()
    _, valid = C.cells([[102.4, 0], [-102.4, 0]])
    assert np.array_equal(valid, [False, True])


def test_warp_translation_rotation():
    import torch
    x = torch.zeros(1, 256, 256)
    x[0, 128, 130] = 1
    y = C.warp_grid(x, (0, 0, 0), (0.8, 0, 0))
    assert y[0, 127, 130] > .999
    z = C.warp_grid(x, (0, 0, 0), (0, 0, np.pi / 2))
    assert z[0, 130, 127] > .999
    assert C.warp_grid(x, (0, 0, 0), (0, 0, 0)) is x


def test_nms_stable_boundary_and_cap():
    def b(x, s): return dict(translation=[x, 0, 0], detection_score=s)
    a, c = b(0, .5), b(0.1, .5)
    assert C.nms_max([a, c])[0] is a
    assert len(C.nms_max([b(0, 1), b(2, .9)])) == 1
    assert len(C.nms_max([b(i * 3, 1) for i in range(10)], cap=3)) == 3


def test_split_and_cold_start():
    root = Path(__file__).resolve().parents[3] / "data/v1.0-mini"
    split = C.make_split(root)
    tr, va, de = (set(split["tokens"][g]) for g in ("train", "val", "dev"))
    assert (len(tr), len(va), len(de)) == (260, 80, 60)
    assert not (tr & va or tr & de or va & de)
    assert split == C.make_split(root)
    seq = next(iter(split["frames"].values()))
    for seed in range(20):
        ctx = C.context(split, seq[0], np.random.default_rng(seed))
        assert all(x["age"] == 0 and x["token"] == seq[0] for x in ctx["helpers"].values())
