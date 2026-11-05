import sys
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
import receiver as R
import tilecodec as TC


def test_zero_extended_initialization_preserves_ego():
    ds = torch.nn.Sequential(torch.nn.Conv2d(80, 4, 3, padding=1, bias=False))
    ego = torch.randn(1, 80, 8, 8)
    expected = ds(ego).detach()
    R.expand_input(ds)
    full = torch.randn(1, R.CHANNELS, 8, 8)
    full[:, :80] = ego
    assert torch.allclose(ds(full), expected, atol=1e-6)
    assert torch.count_nonzero(ds[0].weight[:, 80:]) == 0


def test_only_decoded_messages_and_matched_box_inputs(tmp_path):
    C.save_json(tmp_path / "geometry.json", dict(poses={"now": [0, 0, 0], "old": [0, 0, 0]}))
    (tmp_path / "boxes").mkdir(); (tmp_path / "tiles").mkdir()
    boxes = dict(xy=np.array([[.4, .4]]), z=np.array([1]), size=np.array([[2,4,1]]),
                 yaw=np.array([0]), vel=np.array([[1.,0]]), score=np.array([.9]))
    packet, _ = R.BC.encode(1, 0, [0,0,20], boxes)
    (tmp_path / "boxes/old_1.bin").write_bytes(packet)
    tl = np.full((64, 80, 8, 8), 2.0, np.float32)
    msg = TC.encode(tl, np.arange(64), sender=1)
    (tmp_path / "tiles/old_1_obj.bin").write_bytes(msg)
    data = R.ReceiverData(tmp_path, device="cpu")
    data.ego = lambda tok: (torch.zeros(80,256,256), [])
    ctx = dict(token="now", helpers={"1": dict(token="old", age=1)}, k=16, rank="obj")
    box, _, bit_b = data.build(ctx, False)
    joint, _, bit_j = data.build(ctx, True)
    assert torch.equal(box[:80], joint[:80])
    assert torch.equal(box[160:168], joint[160:168])
    assert torch.equal(box[169], joint[169])
    assert torch.equal(box[171], joint[171])
    assert torch.count_nonzero(box[80:160]) == 0
    assert torch.count_nonzero(box[168]) == 0
    assert joint[168].sum() == 16 * 64
    assert torch.all(joint[80:160, :8, :8] == 2)
    assert bit_j - bit_b == 8 * TC.message_bytes(16)
    ix, _ = C.cells([[.6, .4]])  # sender velocity propagated by 0.2 seconds
    assert joint[169, ix[0,0], ix[0,1]] == 1


def test_empty_helper_cold_start(tmp_path):
    C.save_json(tmp_path / "geometry.json", dict(poses={"now": [0, 0, 0]}))
    data = R.ReceiverData(tmp_path, device="cpu")
    data.ego = lambda tok: (torch.ones(80,256,256), [])
    x, _, bits = data.build(dict(token="now", helpers={}, k=64, rank="obj"))
    assert torch.all(x[:80] == 1) and torch.count_nonzero(x[80:]) == 0 and bits == 0
