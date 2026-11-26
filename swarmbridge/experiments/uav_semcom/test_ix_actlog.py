"""Training-action logging does not change training (review of e964ad41, item 3): short PPO and QR-DQN runs on the
fd_v1 training pool with and without --log-actions give bit-identical parameters; the histograms count every
executed mode action. CPU only, small budgets."""
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def sched():
    return json.loads((HERE.parents[2] / "runs" / "uav_semcom" / "ax9" / "sched.json").read_text())["frozen"]


def run(argv, cwd):
    p = subprocess.run([sys.executable] + argv, cwd=cwd, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]


def tensors_equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tensors_equal(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tensors_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, "shape"):
        import numpy as np
        return np.array_equal(a, b)
    return a == b


def test_ppo_logging_is_read_only(tmp_path):
    f = sched()
    extra = json.dumps(dict(la_mode="sched", la_interf="persistent", obs_la=True))
    base = ["a5_train.py", None, "--regime", "K5_la", "--rl-part", "mode", "--heur", "fixed_avif_20", "--seed", "0",
            "--steps", "2048", "--eval-every", "2048", "--n-env", "2", "--cfg-extra", extra, "--slot-move", f["move"],
            "--slot-ja", f["member"], "--fd"]
    for name, flag in (("plain", []), ("logged", ["--log-actions"])):
        argv = list(base)
        argv[1] = str(tmp_path / name)
        run(argv + flag, HERE)
    sd = {}
    for name in ("plain", "logged"):
        with zipfile.ZipFile(tmp_path / name / "ckpt_2048.zip") as z:
            sd[name] = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    assert tensors_equal(sd["plain"], sd["logged"])
    tj = json.loads((tmp_path / "logged" / "train.json").read_text())
    rec = tj["log"][-1]
    assert sum(rec["train_action_hist"]) == rec["experience"]["rl_steps_mode"] > 0
    assert "train_action_hist" not in json.loads((tmp_path / "plain" / "train.json").read_text())["log"][-1]
    a, b = (json.loads((tmp_path / n / "train.json").read_text())["log"][-1]["val_return"] for n in ("plain", "logged"))
    assert a == b


def test_offpolicy_logging_is_read_only(tmp_path):
    base = ["offpolicy_train.py", None, "--algo", "qrdqn", "--seed", "0", "--steps", "400", "--eval-every", "400",
            "--learning-starts", "200", "--batch", "32", "--buffer", "1000", "--fd"]
    for name, flag in (("plain", []), ("logged", ["--log-actions"])):
        argv = list(base)
        argv[1] = str(tmp_path / name)
        run(argv + flag, HERE)
    a = torch.load(tmp_path / "plain" / "ckpt_400.pt", map_location="cpu", weights_only=False)  # created by this test
    b = torch.load(tmp_path / "logged" / "ckpt_400.pt", map_location="cpu", weights_only=False)
    assert tensors_equal(a, b)
    rec = json.loads((tmp_path / "logged" / "train.json").read_text())["log"][-1]
    h = rec["train_action_hist"]
    assert sum(h["random"]) == 200 and sum(h["policy"]) == 200
