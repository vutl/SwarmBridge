"""paperA_ix data protocol (independent evaluation, pre-registration paperA-ix-prereg): image tables and the
controller-side state for the blocks X1 (VisDrone test-challenge), X2-frozen and X2-retrained (UAVDT).

Two kinds of quantities live in an env.ModeTable:
  * TRUE OUTCOMES per image x mode (bits, psnr, lpips): what the simulator delivers. They always come from the
    dataset's own profile rows.
  * CONTROLLER-SIDE STATE (FROZEN_KEYS): per-mode encode / decode latency, per-mode mean payload / PSNR (MPC future
    arrivals, 'mean' info mode) and the avif_5 probe predictor coefficients. These are FITTED quantities.
      frozen    : copied bit-for-bit from the fd_v1 table (fd_common.load, fitted on fd_v1 rl_train); never recomputed.
      retrained : refitted on U_train rows ONLY (latency means, CPU MBT2018 estimate from a U_train sample file,
                  fit_info on U_train).
    After the state is set, ModeTable.fit_info is disabled on the instance (any call raises), so no later code path can
    refit on test rows. env.ModeTable.__init__ averages enc / dec over ALL rows it reads; those values are always
    overwritten here (tested).
The probe itself (avif_5 payload of the CURRENT image) is an executed encode on the UAV, so predict(img) reads the new
image's own probe payload, exactly as in fd_v1.
Pools: X1 / U_train / U_val / U_test (manifest 'split'). Aliases 'train' / 'val' exist ONLY for the retrained arm
(U_train / U_val); the frozen arms have no train / val pool, so nothing can be tuned on the new data by accident.
Recall tables: U_test only (vehicle recall variants, fidelity fields written by the U_test fidelity pass)."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import a5_common as C
import a5_eval as AE
import env as E
from fd_manifest import REPO

OUT = REPO / "runs" / "uav_semcom" / "paperA_ix"
FROZEN_KEYS = ("enc_s", "dec_s", "mean_bits", "mean_psnr", "coef_bits", "coef_psnr")
# sha256 of the fd_v1 controller-side state (state_hash of fd_common.load()); pinned in the pre-registration
FROZEN_EXPECT = "39018855000424ebb39fb71c5702a626ab268132d0ca22b4b5b8e30843c6d3bf"
RECALL_VARIANTS = ("veh", "cls_vancar", "cls_car")     # (a) primary, (b), (c): prereg §9 / final §3.3


class RefitForbidden(RuntimeError):
    pass


def _forbid(*_a, **_k):
    raise RefitForbidden("controller-side state is locked; fit_info must not be called on a paperA_ix table")


def state_of(t):
    st = {k: np.array(getattr(t, k), dtype=np.float64, copy=True) for k in FROZEN_KEYS}
    st["probe_j"] = int(t.probe_j)
    st["modes"] = list(t.modes)
    return st


def state_hash(st):
    h = hashlib.sha256()
    h.update(json.dumps(st["modes"]).encode())
    h.update(str(st["probe_j"]).encode())
    for k in FROZEN_KEYS:
        a = np.ascontiguousarray(st[k], dtype=np.float64)
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fd_v1_state():
    import fd_common as FC
    t, _, _, _ = FC.load()
    st = state_of(t)
    if FROZEN_EXPECT is not None:
        assert state_hash(st) == FROZEN_EXPECT, "fd_v1 controller-side state differs from the pinned hash"
    return st


def apply_state(t, st):
    assert list(t.modes) == st["modes"], (t.modes, st["modes"])
    for k in FROZEN_KEYS:
        setattr(t, k, np.array(st[k], dtype=np.float64, copy=True))
    t.probe_j = st["probe_j"]
    t._pred_cache = {}
    t.fitted = True
    t.fit_info = _forbid


def build_table(profile, manifest):
    """ModeTable of the 14 modes over the manifest images (true outcomes), raw profile rows, manifest entries."""
    man = json.loads(Path(manifest).read_text())["images"]
    t = E.ModeTable([str(profile)], C.MODES, None)
    names = {e["name"] for e in man}
    assert set(t.images) == names, (len(t.images), len(names))
    rows = defaultdict(dict)
    for line in open(profile):
        r = json.loads(line)
        rows[r["image"]][r["mode"]] = r
    return t, rows, man


def refit_state(t, rows, man, train_split, cpu_est=None):
    """X2-retrained: controller-side state from the rows of `train_split` only (fd_common.load logic)."""
    split = {e["name"]: e["split"] for e in man}
    tr_names = [n for n in t.images if split[n] == train_split]
    tr_idx = np.array([i for i, n in enumerate(t.images) if split[n] == train_split])
    assert len(tr_idx) > 0
    t.enc_s = np.array([np.mean([rows[n][m]["enc_s"] for n in tr_names]) for m in C.MODES])
    t.dec_s = np.array([np.mean([rows[n][m]["dec_s"] for n in tr_names]) for m in C.MODES])
    if cpu_est is not None:
        est = json.loads(Path(cpu_est).read_text())["estimates"]
        for j, m in enumerate(C.MODES):
            if m in est:
                t.enc_s[j] = est[m][est[m]["primary"]]
    E.ModeTable.fit_info(t, tr_idx, "avif_5")
    return state_of(t)


def pools_of(t, man, arm):
    idx = {n: i for i, n in enumerate(t.images)}
    pools = defaultdict(list)
    for e in man:
        pools[e["split"]].append(idx[e["name"]])
    pools = {k: np.array(sorted(v)) for k, v in pools.items()}
    if arm == "retrained":
        pools["train"], pools["val"] = pools["U_train"], pools["U_val"]
    man_idx = np.array([{e["name"]: k for k, e in enumerate(man)}[n] for n in t.images])
    return pools, man_idx


def recall_table(t, rows, man, pools=("U_test",)):
    split = {e["name"]: e["split"] for e in man}
    I, Mn = len(t.images), len(C.MODES)
    rec = {v: dict(tp=np.full((I, Mn), np.nan), ngt=np.full(I, np.nan)) for v in RECALL_VARIANTS}
    for i, n in enumerate(t.images):
        if split[n] not in pools or "tp_veh" not in rows[n].get("original", {}):
            continue
        for v in RECALL_VARIANTS:
            rec[v]["ngt"][i] = rows[n]["original"][f"n_gt_{v}"]
            for j, m in enumerate(C.MODES):
                rec[v]["tp"][i, j] = rows[n][m][f"tp_{v}"]
    return rec


def load(profile, manifest, arm, cpu_est=None):
    """arm: 'frozen' (fd_v1 state) | 'retrained' (refit on U_train). Returns t, pools, rec, man_idx, state."""
    assert arm in ("frozen", "retrained")
    t, rows, man = build_table(profile, manifest)
    st = fd_v1_state() if arm == "frozen" else refit_state(t, rows, man, "U_train", cpu_est)
    apply_state(t, st)
    pools, man_idx = pools_of(t, man, arm)
    return t, pools, recall_table(t, rows, man), man_idx, st


def init(profile, manifest, arm, cpu_est=None):
    """a5_eval-compatible worker initialiser (as fd_common.init)."""
    import torch
    torch.set_num_threads(1)
    t, pools, rec, man_idx, st = load(profile, manifest, arm, cpu_est)
    AE.G["table"], AE.G["pools"], AE.G["man_idx"], AE.G["models"] = t, pools, man_idx, {}
    AE.G["ix_recall"], AE.G["ix_state"] = rec, st
    return t, pools, rec
