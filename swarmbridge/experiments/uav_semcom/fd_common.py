"""fd_v1 data protocol for the simulator (Phase 2+): ModeTable, pools, recall tables, a5_eval initialisation.

ModeTable (env.ModeTable) from runs/uav_semcom/paperA_fd/profile/profile.jsonl, the 14 table modes (C.MODES):
  * payload / PSNR / LPIPS per image and mode: as measured (one pass; seeded generative modes);
  * encode / decode latency per mode: MEANS OVER rl_train ROWS ONLY (pre-registered correction; A5 averaged over all
    table rows);
  * MBT2018 encode latency (UAV CPU): the training-distribution-weighted stratified CPU estimate
    (profile/cpu_learned_stratified.meta.json, fd_cpu_stratified.py); never the GPU profile time, never the
    first-10-images file;
  * probe predictor ('predicted' mode info): env.ModeTable.fit_info on rl_train (avif_5 probe).
Pools (table row indices): train = rl_train, val = rl_val, TD, F, V, TD_extra, TD_all = TD + TD_extra.
Recall tables (pools F / TD / TD_extra / V only; YOLO was trained on the rl_* scenes): tp[row, mode], n_gt[row],
fp[row, mode] = n_det - tp, from the same reconstruction as the profile (fd_profile.py one-pass).
Artifacts are verified against the hashes fixed in the Phase 2 pre-registration (EXPECT)."""
import hashlib
import json
from collections import defaultdict

import numpy as np

import a5_common as C
import a5_eval as AE
import env as E
from fd_manifest import OUT

PROFILE = OUT / "profile" / "profile.jsonl"
CPU_META = OUT / "profile" / "cpu_learned_stratified.meta.json"
MANIFEST = OUT / "manifest_fd_v1.json"
EXPECT = {"manifest_fd_v1.json": "f0760601515f9663f2b7747bd247eddabc3ab8c997755c73ae8bf2f2339d4d50",
          "profile/profile.jsonl": "078e0e231dfeadaf9c9c84f8db2cb2a0d2c4166e4c3470cd35907d10f898781c",
          "profile/cpu_learned_stratified.meta.json": "ac063091b501516c7edb8f0cb1e596306012fc288d1fbb33b5005d92a2832341"}
RECALL_POOLS = ("F", "TD", "TD_extra", "V")


def sha_text(p):
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify():
    for f, h in EXPECT.items():
        assert sha_text(OUT / f) == h, f"{f} differs from the pre-registered artifact"


def load():
    verify()
    man = json.loads(MANIFEST.read_text())["images"]
    split = {e["name"]: e["split"] for e in man}
    t = E.ModeTable([str(PROFILE)], C.MODES, None)
    assert len(t.images) == len(man), (len(t.images), len(man))
    rows = defaultdict(dict)
    for l in open(PROFILE):
        r = json.loads(l)
        rows[r["image"]][r["mode"]] = r
    tr = [n for n in t.images if split[n] == "rl_train"]
    t.enc_s = np.array([np.mean([rows[n][m]["enc_s"] for n in tr]) for m in C.MODES])
    t.dec_s = np.array([np.mean([rows[n][m]["dec_s"] for n in tr]) for m in C.MODES])
    est = json.loads(CPU_META.read_text())["estimates"]
    for j, m in enumerate(C.MODES):
        if m in est:
            t.enc_s[j] = est[m]["weighted_mean"]
    idx = {n: i for i, n in enumerate(t.images)}
    pools = defaultdict(list)
    for e in man:
        pools[e["split"]].append(idx[e["name"]])
    pools = {k: np.array(sorted(v)) for k, v in pools.items()}
    pools["train"], pools["val"] = pools["rl_train"], pools["rl_val"]
    pools["TD_all"] = np.array(sorted(list(pools["TD"]) + list(pools["TD_extra"])))
    t.fit_info(pools["train"], "avif_5")
    I, Mn = len(t.images), len(C.MODES)
    rec = dict(tp=np.full((I, Mn), np.nan), fp=np.full((I, Mn), np.nan), ngt=np.full(I, np.nan))
    for n, i in idx.items():
        if split[n] in RECALL_POOLS:
            rec["ngt"][i] = rows[n]["original"]["n_gt"]
            for j, m in enumerate(C.MODES):
                rec["tp"][i, j] = rows[n][m]["tp"]
                rec["fp"][i, j] = rows[n][m]["n_det"] - rows[n][m]["tp"]
    man_idx = np.array([{e["name"]: k for k, e in enumerate(man)}[n] for n in t.images])
    return t, pools, rec, man_idx


def init():
    """a5_eval-compatible initialisation (worker initialiser): AE.G with the fd_v1 table and pools."""
    import torch
    torch.set_num_threads(1)
    t, pools, rec, man_idx = load()
    AE.G["table"], AE.G["pools"], AE.G["man_idx"], AE.G["models"] = t, pools, man_idx, {}
    AE.G["fd_recall"] = rec
    return t, pools, rec
