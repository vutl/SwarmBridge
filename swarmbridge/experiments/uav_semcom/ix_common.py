"""paperA_ix data protocol (independent evaluation, reports/phases/2026-11-26_independent-eval/paperA-ix-prereg.md):
image tables, controller-side state and the L2 lock for the blocks X1 (VisDrone test-challenge), X2-frozen and
X2-retrained (UAVDT).

Two kinds of quantities live in an env.ModeTable:
  * TRUE OUTCOMES per image x mode (bits, psnr, lpips): what the simulator delivers; always the dataset's own profile.
  * CONTROLLER-SIDE STATE (FROZEN_KEYS): per-mode encode / decode latency, per-mode mean payload / PSNR (MPC future
    arrivals, 'mean' info mode) and the avif_5 probe-predictor coefficients. These are FITTED quantities:
      frozen    : copied bit-for-bit from the fd_v1 table (fd_common.load, fitted on fd_v1 rl_train); pinned hash.
      retrained : fitted ONCE on U_train rows (`fit_retrained`, stage A, with the REQUIRED registered CPU MBT2018
                  estimate) and SERIALISED to state_retrained.json; every later user (H* tuning, RL training, test
                  evaluation) LOADS that file and verifies its hash. Test workers never fit anything.
    After a state is applied, ModeTable.fit_info is disabled on the instance (any call raises). env.ModeTable.__init__
    averages enc / dec over ALL rows it reads; those values are always overwritten here (tested).
The probe (avif_5 payload of the CURRENT image) is an executed encode on the UAV: predict(img) reads the image's own
probe payload, exactly as in fd_v1.

Stages and tables (profiles live under WORK, outside git: they hold captions and per-image values):
  stage A  : U_train + U_val rows, retrained state loaded from state_retrained.json (tuning, RL training / selection);
  'uavdt'  : U_test rows only (X2-frozen and X2-retrained evaluation; recall variants from the U_test fidelity pass);
  'x1'     : X1 rows only (X1 evaluation; no recall).
Test-stage loading (`load_eval`) requires a verified L2 lock (`verify_lock`): manifest, stage-A profile, CPU estimate,
retrained state, frozen state, H* selections, RL checkpoints and the evaluation code are all hash-checked.
Pool aliases 'train' / 'val' exist only in stage A; evaluation tables contain test rows only."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import a5_common as C
import a5_eval as AE
import env as E
from fd_manifest import REPO

HERE = Path(__file__).resolve().parent
OUT = REPO / "runs" / "uav_semcom" / "paperA_ix"
WORK = Path("/home/uservutl/work/paperA_ix")
MANIFEST = OUT / "manifest_ix_v1.json"
MANIFEST_SHA = "734ea49b8d71e1caed6a2fc7f2198dcfb3685bfb2ea05654fb0971bc1f28bd92"
PROFILE = {"A": WORK / "profile" / "profile_A.jsonl", "uavdt": WORK / "profile" / "profile_B_uavdt.jsonl",
           "x1": WORK / "profile" / "profile_B_x1.jsonl"}
SPLITS = {"A": ("U_train", "U_val"), "uavdt": ("U_test",), "x1": ("X1",)}
CPU_EST = OUT / "cpu" / "cpu_learned_stratified_estimators.json"
STATE_FILE = OUT / "state_retrained.json"
FROZEN_FILE = OUT / "state_frozen_fd_v1.json"     # fd_v1 state serialised once (write_frozen_file); pinned
TUNE_FILE = OUT / "tune.json"
CKPTS = OUT / "ckpts.json"
SELECTION = OUT / "selection.json"
LOCK = OUT / "L2_lock.json"
FROZEN_KEYS = ("enc_s", "dec_s", "mean_bits", "mean_psnr", "coef_bits", "coef_psnr")
# sha256 of the fd_v1 controller-side state (state_hash of fd_common.load()); pinned in the pre-registration
FROZEN_EXPECT = "39018855000424ebb39fb71c5702a626ab268132d0ca22b4b5b8e30843c6d3bf"
RECALL_VARIANTS = ("veh", "cls_vancar", "cls_car")     # (a) primary, (b), (c): prereg §3
# code whose behaviour the test blocks depend on; hashed into L2 (a change after L2 is a protocol deviation)
LOCKED_CODE = ("ix_common.py", "ix_eval.py", "ix_profile.py", "ax2_eval.py", "a9_mpc.py", "env.py", "env_la.py",
               "env_survey.py", "env_ev.py", "ax2_env.py", "ja_sched.py", "mode_heur.py", "policies.py",
               "offpolicy_rl.py", "timeppo.py", "a5_fidelity.py", "a5_common.py", "a5_eval.py", "imcodecs.py",
               "metrics.py", "fd_profile.py", "profile_a11.py")


class RefitForbidden(RuntimeError):
    pass


class LockError(RuntimeError):
    pass


def _forbid(*_a, **_k):
    raise RefitForbidden("controller-side state is locked; fit_info must not be called on a paperA_ix table")


def sha_text(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- controller-side state
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


def save_state(st, path, meta=None):
    """JSON with exact float64 round trip (repr); the state hash is stored and re-checked on load."""
    d = dict(meta or {}, modes=st["modes"], probe_j=st["probe_j"], state_hash=state_hash(st),
             arrays={k: np.asarray(st[k]).tolist() for k in FROZEN_KEYS})
    Path(path).write_text(json.dumps(d, indent=1))
    return d["state_hash"]


def load_state(path, expect=None):
    d = json.loads(Path(path).read_text())
    st = {k: np.array(d["arrays"][k], dtype=np.float64) for k in FROZEN_KEYS}
    st["probe_j"], st["modes"] = int(d["probe_j"]), list(d["modes"])
    h = state_hash(st)
    if h != d["state_hash"] or (expect is not None and h != expect):
        raise LockError(f"state file {path}: hash {h} != stored {d['state_hash']} / expected {expect}")
    return st


def fd_v1_state():
    import fd_common as FC
    t, _, _, _ = FC.load()
    st = state_of(t)
    if state_hash(st) != FROZEN_EXPECT:
        raise LockError("fd_v1 controller-side state differs from the pinned hash")
    return st


def write_frozen_file():
    """Serialise the fd_v1 controller-side state once (from fd_common.load, fitted on fd_v1 rl_train)."""
    if FROZEN_FILE.exists():
        raise LockError(f"{FROZEN_FILE} exists")
    save_state(fd_v1_state(), FROZEN_FILE, dict(fitted_on="fd_v1 rl_train (fd_common.load)"))


def frozen_state():
    """Evaluation-time frozen state: the serialised file, hash-checked against the pin; nothing is fitted."""
    return load_state(FROZEN_FILE, FROZEN_EXPECT)


def apply_state(t, st):
    assert list(t.modes) == st["modes"], (t.modes, st["modes"])
    for k in FROZEN_KEYS:
        setattr(t, k, np.array(st[k], dtype=np.float64, copy=True))
    t.probe_j = st["probe_j"]
    t._pred_cache = {}
    t.fitted = True
    t.fit_info = _forbid


# ---------------------------------------------------------------- tables
def read_manifest(manifest=None, check=True):
    manifest = manifest or MANIFEST
    if check and Path(manifest) == MANIFEST and sha_text(manifest) != MANIFEST_SHA:
        raise LockError("manifest_ix_v1.json differs from the registered sha256")
    return json.loads(Path(manifest).read_text())["images"]


def build_table(profiles, manifest, splits=None):
    """ModeTable of the 14 modes over the manifest images of `splits` (true outcomes), raw profile rows, the full
    manifest and the selected entries. The profiles must contain exactly the selected images."""
    man = read_manifest(manifest)
    sel = [e for e in man if splits is None or e["split"] in splits]
    t = E.ModeTable([str(p) for p in profiles], C.MODES, None)
    names = {e["name"] for e in sel}
    if set(t.images) != names:
        raise LockError(f"profile images ({len(t.images)}) != manifest images of {splits} ({len(names)})")
    rows = defaultdict(dict)
    for p in profiles:
        for line in open(p):
            r = json.loads(line)
            rows[r["image"]][r["mode"]] = r
    sha = {e["name"]: e["sha256"] for e in sel}
    bad = [n for n in t.images if any(r.get("sha256") not in (None, sha[n]) for r in rows[n].values())]
    if bad:
        raise LockError(f"profile rows whose image sha256 differs from the manifest: {bad[:3]}")
    return t, rows, man, sel


def refit_state(t, rows, sel, train_split, cpu_est):
    """Controller-side state from the rows of `train_split` only (fd_common.load logic). cpu_est is REQUIRED in
    production (fit_retrained); tests may pass False explicitly to skip the CPU override."""
    if cpu_est is None:
        raise LockError("the registered CPU MBT2018 estimate is required (no silent fallback)")
    split = {e["name"]: e["split"] for e in sel}
    tr_names = [n for n in t.images if split[n] == train_split]
    tr_idx = np.array([i for i, n in enumerate(t.images) if split[n] == train_split])
    assert len(tr_idx) > 0
    t.enc_s = np.array([np.mean([rows[n][m]["enc_s"] for n in tr_names]) for m in C.MODES])
    t.dec_s = np.array([np.mean([rows[n][m]["dec_s"] for n in tr_names]) for m in C.MODES])
    if cpu_est is not False:
        est = json.loads(Path(cpu_est).read_text())["estimates"]
        for j, m in enumerate(C.MODES):
            if m in est:
                t.enc_s[j] = est[m][est[m]["primary"]]
    E.ModeTable.fit_info(t, tr_idx, "avif_5")
    return state_of(t)


def pools_of(t, man, sel, stage_a):
    idx = {n: i for i, n in enumerate(t.images)}
    pools = defaultdict(list)
    for e in sel:
        pools[e["split"]].append(idx[e["name"]])
    pools = {k: np.array(sorted(v)) for k, v in pools.items()}
    if stage_a:
        pools["train"], pools["val"] = pools["U_train"], pools["U_val"]
    full = {e["name"]: k for k, e in enumerate(man)}
    man_idx = np.array([full[n] for n in t.images])
    return pools, man_idx


def recall_table(t, rows, sel, pools=("U_test",)):
    split = {e["name"]: e["split"] for e in sel}
    I, Mn = len(t.images), len(C.MODES)
    rec = {v: dict(tp=np.full((I, Mn), np.nan), ngt=np.full(I, np.nan), tp_orig=np.full(I, np.nan))
           for v in RECALL_VARIANTS}
    for i, n in enumerate(t.images):
        if split[n] not in pools or "tp_veh" not in rows[n].get("original", {}):
            continue
        for v in RECALL_VARIANTS:
            rec[v]["ngt"][i] = rows[n]["original"][f"n_gt_{v}"]
            rec[v]["tp_orig"][i] = rows[n]["original"][f"tp_{v}"]
            for j, m in enumerate(C.MODES):
                rec[v]["tp"][i, j] = rows[n][m][f"tp_{v}"]
    return rec


# ---------------------------------------------------------------- validation API (relabelled fd_v1 in tests)
def load(profile, manifest, arm, cpu_est=None):
    """Single-profile loader used by test_ix_common (fd_v1 relabelled). arm 'frozen' | 'retrained' (refit here)."""
    assert arm in ("frozen", "retrained")
    t, rows, man, sel = build_table([profile], manifest)
    st = fd_v1_state() if arm == "frozen" else refit_state(t, rows, sel, "U_train", cpu_est)
    apply_state(t, st)
    pools, man_idx = pools_of(t, man, sel, arm == "retrained")
    return t, pools, recall_table(t, rows, sel), man_idx, st


def init(profile, manifest, arm, cpu_est=None):
    import torch
    torch.set_num_threads(1)
    t, pools, rec, man_idx, st = load(profile, manifest, arm, cpu_est)
    AE.G["table"], AE.G["pools"], AE.G["man_idx"], AE.G["models"] = t, pools, man_idx, {}
    AE.G["ix_recall"], AE.G["ix_state"] = rec, st
    return t, pools, rec


# ---------------------------------------------------------------- production API
def fit_retrained():
    """Stage A step 'fit': refit on U_train (stage-A profile + REQUIRED CPU estimate) and serialise. Run once."""
    if STATE_FILE.exists():
        raise LockError(f"{STATE_FILE} exists; the retrained state is fitted once")
    if not CPU_EST.exists():
        raise LockError("registered CPU MBT2018 estimate missing (python ix_cpu.py select / measure / estimate)")
    t, rows, man, sel = build_table([PROFILE["A"]], MANIFEST, SPLITS["A"])
    st = refit_state(t, rows, sel, "U_train", CPU_EST)
    h = save_state(st, STATE_FILE, dict(fitted_on="U_train", manifest_sha256=MANIFEST_SHA,
                                        profile_A_sha256=sha_file(PROFILE["A"]), cpu_est_sha256=sha_text(CPU_EST),
                                        n_train=int(sum(e["split"] == "U_train" for e in sel))))
    return h


def load_stage_a():
    """U_train + U_val table with the serialised retrained state (H* tuning, RL training, checkpoint selection)."""
    t, rows, man, sel = build_table([PROFILE["A"]], MANIFEST, SPLITS["A"])
    meta = json.loads(STATE_FILE.read_text())
    if meta["profile_A_sha256"] != sha_file(PROFILE["A"]) or meta["cpu_est_sha256"] != sha_text(CPU_EST):
        raise LockError("stage-A profile or CPU estimate changed after the state was fitted")
    st = load_state(STATE_FILE)
    apply_state(t, st)
    pools, man_idx = pools_of(t, man, sel, True)
    return t, pools, man_idx, st


def init_stage_a():
    import torch
    torch.set_num_threads(1)
    t, pools, man_idx, st = load_stage_a()
    AE.G["table"], AE.G["pools"], AE.G["man_idx"], AE.G["models"] = t, pools, man_idx, {}
    AE.G["ix_state"] = st
    return t, pools


def lock_items():
    """Everything L2 pins (computed fresh; compared with L2_lock.json by verify_lock)."""
    sel = json.loads(SELECTION.read_text())["selection"]
    st = load_state(STATE_FILE)
    return dict(manifest_sha256=sha_text(MANIFEST), profile_A_sha256=sha_file(PROFILE["A"]),
                cpu_est_sha256=sha_text(CPU_EST), state_retrained_file_sha256=sha_text(STATE_FILE),
                state_retrained_hash=state_hash(st), frozen_state_hash=state_hash(frozen_state()),
                frozen_file_sha256=sha_text(FROZEN_FILE),
                tune_sha256=sha_text(TUNE_FILE), H_star=json.loads(TUNE_FILE.read_text())["H"],
                frozen_H_star=json.loads((REPO / "runs" / "uav_semcom" / "paperA_fd" / "phase2" / "tune.json")
                                         .read_text())["H"],
                ckpts_sha256=sha_text(CKPTS), selection_sha256=sha_text(SELECTION),
                checkpoints={f"{k}_s{s}": dict(path=v["ckpt"], sha256=v["ckpt_sha256"])
                             for k, d in sel.items() for s, v in d.items()},
                code={f: sha_text(HERE / f) for f in LOCKED_CODE})


def verify_lock():
    """Raise LockError unless L2_lock.json exists and every pinned item (including checkpoint files) verifies."""
    if not LOCK.exists():
        raise LockError("L2 lock missing: run stage A and `python ix_stage.py lock` first")
    lk = json.loads(LOCK.read_text())
    now = lock_items()
    bad = [k for k in now if now[k] != lk["items"].get(k)]
    for name, v in lk["items"]["checkpoints"].items():
        if not Path(v["path"]).exists() or sha_file(v["path"]) != v["sha256"]:
            bad.append(f"checkpoint {name}")
    if bad:
        raise LockError(f"L2 verification failed: {bad}")
    return lk


def load_eval(stage, arm):
    """Test-stage table: stage 'uavdt' (arm frozen | retrained) or 'x1' (frozen). Verifies L2 first; loads the frozen
    or the serialised retrained state; never fits."""
    assert (stage, arm) in (("uavdt", "frozen"), ("uavdt", "retrained"), ("x1", "frozen")), (stage, arm)
    lk = verify_lock()
    t, rows, man, sel = build_table([PROFILE[stage]], MANIFEST, SPLITS[stage])
    st = frozen_state() if arm == "frozen" else load_state(STATE_FILE, lk["items"]["state_retrained_hash"])
    apply_state(t, st)
    pools, man_idx = pools_of(t, man, sel, False)
    rec = recall_table(t, rows, sel) if stage == "uavdt" else None
    return t, pools, rec, man_idx, st, man


def init_eval(stage, arm):
    import torch
    torch.set_num_threads(1)
    t, pools, rec, man_idx, st, man = load_eval(stage, arm)
    AE.G["table"], AE.G["pools"], AE.G["man_idx"], AE.G["models"] = t, pools, man_idx, {}
    AE.G["ix_recall"], AE.G["ix_state"] = rec, st
    return t, pools, rec
