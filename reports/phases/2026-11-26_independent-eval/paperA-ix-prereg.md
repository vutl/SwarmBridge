# Paper A independent evaluation (`paperA_ix`): pre-registration, FINAL v3 (27/11)

**Status: APPROVED** (ChatGPT final scientific review of `e964ad41`). This version adds the approved corrections and
the execution requirements of that review (§10). Design, data, metrics, hypotheses and scope are unchanged.

**Done so far:** data acquisition, integrity and overlap audit, manifest, frozen/refitted loader and its tests.

**Not done:** no profiling of any new image, no detector run on new images, no tuning, training or simulation. No
performance statistic of any new image exists.

**Answers:**
- ChatGPT review of `ebf9dcc2` (recall decision, execution order, implementation checks, data integrity, statistics,
  mode utilisation, terms).
- Supersedes `paperA-ix-prereg-draft.md`; the draft is kept as history.

**Code:**
- `ix_audit.py`: audit;
- `ix_manifest.py`: manifest;
- `ix_modes.py`: fd_v1 mode audit;
- drivers and tests: §10 (added 27/11).

## 1. Data and integrity (done)

### 1.1 Sources and hashes

| Dataset | File | sha256 | Content |
|---|---|---|---|
| VisDrone2019-DET test-challenge (X1) | `VisDrone2019-DET-test-challenge.zip` (official README Drive link) | `635e899e…a2da69` | 1,580 JPEG, no labels |
| UAVDT-Benchmark-M images | `UAV-benchmark-M.zip` (authors' Drive link; browser download by the user) | `e82675c9…585b08` | 40,735 JPEG, 50 sequences; `unzip -t` OK; images only |
| UAVDT DET/MOT toolkit + GT | `UAV-benchmark-MOTD_v1.0.zip` (authors' Drive link) | `1e565da8…44754b5b` | GT text read; no toolkit code executed |

### 1.2 Cross-dataset audit (`ix_audit.py full` → `runs/uav_semcom/paperA_ix/audit_full.{json,log}`)

**Groups compared:**
- **VD:** VisDrone history, 35,676 frames: DET train / val / test-dev and VID train / val.
- **X1:** 1,580 frames.
- **U_test:** the 20 official test sequences, 16,592 frames.
- **U_train:** 30 sequences, 24,143 frames.

**Checks:**
- identical sha256;
- near-duplicates, defined as dHash ≤ 20 and aHash ≤ 20 of 256 bits (the fd_v1 threshold);
- every query group against every other group, and against itself across sequences.

| Pair | Links | Nearest dHash, 0 / 1 / 5 % |
|---|---|---|
| X1 ~ VD | **1 near-duplicate** (0000332 ~ DET-train 0000331) | 4 / 51 / 63 |
| U_test ~ VD | 0 | 55 / 63 / 68 |
| U_train ~ VD | 0 | 38 / 47 / 64 |
| X1 ~ U_test, X1 ~ U_train | 0, 0 | 53 / 66 / 75; 42 / 56 / 67 |
| U_train ~ U_test | 0 | 58 / 65 / 72 |
| across sequences within X1, U_test, U_train | 0, 0, 0 | — |

**Deterministic exclusion direction** (applied by `ix_audit.full`):
- VisDrone history is never excluded.
- The official UAVDT test split is preserved against every non-history pool.
- X1, which is not an official split, yields to everything else.

| Link | Sequence excluded |
|---|---|
| X1 ~ any pool | the X1 sequence |
| U_test ~ VD | the U_test sequence |
| U_train ~ U_test | the U_train sequence |
| U_train ~ VD | none (U_train only trains the X2-retrained arm) |

**Result:** only X1 0000332 is excluded. No UAVDT sequence is excluded, so the X1 rule (≥ 15 sequences) and the
X2 rule (≤ 5 test sequences excluded) are both met.

**Wording in the paper:** "previously unused images with sequence and duplicate screening", not "geographically
independent scenes".
- Hashes detect copies and near-copies, not shared locations.
- UAVDT location independence rests on the authors' statement. The Mxx id groups mix train and test sequences.

### 1.3 Facts corrected by the audit

- UAVDT frames are **1024×540** (38,829) and **960×540** (1,906), not 1080×540 as stated in the paper and in the
  earlier review.
- **UAVDT M0207** (train) has 885 images but annotations end at frame 571. Rule: only frames 1 … last annotated frame
  of each sequence are eligible (gt_whole ∪ gt_ignore). All other sequences have images = last annotated frame.
- UAVDT image counts: train 24,143, test 16,592; these match the official archive.

### 1.4 Manifest `manifest_ix_v1.json`

sha256, LF-normalised: `734ea49b8d71e1caed6a2fc7f2198dcfb3685bfb2ea05654fb0971bc1f28bd92`.

**Rules:** cap of 30 frames per sequence at `round(linspace(0, n−1, 30))`. U_val: the train sequences sorted, permuted
with `numpy.default_rng(0)`, first `round(0.15·30) = 4`.

| Pool | Sequences | Frames | GT boxes | Resolutions |
|---|---|---|---|---|
| U_train | 26 | 780 | 13,333 | 1024×540 750, 960×540 30 |
| U_val | 4 (M0202, M0603, M1005, M1202) | 120 | 2,166 | 1024×540 |
| U_test | 20 (official) | 600 | 13,494 | 1024×540 |
| X1 | 52 | 631 | n/a | 6 VisDrone resolutions |

Per image, the manifest stores: name, sequence, source, split, w, h, sha256 of the image, frame index, number of
gt_whole boxes and ignore regions, sha256 of the frame's GT lines, and gen_seed (position in the order U_train, U_val,
U_test, X1).
- The GT box count is an annotation statistic, not a performance statistic.
- The manifest holds names and hashes only. No image content is in git.

## 2. Frozen and refitted components (implemented and tested)

**Controller-side state (`ix_common.FROZEN_KEYS`):**
- per-mode encode and decode latency;
- per-mode mean payload and mean PSNR (used for MPC future arrivals);
- avif_5 probe-predictor coefficients;
- the probe mode index.

**Frozen arms (X1, X2-frozen):**
- The state is copied from `fd_common.load()`, which fits on fd_v1 rl_train.
- Pinned hash: `ix_common.FROZEN_EXPECT = 39018855000424ebb39fb71c5702a626ab268132d0ca22b4b5b8e30843c6d3bf`.
- Also frozen: H\* (reference setting), every MPC / gate / band / calibration / LA / scheduler / reward parameter, and
  the YOLO11s weights.

**Retrained arm (X2-retrained):**
- The state is refitted from U_train rows only: latency means, the CPU MBT2018 estimate from a U_train sample (same
  procedure and size as fd_v1), mean payload and PSNR, and the probe predictor.
- H\* is chosen among the same 22 rules on U_train training seeds, for the reference setting and each grid setting.
- RL: PPO, QR-DQN and Discrete SAC × seeds 0–2 × 2M decisions, with fd_v1 hyperparameters.
- Checkpoints are selected on U_val with 20 validation seeds and the fd_v1 rule.

**Locking:** after the state is set, `fit_info` is disabled on the table instance, so any call raises `RefitForbidden`.
`env.ModeTable.__init__` averages latency over all rows it reads, but those values are always overwritten (tested).

**Pools:** the frozen arms have **no** `train` / `val` pool aliases. The retrained arm maps `train` / `val` to
U_train / U_val.

**Tests (`test_ix_common.py`, 11 passed, 132 s).** They use the fd_v1 profile relabelled as an ix manifest.
1. The frozen state equals fd_v1's: arrays bit-identical, same hash, same true-outcome arrays, same pool indices.
2. The pinned hash matches.
3. `fit_info` raises in both arms.
4. Frozen state ignores new rows: every row is perturbed (bits ×2, PSNR +3, latency ×5), the state hash is unchanged,
   the true outcomes change, and `predict` uses the new image's own probe payload with the fd_v1 coefficients.
5. Retrained state uses U_train rows only: perturbing every non-U_train row leaves the state hash unchanged;
   perturbing U_train rows changes it.
6. The refit code path, given fd_v1 rl_train and the fd_v1 CPU estimate, reproduces the fd_v1 state (atol 1e−12).
7. The frozen arm has no train / val pool.
8. Behavioural equivalence under a **deterministic budget** (a `Deadline` subclass that always allows planning).
   Arms H\*, guarded (`+mpcng`) and calibrated (`+mpc`), TD seeds 20000–20001: episode returns and delivery logs are
   identical between `fd_common.init` and `ix_common.init(frozen)`.
9. The **real timing path** (50 ms wall-clock `Deadline`) runs on the ix loader: guarded arm, seed 20002, finite
   return. Wall-clock-budgeted runs are not required to be bit-reproducible, as in fd_v1.

## 3. Recall on UAVDT (approved metric, details fixed)

**Detector:** frozen VisDrone YOLO11s, `a5_fidelity.detect`.
- imgsz 960, NMS IoU 0.7, max 500 detections, as in fd_v1;
- operating confidence ≥ **0.25**, match IoU ≥ **0.5** (`a5_fidelity.CONF, IOU`);
- greedy one-to-one matching by descending confidence (`a5_fidelity.match`).

**Ignore regions and denominator:**
- Detections with ≥ 50% of their area inside a UAVDT ignore region (`_gt_ignore.txt`, same frame) are removed, as for
  VisDrone ignore boxes.
- GT boxes with ≥ 50% of their area inside an ignore region are removed from the denominator.
- Every other gt_whole box of the frame counts (car, truck, bus), **whatever its occlusion (1–4) or out-of-view
  (1–3) level**. UAVDT marks these as attributes, not as exclusions.

**Variants (prereg §9 of the draft):**
- **(a) vehicle detection recall, primary:** all eligible GT boxes; predictions of VisDrone car, van, truck and bus,
  class-agnostic. Non-vehicle predictions are discarded.
- **(b) supporting:** class-aware, with car ← {car, van}, truck ← truck, bus ← bus.
- **(c) supporting:** class-aware, with car ← car only.

**Reporting:**
- System recall is computed as in fd_v1: the sum of TP over on-time, compliant deliveries divided by the sum of
  eligible GT over all frames of the episode.
- Original-image recall is reported first, under (a)–(c).
- Fields written by the U_test fidelity pass: `tp_veh`, `tp_cls_vancar`, `tp_cls_car`, `n_gt_*` (read by
  `ix_common.recall_table`).
- IoU 0.5 is the paper's system-level metric, not the UAVDT AP protocol (IoU 0.7). The absolute value is not compared
  with VisDrone's all-category recall.
- X1 recall is **unavailable**. No pseudo-labels are used.

## 4. Execution order (corrected per review)

| Step | Content | Lock |
|---|---|---|
| 1 | Approve this document | **L0** |
| 2 | Data acquisition and integrity / overlap audit | done |
| 3 | Manifest frozen; loader validated | manifest hash §1.4; loader tests §2; **L1** = commit of L0 including both |
| 4 | Profile **U_train and U_val only** (14 modes; payload, PSNR, LPIPS, times) | profile hashes |
| 5 | U_train / U_val fitting: state, CPU estimate, H\* tuning on training seeds, RL training, checkpoint selection on U_val | — |
| 6 | Hash-lock every fitted component: state hash, H\* names, checkpoint sha256, selection file | **L2** |
| 7 | Profile **X1 and U_test**; frozen-detector fidelity on **U_test** | profile hashes |
| 8 | Run the 8 registered test blocks (10,200 episodes); fail-fast verification | — |
| 9 | One analysis pass with the registered script; full report | — |

- No performance-related statistic of X1 or U_test exists before L2.
- The X1 / U_test profiling driver refuses to start unless the L2 lock file exists and its hashes verify.
- After step 7 begins, no data, arm, parameter or analysis change is allowed.

## 5. Evaluation blocks (unchanged; 10,200 episodes)

| Block | Content | Episodes | Seeds |
|---|---|---|---|
| X1-ref | 6 Phase 2 arms, reference setting | 600 | 30000–30099 |
| X1-abl | 8 ablation arms | 800 | 30100–30199 |
| X2F-ref | 6 arms, U_test, frozen | 600 | 31000–31099 |
| X2F-abl | 8 arms, U_test, frozen | 800 | 31100–31199 |
| X2R-ref | 6 arms, U_test, retrained | 600 | 32000–32099 |
| X2R-abl | 8 arms, U_test, retrained | 800 | 32100–32199 |
| X2R-grid | 6 settings × 4 arms | 2,400 | 32200–32299 |
| X2R-RL | 3 algorithms × 3 checkpoints × 4 arms | 3,600 | 32000–32099 (pairs with X2R-ref H\*) |

X2-frozen tests transfer of the heuristic-incumbent stack only. No zero-shot RL claim is made.

## 6. Statistics and pre-specified findings

**Families:** Holm within each block, exactly as in draft §6.
- reference: H1 and H3b;
- ablation: A1, A2, A4, A4b;
- **new N-family:** N1 nominal − H\*, N2 guarded − H\*, N3 calibrated − guarded;
- grid: G1 and G2 × 6;
- RL: R1 and R2 × 3.

Supporting contrasts get unadjusted CIs. The two X2 conditions share U_test: they are never pooled and never count
as independent replications.

**Principal independent finding (fixed before execution).** It is the paper's central claim, that uncertainty-band
screening improves predictive mode selection under the evaluated wireless conditions. No formal safety or robustness
guarantee is claimed; both tests concern expected return. Operationalised as:
- **N2 > 0:** guarded MPC beats the tuned incumbent;
- **A2 > 0:** the band alone improves on nominal MPC.

**Principal block:** **X1**, because it is unseen, in-domain, uses frozen artifacts and needs no transfer assumption.

Verdict rules, applied in this order (implemented in `ix_analysis.verdict` and unit-tested):

| Order | Verdict in a block | Rule |
|---|---|---|
| 1 | not supported | N2 or A2 Holm-rejected with a **negative** sign. This takes precedence over every other rule |
| 2 | supported | N2 and A2 both Holm-rejected with a positive sign |
| 3 | partially supported | exactly one of them Holm-rejected with a positive sign |
| 4 | not supported | neither Holm-rejected |

The rule is applied to X1 (principal), to X2-retrained (method transfer) and to X2-frozen (artifact transfer), each
reported under its own label. The labels are never swapped after results are seen. A summary sentence quotes all
three verdicts.

**Secondary findings:**
- N1 < 0 (nominal MPC loses to H\*);
- N3 (no detected calibration gain);
- A4 / A4b (horizon);
- recall contrasts in X2;
- grid and RL families.

All are reported with their own Holm results. Reversals and non-rejections are reported. "No detected difference" is
never written as "equivalent".

## 7. Mode utilisation

**Per block / arm / checkpoint:**
- selection counts and shares for all 14 modes;
- outcomes by mode: compliant / on time below the floor / late or dropped / no mode decision;
- per-mode profile statistics;
- modes selected ≤ 0.5%, with the generative modes named.

**Wording:** "rarely selected and rarely compliant under the current objective". No Pareto-dominance claim is made
without a formal dominance analysis. No mode, reward or quality floor is changed.

**Training action histograms (X2-retrained only):**
- Counted from the **actual training actions** in the trainer's step callback (`locals["actions"]`, read-only).
- Validation histograms are a separate statistic.
- Logging consumes no RNG and changes no decision. Test: a short training run with and without logging gives
  identical parameters.
- To be implemented and reviewed with the training driver before step 5.

For fd_v1, training action counts are "not recorded".

**fd_v1 results:** kept as audited (`ix_modes.py`; draft §7).

## 8. Data terms (open; the authors decide)

- **UAVDT:**
  - The authors' pages state research use and ask for citation of ECCV 2018.
  - The toolkit is GPL, but that licence covers the code, not dataset redistribution.
  - Use: research only, cite, no redistribution.
- **VisDrone test-challenge:**
  - The README designates it for the challenge; its labels are unavailable.
  - Our use is label-free communication simulation, not detection benchmarking.
  - No explicit permission exists for this use, so writing to the maintainers is recommended (a draft is given to the
    user).
- **Both:**
  - Images, reconstructions and per-image artifacts (profiles with captions, recall rows) stay outside public git.
  - The ix profile files are written under the WSL work directory, not under `runs/`. Only aggregate results go to
    git.
  - The professor's agreement covers the plan, not the dataset owners' permission.

## 9. Cost (unchanged)

About 1.5–2 days of unattended detached jobs:
- profiling: ≈ 3.8 h GPU;
- fidelity: ≈ 0.75 h GPU;
- RL training: ≈ 13 h CPU;
- tuning: 1–2 h CPU;
- test blocks: 3–4 h CPU.

## 10. Execution implementation (review of `e964ad41`; drivers and tests done 27/11)

### 10.1 Drivers (`swarmbridge/experiments/uav_semcom/`)

| File | Role |
|---|---|
| `ix_common.py` | tables; frozen / retrained controller state; serialisation; L2 lock (`lock_items`, `verify_lock`); stage-A loader (`load_stage_a`); evaluation loader (`load_eval`, verifies L2, never fits) |
| `ix_profile.py` | profiling (fd_v1 pipeline, `IxPipe.recon`) and UAVDT recall (§3); stages `A`, `uavdt`, `x1`; `uavdt` / `x1` refuse to start without a verified L2; prints timing and counts only |
| `ix_cpu.py` | CPU MBT2018 timing on U_train (rules below) |
| `ix_stage.py` | stage A: `fit`, `tune`, `train`, `select`; then `lock`, `verify`, `status` |
| `ix_eval.py` | the 8 test blocks; `x1` and `uavdt` groups run independently |
| `ix_analysis.py` | registered analysis (§6), principal verdict, recall variants, mode utilisation |
| `ix_actlog.py` | training-action histograms (read-only) |
| `a5_train.py`, `offpolicy_train.py` | new opt-in flags `--ix` (stage-A table) and `--log-actions`; defaults unchanged |
| `ax2_eval.py` | learned incumbents `ppoix` / `qrix` / `dsix` (checkpoints from `paperA_ix/ckpts.json`); existing paths unchanged |

### 10.2 State serialisation and the L2 lock

- **Frozen state.** It is serialised once from `fd_common.load()` to `paperA_ix/state_frozen_fd_v1.json`, with a pinned
  hash (`39018855…`). Evaluation loads this file, so no fitting function runs at evaluation time, not even on fd_v1.
- **Retrained state.** `ix_stage.py fit` fits once on U_train, with the CPU estimate **required**
  (`refit_state(cpu_est=None)` raises). The result is written to `paperA_ix/state_retrained.json`, which stores its own
  hash and the hashes of the stage-A profile and the CPU estimate. H\* tuning, RL training and checkpoint selection
  load this file (`load_stage_a`).
- **L2_lock.json** (`ix_stage.py lock`, written once, then committed) pins:
  - the manifest;
  - the stage-A profile (sha256 of the file);
  - the CPU estimate;
  - the retrained state file and its hash;
  - the frozen state file and its hash;
  - H\* (U_train) and the frozen fd_v1 H\*;
  - `ckpts.json`, `selection.json` and the sha256 of all 9 checkpoints;
  - the sha256 of 22 code files the test blocks depend on (`ix_common.LOCKED_CODE`).
- **Verification.** `verify_lock` recomputes every item and re-hashes the checkpoint files. Stage-B profiling,
  `ix_eval` and `ix_analysis` call it before anything else, and evaluation workers call it again inside `load_eval`.
  A change to any pinned item after L2 stops stage B; continuing would require a documented deviation.

### 10.3 CPU MBT2018 estimate on U_train

The fd_v1 procedure, with the following rules (fixed in `ix_cpu.py` before any timing):
- 48 frames;
- strata by resolution, quotas proportional to stratum size with a minimum of 2: 1024×540 → 46, 960×540 → 2;
- RNG `default_rng(20261127)`; one frame per distinct sequence first;
- 6 threads, one warm-up image discarded;
- frame-weighted primary estimate.

The sample is fixed with `ix_cpu.py select` and committed before `measure` runs.

### 10.4 Files kept outside git (WSL `~/work/paperA_ix/`)

- Profiles: `profile/profile_A.jsonl`, `profile_B_uavdt.jsonl`, `profile_B_x1.jsonl`. They hold captions and
  per-image values.
- Per-delivery records: `eval/<block>.deliv.json.gz`.
- RL weights: `rl/`.

**In git** (`runs/uav_semcom/paperA_ix/`):
- manifest, audits, states, tune, ckpts / selection (paths and hashes);
- the CPU sample and estimates;
- logs (timing and counts only);
- per-episode block results `eval/<block>.json` (simulation metrics, no image content);
- the analysis.

### 10.5 Tests (all passing, 27/11)

| Test file | Content |
|---|---|
| `test_ix_common.py` (12) | frozen state bit-identical to fd_v1; serialised frozen file equals fd_v1; perturbation invariance; refit uses U_train only; the refit path reproduces fd_v1; deterministic-budget behavioural equivalence (H\*, guarded, calibrated); real 50 ms timing path |
| `test_ix_lock.py` (12) | exact state round trip; fit requires the CPU estimate; fit runs once; stage A loads the serialised state with refit disabled; **evaluation loading with `refit_state`, `fit_retrained` and `ModeTable.fit_info` all disabled**; lock verification fails on a missing lock, a changed state, checkpoint, H\*, profile or code; stage-B profiling and every `ix_eval` block refuse without a lock |
| `test_ix_recall.py` (10) | one-to-one matching; van on car (veh and van→car hit, car-only miss); wrong vehicle class; non-vehicle predictions discarded; confidence and IoU thresholds; ignored detections; ignored GT removed from the denominator; all other boxes counted; GT reader |
| `test_ix_analysis.py` (10) | on stored fd_v1 results: the ablation block reproduces `summary_ablation_test.json` (A1–A9 CIs, Holm, recall cluster CIs) and the N family; the reference block reproduces the Phase 2 H1–H3b CIs and Holm; verdict precedence (7 cases); utilisation counts |
| `test_ix_actlog.py` (2) | PPO (2,048 steps) and QR-DQN (400 steps) give **bit-identical parameters** with and without `--log-actions`; histograms count every executed mode action (PPO total = `rl_steps_mode`; QR-DQN warm-up 200 + policy 200) |
| `test_ix_profile.py` (2, GPU, run under the GPU lock) | `IxPipe.recon` reproduces the stored fd_v1 profile rows of two rl_train images (codec bytes equal, \|ΔPSNR\| < 0.05 dB, same captions); the full stage path including recall fields runs on one UAVDT **training** image |
| `test_ix_smoke.py` (1) | end to end on the sandbox: `ix_stage tune` (tiny) → `lock` → `ix_eval` x1_ref / x2r_ref / x2r_rl (2 seeds, fd_v1 checkpoints as stand-ins) → `ix_analysis` |

### 10.6 Commands

Launch detached via WMI, from PowerShell:

```powershell
$cmd = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\Documents\SwarmBridge\runs\uav_semcom\paperA_ix\scripts\run_stage.ps1 -Stage A'
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
```

- **Stage A:** `-Stage A`, about 16 h. GPU profile of U_train / U_val (under the GPU lock), then CPU timing, fit,
  H\* tuning, 9 RL runs and selection. It stops with `STAGE_A_DONE` in `logs/stageA.log`.
- **Mandatory stop.** Review the stage-A outputs, then run `python ix_stage.py lock`, commit `L2_lock.json`, and run
  `python ix_stage.py verify`.
- **Stage B, UAVDT:** `-Stage uavdt`, about 6 h. U_test profile + fidelity, then 6 blocks (8,800 episodes) and the
  analysis.
- **Stage B, X1:** `-Stage x1`, about 2.5 h. It is started only after the VisDrone test-challenge question is
  resolved. X1 profile, then 2 blocks (1,400 episodes) and the analysis.
