# Paper A extended evaluation: Phase 0 dataset audit and protocol (23/11)

**Status:** audit done; protocol PROPOSED for review. No profiling, training or test run has been started.

**Brief:** ChatGPT review of 23/11 (two workstreams: dataset generalization, and wireless sensitivity on the new data
protocol).

**Audit script:** `swarmbridge/experiments/uav_semcom/fd_audit.py` (read-only). Output: `runs/uav_semcom/paperA_fd/audit.log`.

All E1–E7 results stay as they are. The new work uses its own namespace:
- results in `runs/uav_semcom/paperA_fd/`;
- reports in `reports/phases/2026-11-23_paperA-fulldata/`;
- a new manifest. The A5 manifest and profiles are not modified.

## 1. What exists today (verified on disk)

| Split | Frames | Sequences (file prefix) | Annotations | On disk |
|---|---|---|---|---|
| VisDrone2019-DET-train | 6,471 | 208 | 6,471 | WSL `~/data/visdrone_dl/` |
| VisDrone2019-DET-val | 548 | 76 | 548 | repo `data/visdrone/` |
| VisDrone2019-DET-test-dev | 1,610 | 61 | 1,610 | WSL `~/data/visdrone_dl/` |

- Sequences are skewed:
  - **DET-train:** median 7 frames per sequence, largest 434. The five largest sequences hold 1,811 frames.
  - **DET-test-dev:** one sequence holds 516 of the 1,610 frames (32%); the next holds 345.
- Consecutive frames of a sequence are near-duplicates.

**The current A5 image table (E1–E7)**, file `runs/uav_semcom/a5/manifest.json`:
- 548 DET-val frames, split by sequence into:
  - train: 304 frames, 32 sequences;
  - val: 84 frames, 14 sequences;
  - test: 128 frames, 28 sequences;
  - 32 frames excluded;
- plus 403 DET-test-dev frames (every 4th frame by name, 49 sequences).
- Every (image, mode) pair is profiled for 14 modes.
- Detector fidelity (YOLO11s recall per image and mode) exists for all 951 frames and all 15 modes, including the
  original image.

**The detector** (`a5_det.py`, weights `~/work/a5_det/yolo11s_visdrone_960/weights/best.pt`):
- YOLO11s trained on DET-train minus every sequence that also occurs in DET-val.
  - training: 5,903 frames, 166 sequences;
  - checkpoint-selection hold-out: 271 frames, 18 sequences.

## 2. Leakage audit

| Check | Result | Consequence |
|---|---|---|
| DET-train ∩ DET-val, by sequence | **24 sequences**: 297 train frames, 28 val frames | The 297 DET-train frames must stay out of any RL training pool, because DET-val is an evaluation pool |
| DET-train ∩ DET-test-dev | 0 | none |
| DET-val ∩ DET-test-dev | 0 | none |
| YOLO training ∩ DET-val / test-dev | 0 / 0 | Recall on DET-val and test-dev is on detector-unseen scenes |
| YOLO training ∩ DET-train | 5,903 frames (all non-hold-out frames outside the 24 shared sequences) | **No detector recall may be reported on DET-train frames.** Return-based results are unaffected, because the reward does not use the detector |
| YOLO hold-out (271 frames, 18 sequences) | Not trained on; used only to select the YOLO checkpoint | Usable for recall, with a small optimistic bias. Never used in any SwarmBridge RL, heuristic or method round |

**Prior use of the evaluation pools** (must be stated in the paper):
- DET-val: all 548 frames were used during development. They served as the E1–E7 train, val and test pools, and the
  heuristic families and MPC parameters were designed on them.
- DET-test-dev: 403 frames were used as the test-dev pool, in E3, step C and E7. These frames come from 49 of the 61
  sequences. Those 49 sequences hold 1,592 of the 1,610 frames, so the remaining 1,207 frames are mostly neighbours
  of frames already seen. Only 12 sequences (18 frames) are untouched.
- **There is no fully untouched public test set with labels.** The DET test-challenge split has no public ground truth.
- The closest thing to a fresh pool is the YOLO hold-out (F below): never seen by RL or by the method design.

## 3. Proposed split (manifest `fd_v1`)

**Unit:** the sequence (file-name prefix). Frames of one sequence never cross pools.

**Per-sequence cap:** at most 30 frames per sequence, evenly spaced in name order. This applies to the RL pools, to F,
and to the primary test-dev pool. Reasons:
- it bounds how much one scene can dominate an episode;
- it removes near-duplicate frames;
- it cuts profiling cost.

The uncapped test-dev pool is a pre-registered secondary analysis.

| Pool | Source | Frames | Sequences | Return | Recall | Prior use |
|---|---|---|---|---|---|---|
| `rl_train` | DET-train minus the 24 val-shared sequences minus the 18 YOLO-hold-out sequences; 85% of the remaining 166 sequences (seed 0) | ≈ 1,600 | ≈ 141 | training / tuning only | — (detector trained on these) | none |
| `rl_val` | the other 15% of those sequences | ≈ 300 | ≈ 25 | checkpoint selection only | — | none |
| `F` (fresh) | YOLO hold-out | 168 | 18 | **test** | yes (small checkpoint-selection bias) | none in SwarmBridge |
| `TD` (primary test) | DET-test-dev, all 61 sequences | 582 | 61 | **test** | yes | 403 frames used earlier (E3, step C, E7) |
| `TD_all` (secondary) | DET-test-dev, uncapped | 1,610 | 61 | test | yes | as above |
| `V` (replication) | DET-val, all frames (maximum 35 per sequence, so the cap does not bind) | 548 | 76 | test | yes | used for development |

**Primary test pool: TD.** It is the largest, has the most sequences, is unseen by the detector, and was only partly
used before. F (the only unused pool) and V are replications. F is small (18 sequences), so its scene-level CIs will be
wide.

**Size compared with today:**
- RL training pool: about 1,600 frames from about 141 sequences, against 304 frames from 32 sequences;
- evaluation pools: 582 + 168 + 548 frames from 155 sequences, against 128 frames from 28 sequences (test) plus 403
  (test-dev).

## 4. What is re-derived on the new protocol, and what stays frozen

**Re-derived, from `rl_train` only:**
- the image–mode table: payload, PSNR, LPIPS (used by `env.ModeTable`) and encode/decode times;
- the probe-based payload/PSNR predictor (`fit_info`, avif_5 probe);
- the per-mode encode/decode latency means. Today `ModeTable` averages over all table rows; in `fd_v1` the mean is over
  `rl_train` rows. This is pre-registered as an implementation fix;
- H\*, chosen among the same 22 rules on training seeds;
- the RL policies (Phase 3).

**Frozen, unchanged from E1–E7:**
- the 14 modes and their codec settings;
- the YOLO11s weights;
- the utility and reward, the LA PHY and its parameters, and the JA scheduler;
- every MPC and calibration parameter: N 8, H 4, 50 ms budget, 2 dB bins, 20 s minimum, band rule, switch rule;
- the RobustMPC-style and nominal baselines.

**Generator seed of the generative modes:** today it is the image's index in the profiled list. In `fd_v1` it becomes
the image's index in the `fd_v1` manifest, recorded per row.

## 5. Phase 1 (profiling): plan and cost

**What is profiled:**
- `profile_a11.py` (10 modes) and `profile_a12.py`, restricted to the 4 generative or thumbnail modes that are in the
  table (cap_t2i, thumb64_up, cap_thumb64_i2i_t0.6, cap_thumb128_i2i_t0.3). The 3 unused a12 modes are skipped.
- Native resolution (`--size 0`), the same hardware as the original profiles (RTX 3070, i5-14400F, WSL2).
- `a5_fidelity.py` (YOLO recall per image and mode, including the generative modes) only for the pools where recall is
  reported: F, TD, and TD_all. V is already done.

**Measured cost**, from the existing profiles (mean encode + decode time per image, on DET-val):
- the 10 a11 modes: 1.53 s;
- the 4 a12 modes used: 2.08 s;
- metric computation (PSNR, MS-SSIM, LPIPS, CLIP) and the BLIP caption come on top; their timing is not logged.
- Planning figure: **6–8 s per image** for profiling, and about 4.5 s per image for fidelity.

| Work | Images | Estimate |
|---|---|---|
| Profile `rl_train` + `rl_val` + F (all new) | ≈ 2,065 | 3.5–4.5 h GPU |
| Profile TD and TD_all frames not profiled yet | 1,207 | 2–2.7 h |
| Fidelity on F + all of test-dev (403 already done) | 168 + 1,207 | ≈ 1.7 h |
| **Total** | | **≈ 7–9 h GPU**, run detached (WMI) with a resumable cache |

Uncapped profiling of DET-train would add about 4,300 frames, roughly 8 more hours. It is not proposed.

**Integrity checks:**
- every new row stores the image's sha256 (from the manifest);
- a cache keyed by (image sha256, mode, code version) skips work already done;
- before the full run, a smoke run re-profiles 5 DET-val images and compares them with the existing rows. Bytes, PSNR
  and LPIPS must match exactly for the deterministic codecs and within tolerance for the GPU codecs; timing is reported
  only.
- Rows that fail for resource reasons are listed, not imputed.

**Predictor check:** the error of the probe predictor (log-bits and PSNR) is reported on `rl_val`, F, TD and V, next to
the error of today's predictor (fitted on 304 frames).

**Disk use is small:** profiles are JSONL, and decoded images are not stored. There are 849 GB free.

## 6. Phases 2–4: design to pre-register before any test run

This section is a sketch. A separate pre-registration file will be committed after the Phase 1 results and before
Phase 2 is run.

**Phase 2: main setting** (K 5, M 2, 1 MHz, nominal LA; incumbent H\* re-tuned on `rl_train`).

| Arm | Purpose |
|---|---|
| H\* | incumbent alone |
| H\* + nominal MPC | the existing nominal baseline: no gate, band [1, 1] |
| **H\* + nominal MPC with gate and band** (new arm, `+mpcng`) | the same planner, gate and band as the calibrated arm, with only the rate predictor nominal. This isolates the ACK calibration, as the review asked: today the nominal arm differs from the calibrated one in both the predictor and the gate/band |
| H\* + RobustMPC-style MPC | existing baseline |
| H\* + calibrated MPC | the method |

**Metrics:**
- return and return per image;
- on-time delivery rate, drop rate, late rate and Qmin-violation rate;
- PSNR of delivered images;
- Qmin-compliant recall (F, TD, V only);
- energy, goodput, decision time (p99) and the share of slots over 50 ms.

**Statistics:**
- paired 95% t-CI over 100 environment seeds;
- plus a **two-way cluster bootstrap over environment seeds and image sequences**, as for the earlier video-level
  CIs. This is the generalization-by-scene requirement.

**Hypotheses** (decision rules to be fixed in the pre-registration):
- H1: calibrated − H\* > 0;
- H2: calibrated − RobustMPC-style > 0;
- H3: calibrated − nominal > 0;
- H3b: calibrated − (nominal with gate and band) > 0, which isolates the calibration;
- each on TD as the primary pool, with F and V as replications.

**New seed ranges** (no overlap with any earlier range):
- tuning: 0–29 on `rl_train`;
- validation: 3000–3019 on `rl_val`;
- tests: TD 20000–20099, F 22000–22099, V 21000–21099, TD_all 23000–23099.

**Phase 3: RL.**
- First, zero-shot evaluation of the existing PPO-LA, QR-DQN and Discrete SAC checkpoints. Their observation does not
  depend on the image pool, so they are compatible.
- Then retraining on `rl_train`: 3 seeds each, 2M decisions, checkpoint chosen on `rl_val`. Hyper-parameters are
  frozen at the E7 choices (QR-DQN lr 1e-4, Discrete SAC entropy ratio 0.3, PPO as A-X9); no new sweep.
- About 9 runs at roughly 3.8 h, two at a time: **about 17 h CPU**.
- Arms: X, X + nominal MPC, X + calibrated MPC (and X + `+mpcng`).
- E7 studies mode-selection RL only. It does not compare with joint trajectory/resource DRL such as Huang's DSAC.

**Phase 4: sensitivity.**
- The E4/E6 grid (K, M) ∈ {(3, 2), (5, 2), (7, 2), (5, 3), (5, 4)} and bandwidth ∈ {0.5, 1, 2} MHz on `fd_v1`.
- H\* and the MPC variants only; no RL per configuration.
- The JA scheduler stays fixed, which will be stated.
- Latency reported; no real-time claim where the scheduler exceeds the budget.

## 7. Risks and open questions for the review

1. **No untouched labelled test set exists.** TD is partly used; F is unused but small (18 sequences). Is
   "TD primary, F and V as replications" acceptable?
2. **The 30-frames-per-sequence cap.** It balances scenes and cuts cost but does not use every frame; TD_all is the
   uncapped secondary analysis. An alternative is no cap for TD, with sequence-weighted statistics instead.
3. **Recall cannot be measured on `rl_train` / `rl_val`** (the detector was trained there). Recall is therefore never
   used for tuning or selection, which is already true today.
4. **DET-train frames are larger** (up to 2000×1500), so payloads and encode times shift. The LA PHY and utility are
   unchanged, so absolute returns are not comparable with E1–E7; only within-protocol contrasts are.
5. **Profiling the generative modes is the dominant cost** (about 2 s per image). Skipping them would change the mode
   set, so they are kept.
6. **Latency means change** from "all table rows" to "`rl_train` rows". This is a small, pre-registered implementation
   change that also fixes a mild test-pool dependence in the current table.

**Next step, if approved:**
- Phase 1 smoke test: 5 DET-val images for the reproduction check, and 20 `rl_train` images for timing;
- then the full detached profiling run;
- then the Phase 1 report: coverage, failures, timing, and predictor error per pool.
