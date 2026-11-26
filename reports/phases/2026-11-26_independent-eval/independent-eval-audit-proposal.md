# Paper A: independent evaluation, stage 1 audit and proposal (26/11)

**Status:** audit + proposal only. No download, profiling, training, simulation or manuscript edit has been done.
Awaiting review before stage 2.

**Trigger:** user + ChatGPT (26/11): the fd_v1 test pools TD and V were involved in earlier development, so the paper
has no fully untouched test set. Options discussed: (A) external dataset UAVDT, (B) DroneVehicle, (C) VisDrone
DET test-challenge, (D) retrain on VisDrone official splits. User proposal: retrain **and** add a new dataset.

## 1. Verified facts (repo, manifests, `paperA-full-data-protocol.md`)

| Claim in the 26/11 discussion | Verified | Source |
|---|---|---|
| "403 of the 582 TD frames were used before" | **Wrong.** 146/582 TD frames are identical to earlier frames; 49/61 TD sequences were seen (of the other 436 frames, 418 are neighbours in seen sequences and 18 come from the 12 unseen sequences). TD_extra: 257/1,028 frames, 8/8 sequences | `a5/manifest.json` vs `manifest_fd_v1.json` (computed 26/11) |
| V used throughout development | Yes: 548/548 frames, 76/76 sequences | same |
| rl_train / rl_val touched earlier | No: 0 frames, 0 sequences | same |
| F used for detector checkpoint selection | Yes, and nothing else. 0 frames and 0 sequences in any earlier SwarmBridge round | protocol §2 |
| Final RL / H\* / predictor fitted on test data | No: all on rl_train (checkpoints on rl_val) | protocol §4, phase 3 prereg |
| UAVDT / DroneVehicle used in the project | No images ever downloaded. Only UAVBench **question JSON files** that name UAVDT / DroneVehicle were counted in the Huang step 0 audit (`runs/huang_step0/uavbench_audit/report.json`), unrelated to Paper A | grep over repo + WSL `~/data` |
| VisDrone DET test-challenge used | No: never downloaded (WSL `~/data/visdrone_dl` holds only DET-train and test-dev; repo `data/visdrone` holds DET-val and VID train/val) | disk listing |
| DET test-challenge has public labels | No (VisDrone README / toolkit; results go through the evaluation server) | r1b notes §6, VisDrone docs |

**The problem is real, but it is test-set reuse at the level of the method design, not training leakage.** Every fitted
component of the final evaluation was fitted on rl_train only.

### 1a. A pool that is already method-unseen: F

F (168 frames, 18 sequences) was never seen by any SwarmBridge method round. Its only prior use is selecting the
YOLO checkpoint, which can bias **recall** but not **return** (the reward does not use the detector). The registered
reference ablation already contains F:

| Contrast (return) | TD | **F** | V |
|---|---|---|---|
| nominal MPC − H\* | −10.6 [−14.8, −6.4] | **−8.6 [−13.1, −4.2]** | −6.8 [−10.7, −2.9] |
| band screening (guarded − nominal, A2) | +15.2 [+11.4, +18.9] | **+16.2 [+12.1, +20.3]** | +13.4 [+10.0, +16.7] |
| guarded − H\* (A8) | +7.4 [+5.1, +9.8] | **+5.4 [+2.6, +8.2]** | +2.3 [+0.2, +4.3] |
| calibrated − guarded (A7) | −0.7 [−2.9, +1.4] | **+2.6 [+0.1, +5.1]** | +2.5 [+0.7, +4.3] |

So the return findings already replicate in direction on a method-unseen pool. Its weakness is size: 18 sequences give
wide cluster CIs. F was a supplementary pool in the registration, so it cannot be promoted to primary after the fact,
but the paper should say explicitly that F is method-unseen.

## 2. What retraining fixes and what it does not

| Problem | Retrain on VisDrone again | New untouched data |
|---|---|---|
| Test images in model training | already absent in fd_v1 | yes |
| Checkpoint / heuristic / predictor selection on test | already absent in fd_v1 | yes |
| Method design (band, gate, H, switch rule) chosen while looking at V / TD | **no** | **yes** |
| Generalisation beyond VisDrone | no | yes, if external |

fd_v1 already **is** "retrain on clean splits": 9 RL runs × 2M decisions on rl_train, H\* re-tuned on rl_train,
predictor fitted on rl_train. Retraining on VisDrone again and testing on TD costs about 13 h CPU and removes
nothing. **Retraining is useful only on a new dataset**, where it tests whether the procedure (not the VisDrone
artifacts) transfers. That is how the user's "retrain + new dataset" idea is used below.

## 3. Proposed design: two new blocks, frozen method

The method is frozen as registered in fd_v1: 14 modes and codec settings, utility and reward weights, LA PHY, JA
scheduler, the network settings, and every MPC parameter (N 8, H 4, 50 ms, 1 s.e. margin, band rule, 20 s gate,
2 dB bins). Nothing is changed after seeing any new outcome.

### Block X1: VisDrone DET test-challenge (same domain, label-free, return endpoints)

- 1,580 images, never downloaded. **Return, delivery, PSNR, payload and mode statistics need no labels** (PSNR uses
  the original image), so every return claim (F1–F3, ablation, network grid, learned incumbents) can be tested on a
  truly untouched VisDrone pool. Recall cannot be computed (no public GT).
- All fd_v1 artifacts stay frozen: RL checkpoints, H\* per setting, predictor, latency means. No retraining
  (same domain).
- **First audit step after download (before any profiling):** sequence-prefix overlap with DET-train / val /
  test-dev. Sequences that share a prefix with any fd_v1 or earlier pool are excluded by rule; the remaining
  sequence count is reported. If fewer than about 15 independent sequences remain, X1 becomes secondary.
- Sampling: the fd_v1 rule (at most 30 frames per sequence, evenly spaced in name order), fixed before download.

### Block X2: UAVDT (external dataset, labels, recall)

Two arms of evidence on the **UAVDT test split** (official sequence split; never used for anything):

- **X2-frozen (zero-shot transfer):** VisDrone fd_v1 artifacts unchanged (RL checkpoints, H\*, predictor, latency
  means). Tests whether the trained system transfers.
- **X2-retrained (procedure transfer):** the fd_v1 procedure re-applied on the UAVDT **train** split only:
  profile table, latency means, probe predictor, H\* chosen among the same 22 rules on training seeds, RL 3 algorithms
  × 3 seeds × 2M decisions, checkpoints chosen on a UAVDT-train sequence hold-out (15%, same rule as rl_val).
  Tests whether the method works when everything data-dependent is refitted on a new domain. This is the "retrain"
  part.
- **Recall:** frozen VisDrone YOLO11s, class-aware matching mapped to the UAVDT classes (car, truck, bus), UAVDT
  ignore regions treated like VisDrone ignore boxes. Original-image recall is reported first so readers see the
  domain shift. No detector fine-tuning on UAVDT test. (Optional, decided now or never: fine-tune on UAVDT train as a
  secondary detector.)
- **Checks before profiling:** official download and terms, sequence split, exact class definitions (whether UAVDT
  "car" includes vans; the mapping VisDrone car+van → car is decided before looking at any recall), resolution
  (UAVDT frames are about 1080×540, smaller than most VisDrone frames, so payloads and network load are lower; the
  network settings are **not** rescaled; this is reported as is).
- Sampling: at most 30 frames per sequence, evenly spaced, from all test sequences (about 20 × 30 = 600 frames), and
  the same rule on train sequences (about 30 × 30 = 900 frames; smaller than VisDrone rl_train's 1,523).

DroneVehicle is the fallback if UAVDT cannot be obtained or its terms do not allow this use.

### Evaluation blocks (per dataset block; fresh seed ranges, registered before running)

| Block | Arms | Episodes |
|---|---|---|
| Reference comparison | 6 (Phase 2 arms) | 600 |
| Component ablation | 8 | 800 |
| Network grid | 6 settings × 4 arms | 2,400 |
| Learned incumbents | 3 algorithms × 3 checkpoints × 4 arms | 3,600 |
| **Total** | | **7,400** |

X1 (frozen) 7,400 + X2-frozen 7,400 + X2-retrained 7,400 = **22,200 episodes**. If time is short, the minimum that
still answers the reviewer is X1 reference + ablation (1,400) and X2-frozen + X2-retrained reference + ablation
(2 × 1,400).

### Hypotheses and statistics (copied from fd_v1, not re-chosen)

- Registered keys per block: H1 calibrated − H\*, H3b calibrated − guarded; ablation contrasts nc-H, A2, A8, A7;
  network grid G1 / G2; learned incumbents R1 / R2. Same Holm families, paired seeds, sequence-cluster bootstrap for
  recall, same marks.
- Reporting rule: every registered contrast is reported whatever its sign. A reversal on new data is reported as a
  reversal. No re-tuning, no new arms, no dropped arms after the first outcome is seen.
- New data are reported as a separate section ("independent evaluation"); fd_v1 is relabelled "development /
  internal evaluation".

## 4. Cost (from measured fd_v1 times)

| Work | Size | Estimate |
|---|---|---|
| X1 download + overlap audit | ~0.3 GB | < 1 h |
| X1 profiling (≤ 1,580 frames, 6.4 s each) | | ≤ 2.8 h GPU |
| UAVDT download + audit | ~ several GB | 1–2 h |
| X2 profiling (≈ 1,500 frames) + fidelity on test (≈ 600 × 4.5 s) | | ≈ 3.5 h GPU |
| X2-retrained RL (9 runs, 2 in parallel) | | ≈ 13 h CPU |
| Evaluation, 22,200 episodes (14,400 took ≈ 4 h in fd_v1) | | ≈ 6–8 h CPU |
| Analysis, tables, report | | ≈ 1 day |
| **Total** | | **≈ 2–3 days wall clock**, mostly unattended (detached WMI jobs) |

Storage: profiles and delivery logs of fd_v1 size scale, a few GB; images stay out of git.

## 5. Order and decision points

1. Review this proposal (fix the design and the seed ranges; write the pre-registration file).
2. Download DET test-challenge + overlap audit → go / secondary decision for X1 by the rule in §3.
3. Obtain UAVDT; verify terms, split, classes → go / fallback to DroneVehicle.
4. Register (commit) the pre-registration **before** any profiling of the new data.
5. Profiling → X2 training → evaluation → one analysis pass → report.
6. Only then revise the manuscript.

## 6. Manuscript

Presentation work stops except Fig. 1, bibliography and model formulation. Abstract, results, discussion and
conclusion wait for the new blocks. Even without new data, the text should already say that F is method-unseen
and give the corrected TD history (146/582 frames, 49/61 sequences).
