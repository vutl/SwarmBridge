# A-T1 step C protocol (pre-registered): causal object-value predictor + Qmin diagnostic

Written 2026-11-18 (review #46), **before** any step-C run. Nothing below has been run on the val pool or on the
test-dev pool when this file is committed.

**Code (committed with this file):**
- `atc_predict.py`: object-count predictors.
- `atc_run.py`: stages dev / test.
- `atc_summary.py`: the gate, guardrails, C0, AP50 and concentration.
- `at1_task.py`: new sources `pA`/`pB`/`pC`, the `rvq` rule, test-dev tables, FP.
- `env.py`: `info_extra_s`, which charges the predictor's on-board compute.
- `test_atc.py`: tests.

Results go to `runs/uav_semcom/atc/`.

## What this step answers

A-T1 (review #45) showed in-network headroom (cross-checked +1,71 recall points). An exploratory oracle that knows
only the frame's **object count** reached +1,53.
- **C1:** can a causal on-board estimate of the object count recover that headroom, in the **detector-only** problem?
- **C0:** does the headroom survive when the Qmin PSNR floor is a constraint inside the action set?

**Problem definition (review #46, decision 1).**
- **Detector-only task-oriented offloading:** the objective is on-time detector recall, and PSNR is a secondary metric,
  not a constraint.
- This is a **different problem formulation** from A5 (whose utility penalises Qmin violations). No result here is
  reported as an improvement of the A5 problem.
- Qmin is kept as the separate quality-sensitive scenario C0.

## Data independence

| Pool | Source | Frames / videos | Use in step C |
|---|---|---|---|
| train | VisDrone2019-DET-val | 304 / 32 | fit predictors (labels allowed), tune λ / δ |
| val | VisDrone2019-DET-val | 84 / 14 | choose the predictor (labels used for selection only) |
| **test-dev** | **VisDrone2019-DET-test-dev** | **403 / 49** | **final test, run once** |

- The pools are split by video sequence. The code asserts that no sequence and no image hash (manifest sha256) is
  shared across pools.
- The detector (YOLO11s) was trained on VisDrone2019-DET-train, which is none of these pools.
- **Test-dev history (declared):**
  - Its frames were evaluation-only in A5 / A6-L1 / G-L2. Their per-image detector outcomes exist
    (`runs/uav_semcom/a6l1/fidelity/per_image_testdev.jsonl`) and so do their G-L2 features.
  - No step-C choice (feature set, model, λ, δ, predictor selection) looks at test-dev labels or outcomes.
  - The A-T1 test pool (128 frames, seeds 9200..9299) is **not** used again.
- **Dev peek (declared):** while writing this protocol, an unregularised fit of log(1 + n_gt) on log(probe bits) was
  checked on train → val.
  - Spearman 0,15 with probe bits alone; −0,02 when log pixels were added.
  - pA below keeps both features, as planned (the ridge may shrink the pixel term).
  - The val selection of pA is therefore not blind. No other val number has been seen.

## Environment (fixed, as A-T1)

- K5_la, scheduler-aware LA, persistent interference, no SRS, K5 observation.
- Scheduler and movement frozen by A-X9 stage 1: `JA:stat:R:a1:m3 @ pacedB3`.
- Same deadlines and Qmin draw (U(22, 30) dB).
- The env table is the A5 table **with the test-dev profiles** (`a5_eval.init(True)`). Encoder latencies are the val
  means and the probe fit is train-only, so train / val episodes are unchanged by adding the test-dev rows.

## Metrics

**Primary (C1):** full-workload object-pooled on-time recall = Σ TP of frames delivered by their deadline (any PSNR) /
Σ GT objects of all captured frames.
- Late or dropped frames contribute 0 TP.
- TP is from the A5 detector at conf ≥ 0,25, IoU ≥ 0,5 (L1 measurement).

**Secondary and guardrails** (all over the full workload):
- **Macro per-frame recall:** mean over captured frames of TP_on-time / n_gt. Every manifest frame has n_gt > 0, so
  there are no empty frames to handle separately.
- **FP per captured frame:** n_det − TP of on-time delivered frames, at conf ≥ 0,25; late or dropped frames have no
  detections.
- **Workload AP50:** COCO AP50 (IoU 0,5, maxDets 500, 101 points, mean over classes).
  - Each captured frame of each episode is one evaluation image; late or dropped frames have no detections.
  - It uses the stored detections and `a6l1_ap.py` (hashes in `runs/uav_semcom/a6l1/fidelity/sha256.txt`).
  - Reported as one value pooled over all test episodes, plus the mean over episodes.
- On-time frame rate, drops, delivered Mbit, energy.
- **Integrated decision latency p99** (`ms_int_max_p99`).
- **Predictor compute:** on-board milliseconds per frame, charged as extra delay before the mode decision
  (`info_extra_s`). UAV compute energy is not modelled, so it is reported as CPU-ms per frame.
- **Qmin-compliant recall:** on time **and** PSNR ≥ Qmin; this is C0's primary metric.

## Object-count predictors (causal: on-board information at decision time; labels only for fitting)

- **Target:** log(1 + n_gt).
- **Model:**
  - ridge on standardised features, as G-L2;
  - α ∈ {0,1, 1, 10, 100, 1000} chosen by 5-fold GroupKFold over train sequences;
  - fitted on the train pool only.
- **Train-pool predictions** (used to tune λ) are **out-of-fold**: GroupKFold(5) by sequence, with α chosen inside
  each fold's training part.
- **Val and test-dev predictions** use the model fitted on all train frames.

| Set | Features | On-board cost charged |
|---|---|---|
| **pA** (payload) | log probe payload (AVIF-5, already encoded and charged in every arm), log pixels | 0 |
| **pB** (+ image statistics) | pA + the 13 G-L2 B statistics (grey frame at width 480) | +10 ms (G-L2: 9,6 ms median, CPU, 1 thread) |
| **pC** (+ light detector) | pA + the 14 G-L2 C detector features (A5 detector at imgsz 480 on the original frame) | +35 ms (G-L2: 35,1 ms median, CPU, 1 thread) |

pC uses the **same weights as the ground-station detector**, which is optimistic. If pC is selected, this is stated
with the result.

**Utility used by the controller:** U_i(m) = n̂_i / mean_train(n_gt) · μ_m.
- This is the same form as the exploratory count oracle (`count`: n̂ = n_gt).
- n̂ = max(exp(pred) − 1, 0).

## Controllers (same scheduler, movement, channel information)

- 14 fixed modes.
- **q3 + MPC:** the A5-objective calibrated selective MPC, the best A-X9 stack.
- **RV** `rv:SRC:λ`: argmax_m U(m) − λ·b̂(m), with SRC ∈ {mean, pA, pB, pC, count, gt}.
  - count and gt are non-deployable oracles.
  - λ ∈ {0, 2, 5, 10, 20, 40, 80}.
- **RVQ** `rvq:SRC:λ:δ` (C0): the same rule, but only modes with probe-predicted PSNR q̂(m) ≥ Qmin + δ are allowed.
  - If none qualifies, it takes the highest-q̂ mode.
  - δ ∈ {0, 1, 2} dB.
  - The PSNR information is causal and the same in every arm.

TB rules and the task MPC are not carried over: they lost to RV and to q3 + MPC on train in A-T1.

## Dev stage (no test-dev)

**D1, fit (train pool only).**
- Fit pA, pB, pC. Write n̂ for every manifest frame: out-of-fold on train, full fit elsewhere.
- Report offline metrics on **val only**: Spearman ρ(n̂, n_gt), RMSE of log(1 + n̂).

**D2, tune** (train pool, seeds 0..29). Every RV / RVQ member and the 14 fixed modes are run, plus q3 + MPC.
- **C\*** = best train on-time recall among {14 fixed, q3 + MPC, RV-mean (best λ)}.
- For SRC ∈ {pA, pB, pC, count, gt}: best λ by train on-time recall.
- **Q-C\*** = best train **Qmin-compliant** recall among {14 fixed, q3 + MPC, RVQ-mean (best λ, δ)}.
- For SRC ∈ {pA, pB, pC, count, gt}: best (λ, δ) by train Qmin-compliant recall.

**D3, select** (val pool, seeds 3000..3019), detector-only. Run C\*, RV-pA, RV-pB, RV-pC and RV-count with the
train-chosen λ. The predictor P\* is chosen progressively (review #46: payload first):
1. P\* = pA.
2. pB replaces P\* only if **both** hold:
   - val recall(RV-pB) ≥ val recall(RV-P\*) + 0,3 points;
   - val RMSE(pB) ≤ 0,9 × val RMSE(pA).
3. pC is considered only if:
   - the current P\* recovers < 50 % of the val count-oracle headroom, (RV-P\* − C\*) / (RV-count − C\*);
   - **and** that headroom is ≥ 1,0 point.

   It then replaces P\* under the same two conditions (+0,3 points over the current P\*; RMSE ≤ 0,9 × that of the
   current P\*). Its 35 ms cost is within the 50 ms per-frame budget.

**Freeze.** `dev.json` records C\*, Q-C\*, every chosen λ / δ and P\*. `dev.json` is committed before the test
stage.

## Test (test-dev pool, fresh seeds 9600..9699; never used; run once)

Arms:
- C1: q3, F_task (best fixed by train recall), q3 + MPC, RV-mean, RV-pA, RV-pB, RV-pC, RV-count (oracle), RV-gt
  (oracle).
- C0: Q-C\*, RVQ-mean, RVQ-pA / pB / pC, RVQ-count, RVQ-gt.

All arms are run (not only P\*), so the progressive choice can be audited. Only **P\*** enters the gate.

**Uncertainty:**
- **Primary:** two-way bootstrap over environment seeds **and** test-dev videos (10 000 draws; sequences and episodes
  resampled independently; pooled ratio recomputed).
- Also: paired t CI over environment seeds.

## C1 decision gate (fixed now)

Let ΔR = recall(RV-P\*) − recall(C\*) and H_cnt = recall(RV-count) − recall(C\*), both on test-dev.

**GO** (evidence that causal task-value estimation improves constrained network allocation in this simulator) only
if all hold:
1. ΔR ≥ +1,0 point;
2. the two-way bootstrap 95 % CI lower bound of ΔR > 0;
3. ΔR ≥ 0,5 · H_cnt (at least half of the count-oracle headroom, measured on the same test);
4. macro per-frame recall of RV-P\* not lower than C\* by more than 1,0 point;
5. FP per captured frame not higher than C\* by more than 10 %;
6. frame on-time rate not lower than C\* by more than 1 point;
7. energy not higher than C\* by more than 2 %;
8. integrated decision latency p99 ≤ 50 ms.

**Otherwise: STOP the task-value direction.** No larger detector, no more complex learner.

**Simpler-rule clause:**
- RV-pA − C\* and RV-P\* − RV-pA are reported with the two-way CI.
- If P\* ≠ pA and the lower bound of RV-P\* − RV-pA is ≤ 0, the claim is only that **payload suffices**. No claim is
  made that learned image features are necessary.

## C0 (Qmin diagnostic; no gate; does not change A-T1)

- **Reported:** H_Q(SRC) = Qmin-recall(RVQ-SRC) − Qmin-recall(Q-C\*) for SRC ∈ {count, P\*, gt}, with the two-way CI.
  The detector-only recall of the same arms is reported too.
- **Reading, fixed now:**
  - If H_Q(count) has a 95 % CI lower bound > 0 **and** is ≥ half of H_cnt, the two formulations are close: count
    knowledge also helps under the quality floor.
  - Otherwise they are materially different problems, and the detector-only headroom depends on dropping the floor.

## Secondary (reported)

- Mode distribution per arm.
- Offline count accuracy on test-dev (after the test).
- **Concentration of RV-P\* − C\*:** by video, by tercile of n_gt, by tercile of C\* congestion (drop rate).
- AP50 of all arms.

## Known limits

- One detector at one operating point; A5 mode menu.
- Frames are drawn i.i.d. from the pool (no temporal information; temporal predictors are not allowed here).
- 403 test-dev frames re-sampled across episodes; content generalisation rests on 49 test-dev videos.
- Independent multi-UAV offloading, not collaborative perception.
- **Novelty limit (review #46):** a positive result is evidence about task-value-aware airtime allocation, not an
  algorithmic contribution. Per-image value estimation from local detections already exists (ORIC, SEC 2024).
