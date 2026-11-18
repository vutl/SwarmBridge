# A-T1 protocol (pre-registered): task-utility headroom in the network simulator

Written 2026-11-18 (review #45, step B), **before** the test stage. The train-pool tuning stage was started at the
same time; it uses only train seeds 0..29 and decides nothing outside the selection rules below.

**Code:**
- `at1_task.py`: utility tables, rules, task MPC.
- `at1_run.py`: stages tune / test.
- `ax2_eval.py`: `t=` / `tc=` rules, `+tmpc=SRC`, `DELIV_EXT`.
- `test_at1.py`: 5 tests.

Results go to `runs/uav_semcom/at1/`.

## What already exists (not repeated)

- **A6-L1 (09–10/11):** a task reward (TP − 0,5·FP) replacing the PSNR term changed the chosen mode to a cheaper fixed
  mode. It had no state-dependent selection and did not raise recall.
- **G-L2 (10–12/11):** the same per-image oracle question, **offline**: one frame, one mode, no network.
  - Oracle headroom over the strongest content-blind rule: +2,29 / +1,84 / +1,12 recall points at three bit levels.
  - Cross-checked with a flipped-image re-measurement: **+0,95 / +0,21 / −0,40**.
  - Causal predictors (image statistics, a light on-board detector) did not beat a payload-aware rule.
- **The question here:** does the network change this picture? Queues, deadlines, LA and five UAVs sharing two channels
  might let per-image utility knowledge decide which frames to protect, degrade or let drop. Answering it needs the
  oracle inside the simulator.

## Question

With channel information kept causal (the same online-calibrated rate model as the best existing stack), how much
full-workload detector recall is gained by additionally knowing each image's true per-mode detector utility?

## Environment (fixed)

- K5_la, scheduler-aware LA, persistent interference, no SRS, K5 observation.
- Scheduler and movement are frozen by A-X9 stage 1: `JA:stat:R:a1:m3 @ pacedB3`.
- Same power, deadlines and Qmin draw as every earlier LA study.

## Metric

**Full-workload on-time recall** = Σ TP of frames delivered by their deadline / Σ GT objects of every captured frame.
- A late or dropped frame counts 0 TP.
- TP is from the A5 detector (YOLO11s, conf ≥ 0,25, IoU ≥ 0,5): `runs/uav_semcom/a6l1/fidelity`. This file is
  identical to the A5 file on common rows and complete for the generative modes.
- Also reported:
  - Qmin-compliant recall (on time **and** PSNR ≥ Qmin; the definition used in A-X9);
  - frame on-time rate, drops, delivered Mbit, energy, A5 return, decision latency.

## Utility sources

| Source | What the controller knows about the current image | Deployable |
|---|---|---|
| mean | μ_m = mean TP of mode m over the TRAIN images | yes |
| gt | TP_i(m) of this image | **no** (diagnostic oracle) |
| flip | TP_i(m) from the flipped-image re-measurement (G-L2 0.4) | no |

- The flip measurement is new for the test images (`runs/uav_semcom/at1/flip`, same script `g2_flip.py`, `--splits
  test`).
- The three caption-generative modes have no flip measurement; they fall back to the original measurement.

## Controllers

All use the same scheduler, movement and channel information:
- fixed modes;
- **RV** `rv:SRC:λ`: argmax_m U(m) − λ·b̂(m), with b̂ = probe-predicted payload (G-L2's payload-aware rule);
- **TB** `tb:SRC:τ`: backlog-budgeted rule, with value U(m) if predicted on time, else 0, on the calibrated rate;
- **TMPC**: the A-E8 calibrated selective MPC, unchanged (N 8, H 4, causal rollout, Z 1, 50 ms budget), with
  objective "Σ on-time TP" instead of the A5 utility.
  - Incumbent = F_task.
  - Queued, already-probed images use U; unprobed and future frames use μ.
- For reference: q3 (H\*) and q3 + calibrated MPC with the A5 objective (I\* + MPC, the best A-X9 stack).

## Tune stage (TRAIN pool, seeds 0..29; selection by train on-time recall)

- **F_task** = best of the 14 fixed modes.
- RV best λ ∈ {0, 2, 5, 10, 20, 40, 80}, separately for mean and gt.
- TB best τ ∈ {0,5, 1, 1,5, 2, 3}, separately for mean and gt.
- The TMPC (mean, gt) and q3 + MPC are also run on train, for the selection below.

## Selection rules (fixed now)

- **C\*** (strongest causal arm) = best train on-time recall among {q3, F_task, q3 + MPC, RV-mean, TB-mean,
  TMPC-mean}.
- **O\*** (primary oracle) = best train on-time recall among {RV-gt, TB-gt, TMPC-gt}.
- **O\*_flip** = the same controller selecting with the flip source.

## Test (TEST pool, fresh seeds 9200..9299; never used; run once)

Arms:
- causal: q3, F_task, q3 + MPC, RV-mean, TB-mean, TMPC-mean;
- oracle: RV-gt, TB-gt, TMPC-gt;
- cross-check: RV-flip, TB-flip, TMPC-flip.

Each arm is scored with both measurements (gt and flip). CIs are paired over the 100 environment seeds.

**Headroom:**
- **H** = recall_gt(O\*) − recall_gt(C\*). Optimistic: selection and scoring use the same detector measurement.
- **H_x** (cross-checked) = ½[recall_gt(O\*_flip) + recall_flip(O\*)] − ½[recall_gt(C\*) + recall_flip(C\*)].
  - It selects with one measurement and scores with the other, in both directions, as in G-L2.
  - It catches only the detector variation visible under a horizontal flip. It is a stability check, not an exact
    noise correction.

## Decision gate (fixed now)

**Go to step C** (a causal utility predictor) only if all hold:
1. H_x ≥ +1,0 recall point;
2. the 95 % CI lower bound of H_x over environment seeds > 0;
3. matched constraints: O\* frame on-time rate not lower than C\* by more than 1 point, and O\* energy not higher than
   C\* by more than 2 %.

**Otherwise: STOP the task-utility direction.** No predictor, no task-aware algorithm on this mode menu and detector.

## Secondary (reported, not part of the gate)

- Same-controller information value: TMPC-gt − TMPC-mean, RV-gt − RV-mean, TB-gt − TB-mean, with flip versions.
- Mode-selection distribution per arm.
- **Concentration:**
  - share of the summed positive per-frame TP gain (O\* vs C\*, same episode and frame) contributed by the top 10 % of
    frames;
  - gain by tercile of frame object count;
  - gain by tercile of episode congestion (C\*'s drop rate);
  - gain per test video.
- Qmin-compliant recall for all arms.

## Limits (known before running)

- One detector at one operating point, A5 mode menu (whole-image codecs + 3 generative modes), 128 test images
  re-sampled across episodes.
- This is independent multi-UAV offloading, not collaborative perception (no view fusion).
