# Paper A extended evaluation: Phase 3 (learned incumbents) pre-registration (24/11)

**Status:** submitted for review. Only validation-pool smoke runs exist; no test seed has been simulated and no fd_v1
training run has started.

**Question (revised after Phase 2):**
1. Does the error-aware selective MPC layer improve incumbents from different policy families?
2. Does the ACK-calibrated empirical rate model add anything on top of it?
3. Are the learned incumbents better than the strong heuristic? This is asked as is, with no requirement that RL win.

**Scope:** E7 and this phase study **mode-selection** RL only. They say nothing about joint trajectory or resource
DRL, such as Huang et al.'s DSAC.

## 1. Incumbents and training

| Incumbent | Source | Training on fd_v1 |
|---|---|---|
| H\* | fixed_avif_20 | — (the Phase 2 rows are reused: same protocol and seeds) |
| PPO-LA, zero-shot | A-X9 checkpoints (trained on the 304 A5 frames) | none |
| QR-DQN / Discrete SAC, zero-shot | E7 checkpoints | none |
| **PPO-LA retrained (`ppofd`)** | `a5_train.py --fd`, exactly the A-X9 command otherwise | 3 seeds (0, 1, 2) |
| **QR-DQN retrained (`qrfd`)** | `offpolicy_train.py --fd --algo qrdqn --lr 1e-4` (E7 choice) | 3 seeds |
| **Discrete SAC retrained (`dsfd`)** | `offpolicy_train.py --fd --algo dsac --ent-ratio 0.3` (E7 choice) | 3 seeds |

**The A-X9 PPO command:**
- `--regime K5_la --rl-part mode --heur fixed_avif_20 --steps 2e6`;
- cfg-extra: scheduler-aware LA, persistent interference, LA observation features;
- frozen JA member on pacedB3.

**Shared training settings:**
- **Budget and selection:** 2M decisions per run. Every 200k decisions a checkpoint is saved and validated on
  **rl_val** (seeds 3000–3019). The selected checkpoint is the one with the best mean validation return; ties go to the
  earlier checkpoint.
- **No hyper-parameter search.** All values are frozen at the E7 / A-X9 choices.
- **Compatibility:** the observation (464-dimensional with the LA features) and the action space depend on K and on the
  LA feature set, not on the image pool. The old checkpoints therefore run unchanged on fd_v1. The mode-info features
  use the fd_v1 probe predictor (fitted on rl_train), as for every arm.
- **Compute:**
  - PPO: about 1 h per run (A-X9);
  - QR-DQN: about 3.9 h; Discrete SAC: about 3.4 h (E7);
  - two runs in parallel, detached, resumable queue: **about 13 h of CPU** in total;
  - weights are not committed.

## 2. Evaluation

**Arms** for each learned incumbent X:
- X alone;
- X + `+mpcnc` (unguarded nominal);
- X + `+mpcng` (error-aware design, nominal model);
- X + `+mpc` (error-aware design, calibrated model).

Learned arms run each of their 3 training-seed checkpoints.

**Pools / seeds:**

| Part | Pools and seeds | Episodes |
|---|---|---|
| Zero-shot (old checkpoints, supporting) | TD 20000–20099 | 3 × 4 × 3 × 100 = 3,600 |
| Retrained (main) | TD 20000–20099 (primary), F 22000–22099, V 21000–21099, TD_all 20000–20099 (the Phase 2 seeds, so the H\* rows pair) | 3 × 4 × 3 × 4 × 100 = 14,400, about 4 h CPU |

## 3. Hypotheses (retrained incumbents, TD primary)

**Statistics:** paired difference per environment seed, after averaging the 3 training-seed checkpoints per environment
seed (as in E7); 95% t-CI over the 100 environment seeds.

**This CI measures environment variability only.** Training-seed variability is reported separately:
- each training seed's contrast, with its own CI;
- the spread of the 3 per-seed means;
- with 3 training seeds no inferential claim about training-seed variance is made.

| ID | Contrast, for X ∈ {PPO-LA, QR-DQN, Discrete SAC} | Family |
|---|---|---|
| **R1_X** | X + error-aware design (nominal model) − X | **confirmatory** |
| **R2_X** | X + calibrated − X + error-aware design (nominal model) | **confirmatory** |
| R3_X | X + calibrated − X | supporting (E7-comparable B1) |
| R4_X | X + error-aware design − X + unguarded nominal | supporting |
| R5_X | X − H\*, and X + error-aware design − H\* + error-aware design | supporting (is RL better than the heuristic?) |

- **Multiplicity:** Holm over the 6 confirmatory tests {R1, R2} × 3 families on TD, family-wise α = 0.05, two-sided.
- **Wording:**
  - "The error-aware layer improved incumbents from k of 3 learned families", counting only Holm-rejected positive R1s.
  - The same rule applies to R2 for the calibrated predictor.
  - Non-rejections are "no detected difference".
  - F, V and TD_all are replications; they never upgrade a TD statement.
  - **RL versus heuristic** (R5) is reported descriptively, whatever its direction. The method does not need RL.
  - Zero-shot results are supporting only. They answer "do the old checkpoints transfer to the new images?".

**Recall:** seed-level paired CIs for R1, R2 and R5, plus the video-cluster bootstrap (supplementary), as in Phase 2.
Return and recall are reported separately; no task-level superiority is claimed from return.

**Latency:**
- max, sum and full-sum p99 (full-sum includes the tracker / calibrator update);
- update overhead;
- the unrounded share of slots over 50 ms;
- incumbent inference time (PPO about 3 ms, MLPs below 1 ms; E7).

## 4. Validation smoke (zero-shot old checkpoints on rl_val, 2 seeds)

See the submission reply; the numbers are not interpreted.

## 5. Run order after approval

1. Zero-shot on TD.
2. The training queue (`fd_phase3_queue.py`; 9 runs; `phase3/ckpts.json` and `selection.json` written from the
   validation logs only).
3. The retrained evaluation.
4. The pre-committed summary script.

Everything runs detached, with resume, and `phase3/AUTHORIZED` is required.

## 6. Amendment after the review of 1c8023c4 — the FROZEN version

1. **Zero-shot wording:** the old checkpoints are frozen, but their mode-info features use the fd_v1 probe predictor,
   fitted on rl_train. This is *frozen-policy transfer under an updated image workload and observation estimator*,
   not an unchanged old pipeline.
2. **Training budget:**
   - `a5_train.py` now logs `num_timesteps` (actual) and `n_updates` at every checkpoint, next to the nominal
     200k-step counter.
   - PPO rollout collection in blocks of n_steps × n_env can exceed the nominal interval. The A-X9 procedure itself is
     unchanged.
   - QR-DQN and Discrete SAC log the transitions and gradient updates they already counted.
3. **Checkpoint integrity:**
   - `fd_phase3_queue.py` writes `phase3/selection.json` with, per run: the selected checkpoint path and **sha256**,
     the exact command, the seed, the nominal and actual steps, the updates, the training wall time, the table tag, and
     the dependency versions.
   - Copies of every train.json (validation logs) are kept.
   - `fd_next.py rl retrained` verifies every checkpoint's sha256 before evaluating.
   - Interrupted runs are moved aside, kept identifiable, and restarted from scratch; their optimiser, replay and RNG
     states are not restored.
4. **Statistical scope:**
   - The environment-seed CIs, computed after averaging the 3 checkpoints, are conditional on those trained policies.
   - Every training-seed contrast (R1, R2 per seed) is reported.
   - The 3 × 100 evaluation episodes are not 300 independent training replicates.
5. **Claims:**
   - R1 and R2 are central. R5 (versus the heuristic) is reported whatever its direction.
   - Nothing is claimed about joint trajectory / resource DRL or general RL superiority.
6. **Analysis:** `fd_next_summary.py rl --name zeroshot | retrained`, committed before the runs and exercised on
   synthetic data. R5 uses the Phase 2 H\* rows, which have the same seeds.

## 7. Execution after approval (all three phases, review of 1c8023c4)

**Order:** ablation → Phase 4 → Phase 3 (zero-shot, then training, then the retrained evaluation).

**One detached chain** (`run_next.sh`):
- the steps run sequentially; RL training never overlaps an MPC benchmark;
- the chain stops at the first failure;
- each step needs its `<phase>/AUTHORIZED`, created only after this frozen amendment, the analysis code, the tests and
  the smokes are committed.

If a correctness bug appears: the affected run is stopped, the evidence preserved, and the deviation documented
before anything continues.
