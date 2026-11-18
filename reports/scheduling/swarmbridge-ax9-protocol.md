# A-X9 protocol (pre-registered): the final bounded test of the DRL hypothesis

Written 2026-11-18, **before** any A-X9 tuning, training or testing (review #44). Code: `ax9_run.py`, `a5_train.py`
(options `--cfg-extra`, `--slot-move`, `--slot-ja`), `ax2_eval.py` (`SCHED~MEMBER@MOVE`, `ppola`, `+mpcnc`),
`a9_mpc.py` (rate model for phy "la"), `test_ax9.py`. Results will go to `runs/uav_semcom/ax9/`.

## Hypothesis

A PPO mode-only policy trained **on the realistic LA PHY**, with the same channel scheduler, movement, power, mode
action space and reward as the evaluation, is a better incumbent for the calibrated selective MPC than the best tuned
heuristic incumbent.

## Environment (fixed)

- K5_la, scheduler-aware link adaptation, persistent interference.
- No SRS.
- `obs_la`:
  - The LA-trained PPO observes the causal link-adaptation features: latest delayed CQI report, report age, measured
    INR, OLLA offset, NACK rates over 10 / 50 blocks, NACK rate on shared blocks. It never sees hidden channel
    realisations or simulator parameters.
  - Heuristics and MPC use only their own causal inputs (statistical CSI, queues, and the online calibrator from
    observed throughput).
  - The Shannon-trained PPO keeps its K5 observation layout.
  - `obs_la` changes only the observation, not the dynamics.

## Stage 1: scheduling and movement (TRAIN pool, seeds 0..29)

- Grid: movement {mission, pacedB1.5, pacedB2, pacedB3} × scheduler {12 JA members (`ja_sched.family("stat")`),
  least-slack}.
- Each cell is run under 3 representative mode rules (fixed_mbt2018-mean_q3, fixed_avif_20, greedy+budget1).
- **Selection rule:** the (movement, JA member) with the best train return averaged over the 3 mode rules.
- It is compared against the A-X8 setting (JA:stat:NR:a1:m0 @ pacedB2) and least-slack at the same movement.
- **Frozen** from here on.

## Stage 2: heuristic incumbents (TRAIN pool, seeds 0..29, frozen scheduler and movement)

- 22 mode rules (14 fixed, greedy, greedy+budget τ ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 3}) plus the 8 rate-based rules
  with the MPC's online calibrated rate.
- **Best heuristic H\*** = best train return.
- The 5 best rules are each run as the incumbent of the calibrated selective MPC.
- **MPC incumbent I\*** = best train return with MPC. It is not assumed to equal H\*.

## Stage 3: PPO trained on LA

- `a5_train.py`, mode-only.
- 3 seeds (0, 1, 2), 2M decision steps each (the A5 budget).
- A5 hyper-parameters: TimeMaskablePPO, MLP [256, 256], 6 envs.
- Slot decisions by the frozen JA scheduler and movement.
- **Checkpoint selection:** every 200k steps on the VAL pool (seeds 3000..3019); best validation return, ties to the
  earlier one.
- No test data is used.

## Stage 4: test (TEST pool, fresh seeds 8300..8399, never used before; run once)

All arms use the frozen scheduler and movement:

| Arm | Content |
|---|---|
| H\* | best tuned heuristic |
| I\* | the MPC incumbent alone |
| I\* + MPC | calibrated selective MPC (A-E8: N 8, H 4, causal rollout, Z 1, 50 ms budget) |
| I\* + MPC, no calibration | same search, budget and incumbent; nominal rate model `env.link_rate`, no bias-derived scales; incumbent rule uncalibrated |
| PPO-LA | 3 checkpoints |
| PPO-LA + calibrated MPC | 3 checkpoints |
| PPO-Shannon | the existing K5 checkpoints; distribution-shift reference |
| PPO-Shannon + calibrated MPC | the existing K5 checkpoints; distribution-shift reference |

**Reported:**
- return and paired 95 % CIs over the 100 environment seeds;
- every PPO training seed separately, plus a t-interval over the 3 training seeds;
- detector recall on delivered images, with paired CIs over environment seeds;
- deadline success, drops, goodput, energy, Jain, worst UAV;
- latency: slot, mode, integrated, share over 50 ms.

## Decision gate (fixed now)

Primary contrast: **Δ = (PPO-LA + MPC) − (I\* + MPC)**, return, averaged over the 3 training seeds.

**DRL GO only if all hold:**
1. mean Δ ≥ +2.0;
2. the lower bound of the 95 % CI over environment seeds is > 0;
3. Δ > 0 for each of the 3 training seeds;
4. the detector-recall difference is not significantly negative (upper bound of its 95 % CI ≥ 0);
5. integrated decision latency p99 ≤ 50 ms.

**Otherwise: STOP the DRL line.**
- No SAC / DSAC / autoregressive PPO / MAPPO to rescue the DRL framing.
- Paper A is reassessed as a model-based resource-control study (causal joint scheduling, mission pacing,
  deadline-bounded selective planning).
- A genuinely new coordination mechanism, with downstream-task value, has to be identified before any Method rewrite.

## Secondary (reported, not part of the gate)

- Calibration vs planning: (I\* + MPC) − (I\* + MPC without calibration).
- PPO-LA − H\*.
- (PPO-LA + MPC) − (PPO-Shannon + MPC): distribution-shift effect.
- Recall of every arm against H\*.
