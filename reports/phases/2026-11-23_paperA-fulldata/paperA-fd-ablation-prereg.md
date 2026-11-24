# Paper A extended evaluation: mechanism ablation (Phase 2b) pre-registration (24/11)

**Status:** submitted for review. No test seed has been simulated; only the rl_val smoke run (validation) below exists.

**Motivation:** these hypotheses were **motivated by the Phase 2 results**. The matched nominal design recovered about
18.7 points over the unguarded nominal planner, and the calibrated predictor added nothing on TD. They are therefore
tested on **fresh seeds** and kept separate from the Phase 2 benchmark.

## 1. Setting (identical to Phase 2 unless stated)

| Item | Value |
|---|---|
| Simulator | fd_v1 table (fd_common, hash-verified); K = 5, M = 2, 1 MHz, nominal LA; JA scheduler on pacedB3 |
| Incumbent | H\* = fixed_avif_20 (Phase 2 tune) |
| Pools / seeds | **TD 24000–24099 (primary)**; V 25000–25099; F 26000–26099. All new seed ranges |
| Code | `fd_next.py ablation`; arms in `ax2_eval.py` (commit given with the submission) |

## 2. Arms (all use the nominal LA rate model, except the reference arm)

| Arm | Suffix | Tracker gate (≥ 20 s of data) | Band / robustness check | Lookahead H | Cold start |
|---|---|---|---|---|---|
| H\* | — | — | — | — | — |
| Unguarded nominal | `+mpcnc` | no | none | 4 | plans from the first decision |
| **Gate only** | `+mpcngg` | yes | band fixed to [1, 1] | 4 | keeps the incumbent until data |
| **Band only** | `+mpcngb` | no | tracker band once data exist | 4 | **plans unguarded, band [1, 1], until the tracker has data**; logged as `cold_unguarded` |
| Gate + band (matched nominal) | `+mpcng` | yes | yes | 4 | keeps the incumbent until data |
| **Gate + band, H = 1** | `+mpcngh1` | yes | yes | **1** (screen and full plan) | keeps the incumbent until data |
| Calibrated (reference) | `+mpc` | yes | yes | 4 | keeps the incumbent until data |

All other parameters are as in Phase 2: N 8, screen at [1, ℓ], two candidates, switch if gain > 1 s.e., 50 ms budget.

## 3. Hypotheses (episode return; paired t over 100 seeds; TD primary)

| ID | Contrast | Question | Family |
|---|---|---|---|
| **A1** | gate only − unguarded | is cold-start gating useful on its own? | **confirmatory** |
| **A2** | band only − unguarded | is the uncertainty-band check useful on its own? | **confirmatory** |
| **A4** | gate + band (H = 4) − gate + band (H = 1) | does multi-step lookahead add value under the same error-aware design? | **confirmatory** |
| A3 | gate + band − unguarded | the combined design on fresh seeds (replicates the post-hoc Phase 2 contrast) | supporting |
| A5 | gate + band − gate only | the band given the gate | supporting |
| A6 | gate + band − band only | the gate given the band | supporting |
| A7 | calibrated − gate + band (H = 4) | replicates H3b on fresh seeds | supporting |
| A8 | gate + band − H\* | the matched design against the incumbent | supporting |

**Multiplicity:** Holm over {A1, A2, A4} on TD at family-wise α = 0.05, two-sided. Supporting contrasts get unadjusted
95% CIs and no confirmatory statement. V and F are reported alike as replications.

**Wording:**
- **A1 / A2:** "gating (band) alone improved / reduced return" only if Holm-rejected, with the sign of the effect.
  Otherwise "no detected effect".
- **A4:**
  - rejected with a positive sign: "multi-step lookahead adds value";
  - not rejected: "no detected benefit of the 4-image lookahead over a 1-image controller with the same error-aware
    design". The paper must then justify the planner by other means or simplify it;
  - rejected with a negative sign: "the 1-image controller is better".
- **Never** "equivalent" from a non-rejection.

**Recall (Qmin-compliant):**
- the same contrasts at seed level, paired t-CI;
- the video-cluster bootstrap, as in Phase 2, for A1, A2 and A4 (supplementary; conditional on the simulated outcomes).

**Mechanism and latency, per arm:**
- planned / gated / cold-unguarded decisions; switches; aborts;
- robustness checks and rejects; band and bias quantiles;
- **per-slot latency:**
  - p99 of max (parallel assumed) and of sum (sequential);
  - p99 of full-sum, which **includes the calibrator / tracker update** (newly instrumented);
  - update overhead p99;
  - the unrounded share of slots over 50 ms, for both max and full-sum, and the number of episodes with any.

**Integrity asserts:** as in Phase 2 (260 tasks; identical images and GT across arms per seed; pool indexing).

## 4. Validation smoke (rl_val seeds 3000–3004; not interpreted)

| Arm | Return | Gated / cold-unguarded | Switches | Rejects / checks | p99 max / sum / full-sum | Update p99 |
|---|---|---|---|---|---|---|
| H\* | −72.4 | — | — | — | 8.5 / 8.5 / 8.5 ms | 0 |
| +mpcnc | −66.8 | 0 / 0 | 188.8 | 0 / 189 | 44.7 / 163.6 / 163.6 ms | 0 |
| +mpcngg | −77.9 | 48.2 / 0 | 158.2 | 0 / 158 | 44.0 / 141.2 / 141.7 ms | 0.75 ms |
| +mpcngb | −69.4 | 0 / 47.0 | 109.6 | 84 / 193 | 41.8 / 161.1 / 161.5 ms | 0.56 ms |
| +mpcng | −68.6 | 48.2 / 0 | 73.2 | 80 / 154 | 40.8 / 137.0 / 137.4 ms | 0.56 ms |
| +mpcngh1 | −76.0 | 48.2 / 0 | 86.0 | 90 / 176 | 29.4 / 102.9 / 103.3 ms | 0.55 ms |
| +mpc | −64.0 | 48.4 / 0 | 87.4 | 74 / 161 | 40.7 / 137.8 / 138.0 ms | 0.37 ms |

- H\*, +mpc, +mpcnc and +mpcng give exactly the Phase 2 smoke values, so the new instrumentation does not change
  behaviour.
- The update overhead is below 1 ms.

## 5. Run

- After approval: `runs/uav_semcom/paperA_fd/ablation/AUTHORIZED`, then `fd_next.py ablation test`: 7 arms × 3 pools × 100 seeds = 2,100 episodes,
  about 40 min, detached, with nothing else on the machine.
- The analysis script follows the Phase 2 summary pattern. It is committed before the run.
