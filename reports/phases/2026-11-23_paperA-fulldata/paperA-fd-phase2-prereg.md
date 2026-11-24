# Paper A extended evaluation: Phase 2 pre-registration (24/11)

**Status:** submitted for review. **No test pool (TD, F, V, TD_all) has been simulated.**

**What has been run on the training and validation pools only:**
- the H\* tuning on rl_train (seeds 0–29);
- a 5-seed smoke run on rl_val (seeds 3000–3004).

The test command refuses to run until `runs/uav_semcom/paperA_fd/phase2/AUTHORIZED` exists. That file is created only
after this document is approved.

**Code:**
- `fd_common.py`: fd_v1 table, pools, recall table, hash verification;
- `fd_phase2.py`: tune, smoke, test;
- `ax2_eval.py`: new arms `+mpcng` and `+mpcrhnb`, plus logging of the band and of the robustness check.

The commit hash is given in the reply that submits this file. The code is frozen from that commit; any later change
voids this pre-registration.

## 1. Fixed artifacts (sha256, CRLF→LF normalised; `fd_common.verify()` checks them before every run)

| Artifact | sha256 |
|---|---|
| manifest_fd_v1.json | `f0760601515f9663f2b7747bd247eddabc3ab8c997755c73ae8bf2f2339d4d50` |
| profile/profile.jsonl (61,448 records) | `078e0e231dfeadaf9c9c84f8db2cb2a0d2c4166e4c3470cd35907d10f898781c` |
| profile/cpu_stratified_sample.json (fixed before timing, commit df9e7330) | `977290ed4450540b50f187f0c8147921016b880d77c69cf84cec2bed5b82cd07` |
| profile/cpu_learned_stratified.jsonl | `9ca0f392642ad9960f2195fba7d72b5dd50d2afce743c9ca5d73e97773772074` |
| profile/cpu_learned_stratified.meta.json | `ac063091b501516c7edb8f0cb1e596306012fc288d1fbb33b5005d92a2832341` |

## 2. Simulator inputs

**Latency table** (`fd_common.load`):
- encode and decode times are means over the rl_train rows;
- the MBT2018 encode time is the UAV-CPU stratified estimate, weighted by the training distribution (48 frames,
  6 threads);
- these are CPU timings measured on the stated hardware, not latencies validated on UAV hardware.

| Mode | Encode (s) | Note |
|---|---|---|
| mbt2018 q1 / q3 / q5 / q8 | **0.593 / 0.591 / 1.203 / 1.139** | stratified CPU; first-10 rule gave 0.374 / 0.386 / 0.735 / 0.718 (not used); the A5 table had 0.463 / 0.476 / 0.912 / 0.894 |
| avif 5 / 20 / 50 / 80 | 0.091 / 0.118 / 0.177 / 0.176 | CPU profile |
| jpeg2000 100 / 25 | 0.276 / 0.275 | CPU profile |
| cap_t2i, cap_thumb64_i2i_t0.6, cap_thumb128_i2i_t0.3 | 0.239 / 0.256 / 0.269 | caption on GPU (as A5) |
| thumb64_up | 0.017 | |

- Decode times are rl_train means of the base-station GPU or CPU decoders (Phase 1 report, section 2.3).
- The probe predictor is fitted on rl_train.

**Unchanged from E1–E7:**
- the 14 modes;
- utility and reward;
- LA PHY: `K5_la` with scheduler-aware LA and persistent interference;
- the JA scheduler and pacedB3 movement (frozen, `ax9/sched.json`);
- every MPC and calibration parameter: N 8, H 4, 50 ms budget, margin 1 s.e., 2 dB bins, 20 s minimum, band rule;
- the YOLO11s weights.

## 3. Pools and seeds

| Pool | Role | Frames / sequences | Seeds | Prior use |
|---|---|---|---|---|
| rl_train | H\* tuning | 1,523 / 141 | 0–29 | none |
| rl_val | smoke only (RL checkpoints in Phase 3) | 374 / 25 | 3000–3004 (smoke) | none |
| **TD** | **expanded primary evaluation** | 582 / 61 | **20000–20099** | 403 frames from 49 sequences used in E3, step C and E7 |
| F | method-unseen supplementary evaluation | 168 / 18 | 22000–22099 | used to select the YOLO checkpoint; bias not quantified |
| V | historical replication | 548 / 76 | 21000–21099 | used throughout E1–E7 development |
| TD_all | frame-weighted secondary | 1,610 / 61 | 23000–23099 | as TD |

**Episode construction:**
- each episode draws its images with replacement from the pool, with the environment seed;
- arms with the same seed see the same images, so the comparisons are paired.

## 4. Incumbent H\* (tuned on rl_train before this document; no test data involved)

- **Rule:** the best mean return over rl_train seeds 0–29 among the 22-rule family (`ax9_run.H_FAMILY`) used in E4/E6.
- **Result: H\* = fixed_avif_20.** rl_train return per rule:
  - fixed_avif_20 −71.2;
  - fixed_mbt2018-mean_q1 −75.8;
  - greedy+budget1 −79.6;
  - greedy+budget0.75 −80.2;
  - fixed_mbt2018-mean_q3 −87.4.
- **Difference from E1–E7** (where H\* = mbt2018 q3): consistent with the higher measured MBT2018 CPU encode time and
  the larger frames. It is reported, not tuned around.

## 5. Arms

| Arm | Suffix | Rate model | Gate (≥ 20 s of tracker data) | Band / robustness check | Role |
|---|---|---|---|---|---|
| H\* | — | — | — | — | incumbent |
| **Calibrated MPC** | `+mpc` | ACK-calibrated, SNR-binned empirical | yes | yes, from the tracker | **the method** |
| Nominal MPC | `+mpcnc` | LA mapping on statistical fading | no | none (scales [1, 1]) | existing baseline |
| **Nominal MPC, matched design** | `+mpcng` | nominal | yes, tracker on nominal predictions | yes, same rule | **isolates the rate model within a common control design** |
| RobustMPC-style | `+mpcrh` | harmonic mean of the last 5 rates / (1 + max relative error) | yes | yes, same rule | existing baseline |
| RobustMPC-style, no band | `+mpcrhnb` | as `+mpcrh` | yes | **band fixed to [1, 1]** | **proposed fairness variant, see section 6** |

**Common to all MPC arms:**
- the same planner: screen at scales [1, ℓ], two candidates, full plan with N 8 and H 4, switch if gain > 1 s.e. and
  robust at ℓ and h;
- the same 50 ms budget and the same incumbent.

**`+mpcng` caveat** (review of 50ce9022):
- it shares the gate and band **algorithm** with the calibrated arm, not their realised values;
- the nominal model's prediction errors differ, so its tracker yields different gates and bands. Smoke run: median ℓ/h
  0.50/1.15 against 0.67/1.25 for the calibrated arm;
- the H3b contrast therefore estimates the effect of *substituting the rate model within a common control design*, not
  a strictly isolated predictor effect.

## 6. Finding from the smoke run: the RobustMPC-style arm is disabled by the shared band

On rl_val (5 seeds, **validation only**):
- the RobustMPC-style predictor's built-in pessimism is read by the tracker as a large bias: median relative bias
  +1.2 to +1.4, so observed rates are about 2.2–2.4 times the predictions;
- the band's upper end explodes (median h ≈ 6,100; up to 2×10⁴);
- the robustness check then rejects almost every candidate switch: rejects ≈ checks; 1.6 switches per episode.

**E1–E7 show the same behaviour:**
- in E4 N5x2, `+mpcrh` switched 0.8 times per episode, against 45.6 for the calibrated arm.
- The E1–E7 statement "calibrated > RobustMPC-style in 11/12" therefore largely compares the calibrated MPC with an
  almost inactive baseline. This will be stated in the paper. The E1–E7 numbers are not changed.

**Proposed remedy: register a second variant, `+mpcrhnb`**, which keeps the gate and fixes the band to [1, 1].
- The original RobustMPC already encodes its caution in the predictor's discount, so it has no separate band.
- In the same smoke run, `+mpcrhnb` switched 163 times per episode and returned −117.8, against −72.4 for H\* alone
  (5 seeds; not interpreted).
- Both variants are reported. Nothing about them is tuned further. The reviewer may reject `+mpcrhnb`.

**Smoke run, all arms** (rl_val seeds 3000–3004; validation only; shown so the reviewer can check the mechanics; not
interpreted):

| Arm | Return | Planned / gated | Switches | Rejects / checks | Band median ℓ / h | Integrated p99 |
|---|---|---|---|---|---|---|
| H\* | −72.4 | — | — | — | — | 8.6 ms |
| +mpc | −62.2 | 211.6 / 48.4 | 89.2 | 69 / 158 | 0.67 / 1.25 | 42.4 ms |
| +mpcnc | −77.7 | 260 / 0 | 192.6 | 0 / 193 | — | 40.5 ms |
| +mpcng | −73.3 | 211.6 / 48.4 | 76.0 | 81 / 157 | 0.50 / 1.15 | 40.3 ms |
| +mpcrh | −73.1 | 211.8 / 48.2 | 1.6 | 115 / 116 | 0.05 / 6,101 | 42.4 ms |
| +mpcrhnb | −117.8 | 212 / 48 | 162.8 | — | 1.00 / 1.00 | 37.4 ms |

## 7. Hypotheses, estimands and decision rules

**Primary pool:** TD. F, V and TD_all are reported in the same way, as supplementary, historical and frame-weighted
analyses.

**Primary estimand:** episode return. Paired difference per environment seed; 95% t-CI over the 100 seeds.

| Hypothesis | Contrast | Question |
|---|---|---|
| H1 | cal − H\* | does the full calibrated selective-MPC stack improve the tuned incumbent? |
| H2 | cal − RobustMPC-style (band) | improvement over the existing E1–E7 baseline (interpret with section 6) |
| H2b | cal − RobustMPC-style (no band) | improvement over a RobustMPC-style variant that is actually active |
| H3 | cal − nominal | improvement over the nominal planner (the predictor and the gate/band differ) |
| **H3b** | **cal − nominal with matched design** | **benefit of the ACK-calibrated rate model within a common control design** |

**Wording rules** (on TD; F, V and TD_all can only qualify a statement, never upgrade it):

- **H3b (the key question):**
  - CI > 0: "the ACK-calibrated rate model improved return over a nominal model within the same control design".
    Add "replicated on F / V / TD_all" only for pools whose CI is also > 0; name the pools that disagree.
  - CI contains 0: "no detected benefit of calibration beyond the control design". The benefit attributable to the
    selective-MPC design is then reported through H1 and H3.
  - CI < 0: "calibration reduced return relative to the nominal model in the same design", followed by an analysis
    of why.
- **H1:** "improves the tuned incumbent" only if its CI > 0. Otherwise, the E4/E6 wording: condition-dependent, not
  shown here.
- **H2 / H2b:** "better than a RobustMPC-style baseline" only if both CIs > 0. If only H2 holds, the paper says the
  RobustMPC-style comparison is not informative in this design (section 6).
- **H3:** stated as a comparison of the full stack with the nominal planner, not as a calibration effect.
- **No pooled or meta-analytic verdict across pools.** No new arm, parameter or pool after any test run.

**Secondary estimand: Qmin-compliant recall.**
- **Definition:** objects found in images delivered on time with PSNR ≥ Qmin, divided by all ground-truth objects of the
  episode's images. Pools F, TD, TD_all and V only.

**Two analyses of recall:**
1. **Seed level (primary for recall):** per-episode recall, paired difference, 95% t-CI over 100 seeds.
2. **Video-cluster analysis (scene generalization of recall):**
   - **Unit:** the source sequence s.
   - For each arm, seed e and sequence s, TP_{e,s} and GT_{e,s} are summed over that episode's image occurrences from
     sequence s. GT_{e,s} is identical across arms, because the images are the same and every captured image ends in
     the finished list, including drops.
   - **Estimator:** Δ = Σ_e Σ_s (TP^A_{e,s} − TP^B_{e,s}) / Σ_e Σ_s GT_{e,s}. This is object-weighted; each sequence
     weighs by its objects among the sampled occurrences.
   - **CI:** percentile, from B = 2,000 bootstrap draws (seed 0) that resample the pool's sequences with replacement;
     each draw multiplies the terms of sequence s by its multiplicity.
   - **What it captures:** uncertainty about which scenes are in the pool, *conditional on the simulated delivery
     outcomes*. It does not resample the queue coupling between images of different sequences within an episode.
   - **Reported for:** H1, H3 and H3b.
3. **No scene-level claim about episode return** is made: episodes mix sequences and queue dynamics couple them. A
   scene-stratified design would be a separate, separately registered study.

**Other metrics, descriptive, per arm and pool:**
- return per image; on-time, drop, late and Qmin-violation rates;
- mean PSNR of delivered images; energy; goodput.

**Mechanism log, per arm:**
- planned / gated decisions; switches; aborts;
- robustness checks and rejects;
- band quantiles (ℓ and h at 10, 50 and 90%); relative-bias quantiles;
- integrated decision time p99; share of slots over 50 ms.

**Reproducibility note:** the 50 ms budget is wall-clock, so MPC arms are not bit-reproducible. Phase 2 runs with
6 worker processes and nothing else on the machine.

## 8. Run plan after approval

1. Create `phase2/AUTHORIZED` (commit), then run `fd_phase2.py test --procs 6`: 6 arms × 4 pools × 100 seeds =
   2,400 episodes, about 45 min CPU. Launched detached.
2. Run `fd_phase2_summary.py`, written and committed **before** the test run. It implements section 7 exactly.
3. Write the result report with every hypothesis and pool, whether positive or negative.

**Outside Phase 2:**
- RL (Phase 3) and the network/bandwidth grid (Phase 4) need their own pre-registrations.
- E1–E7 stay unchanged.
