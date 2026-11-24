# Paper A extended evaluation: Phase 2 results (24/11)

**Pre-registration:** [paperA-fd-phase2-prereg.md](paperA-fd-phase2-prereg.md). Frozen at section 9, commit 66dd28c4;
AUTHORIZED at 73b45071.

**Run:** 24/11, 13:11–13:46; 6 arms × 4 pools × 100 seeds = 2,400 episodes; EXIT 0.

**Analysis:** `fd_phase2_summary.py`, committed before the run.

**Files:**
- `runs/uav_semcom/paperA_fd/phase2/test_{TD,F,V,TD_all}.json` (+ `.deliv.json.gz`);
- `summary_test.json` / `.log`; `phase2_test.log`.

**Setting:** K = 5, M = 2, 1 MHz, nominal LA. The incumbent is H\* = fixed_avif_20, tuned on rl_train.

**Integrity:** in all four pools,
- every episode of every arm has 260 tasks;
- arms with the same seed have identical images;
- every delivery belongs to its pool.

## 1. Confirmatory family (Holm, TD = expanded primary evaluation)

| Contrast | TD (primary) | F (method-unseen) | V (historical) | TD_all (frame-weighted, same seeds as TD) |
|---|---|---|---|---|
| **H1** cal − H\* | **+7.28 [+5.05, +9.51]**, p_Holm < 1e-4, **rejected** | +7.37 [+5.00, +9.75]\* | +4.05 [+2.47, +5.63]\* | +8.30 [+5.82, +10.78]\* |
| **H3b** cal − nominal (matched design) | **−0.10 [−1.95, +1.76]**, p_Holm 0.92, **not rejected** | +3.97 [+1.27, +6.67]\* | +1.94 [−0.14, +4.01] | +0.16 [−2.05, +2.36] |

(\* CI excludes 0.)

**Verdicts by the pre-registered rules:**
- **H1, confirmatory:** the calibrated selective-MPC stack improved return over the tuned incumbent on TD. It replicated
  with CI > 0 on F, V and TD_all.
- **H3b:** no detected difference between the ACK-calibrated and the nominal rate model within the common control
  design on TD. This is not evidence of equivalence.
  - Replication status: F has CI > 0 (+4.0); V and TD_all include 0.
  - The paper may therefore **not** claim that the calibrated rate model itself improves return. Most of the
    improvement over the nominal planner comes from the shared control design: the tracker gate, the band and the
    robustness check (compare H3 below).

## 2. Supporting contrasts (unadjusted 95% CIs; no additional confirmatory claim)

| Contrast | TD | F | V | TD_all |
|---|---|---|---|---|
| H2 cal − RobustMPC-style (band) | +6.34 [+3.98, +8.71] | +7.59 [+5.29, +9.90] | +4.06 [+2.37, +5.75] | +8.29 [+5.80, +10.79] |
| H2b cal − RobustMPC-style (no band) | +51.23 [+47.81, +54.66] | +54.81 [+50.82, +58.80] | +52.95 [+50.19, +55.70] | +51.84 [+48.27, +55.42] |
| H3 cal − nominal (no gate / band) | +18.59 [+15.09, +22.09] | +12.09 [+8.78, +15.40] | +12.77 [+9.26, +16.27] | +11.74 [+8.32, +15.16] |

- **RobustMPC-style.** The calibrated MPC beat both *implemented* variants in every pool. This says nothing about
  RobustMPC methods in general.
  - The band variant switched 1.2–2.1 times per episode, and its return equals H\* alone within 1 point. It is a
    near-inactive baseline, as found in the smoke run and in E4.
  - The no-band variant switched about 158 times per episode and lost about 45 points to H\*.
- **H3 against H3b.** Adding the tracker gate and band to the nominal planner (`+mpcng`) recovers the whole gap: on
  TD, nominal −59.4 → matched design −40.7, against −40.8 for the calibrated arm.

## 3. Return levels and mechanism (TD; the other pools in summary_test.log)

| Arm | Return | On-time | Drop | Recall (%) | Switches / ep. | Rejects / checks | Band median ℓ / h | Integrated p99 |
|---|---|---|---|---|---|---|---|---|
| H\* | −48.09 | 0.831 | 0.024 | 36.94 | — | — | — | 3.7 ms |
| +mpc (calibrated) | −40.81 | 0.861 | 0.019 | 37.70 | 71.7 | 70 / 142 | 0.65 / 1.26 | 42.4 ms |
| +mpcnc (nominal) | −59.40 | 0.863 | 0.085 | 40.27 | 181.5 | 0 / 182 | — | 43.5 ms |
| +mpcng (nominal, matched) | −40.71 | 0.865 | 0.023 | 38.54 | 57.4 | 93 / 150 | 0.48 / 1.15 | 43.6 ms |
| +mpcrh | −47.15 | 0.833 | 0.023 | 36.99 | 2.1 | 120 / 122 | 0.05 / 10,280 | 42.3 ms |
| +mpcrhnb | −92.04 | 0.673 | 0.007 | 26.10 | 157.6 | 0 / 158 | 1.00 / 1.00 | 37.2 ms |

- **Gating** is the same for the calibrated and matched-nominal arms: about 51 of 260 decisions gated in both.
- **Band and switches.** The matched-nominal band is wider and lower (ℓ 0.48 against 0.65), because the nominal model
  over-predicts rates (median relative bias around −0.35). So it rejects more candidate switches (93/150 against
  70/142) and switches less (57 against 72). The realised gate and band values differ, as noted in the
  pre-registration.
- **Decision time:** p99 42–44 ms for every MPC arm; no slot over 50 ms in any pool (`int_over50` = 0.000 for all
  arms on TD).

## 4. Recall (Qmin-compliant; secondary)

| Contrast | TD seed-level | TD video-cluster | F seed / cluster | V seed / cluster | TD_all seed / cluster |
|---|---|---|---|---|---|
| H1 cal − H\* | **+0.75 [+0.47, +1.04]** | +0.76 [−0.11, +1.61] | +1.51\* / +1.52\* | +1.00\* / +1.00\* | +0.84\* / +0.84 [−0.51, +1.78] |
| H3 cal − nominal | **−2.58 [−3.05, −2.10]** | −2.57\* | −3.75\* / −3.75\* | −4.77\* / −4.77\* | −2.64\* / −2.64\* |
| H3b cal − nominal (matched) | **−0.85 [−1.06, −0.64]** | −0.85\* | −0.34 / −0.33\* | −0.39\* / −0.39\* | −0.74\* / −0.75\* |

- **Return and recall disagree again:**
  - The nominal planner has the lowest return among the active MPC arms but the **highest recall**: 40.3% against
    37.7% on TD.
  - The calibrated rate model has slightly lower recall than the matched nominal one in every pool (−0.3 to −0.9
    points), while its return is not distinguishable on TD.
- **Calibrated stack against H\*:** recall rises at seed level in every pool (+0.75 to +1.5 points). The video-cluster
  CI includes 0 on TD and TD_all, so the TD recall gain is not robust to scene composition.
- **Scope of the video-cluster analysis:** it is a scene-composition sensitivity analysis conditional on the simulated
  outcomes, not a scene-generalisation CI for return.

## 5. Interpretation and consequences for Paper A

1. **The full method improves the tuned incumbent on the extended data.** This holds in every pool: +4 to +8 return
   points, with recall up about 1 point at seed level.
   - It differs from E4/E6, where the gain over H\* was condition-dependent (+2.7 n.s. at N5x2).
   - Here the incumbent is AVIF-20 rather than MBT2018 q3. MBT2018's measured CPU encode time is about 60% higher than
     before, and the frames are larger.
2. **The ACK-calibrated rate model is not what produces the gain on the primary pool.** A nominal model inside the
   same tracker-gated, band-checked design performs the same on TD (−0.1) and on TD_all (+0.2). It is better only on
   F (+4.0, a Holm rejection within F) and in a non-significant direction on V.
   - The earlier comparisons "calibrated vs nominal: +5 to +36" (E1–E7) mostly measured **the gate, band and
     robustness check that came with the calibrated arm**, not the calibration itself.
   - The method contribution of Paper A must be restated: the selective, error-tracked, band-checked MPC layer is what
     helps. ACK calibration of the rate model is not shown to add return on the primary evaluation.
3. **RobustMPC-style:** the E1–E7 wins were against a near-inactive baseline (confirmed here: about 2 switches per
   episode, return ≈ H\*). Removing its band makes it much worse. Neither variant is a faithful RobustMPC.
4. **Return and recall still disagree,** most clearly for the nominal planner. Calibration also costs a little recall
   relative to the matched nominal model.

These results agree with the second scenario of the review: the benefit comes mainly from the MPC and intervention
mechanism, not from the calibration.

## 6. Status

- **Phase 2 is complete.**
- **Next:** reviewer feedback on the consequences for the paper's claims.
- **Phase 3 (RL) and Phase 4 (network/bandwidth grid)** need their own pre-registrations. Given result 2, the reviewer
  may want Phase 4 to carry `+mpcng` as well.
- **E1–E7 are unchanged.** Their interpretation (sections 5.2 and 5.3) must be corrected in the manuscript.

## 7. Corrections after the review of 50835621 (descriptive; no rerun)

### 7.1 Latency: the statement in section 3 was wrong

Section 3 said "no slot over 50 ms in any pool (`int_over50` = 0.000 for all arms on TD)". The rounded mean hid
non-zero episodes. The code also records two integrated-latency metrics:
- **max** = scheduler time + the largest mode decision of the slot. This assumes the mode decisions of a slot run in
  parallel.
- **sum** = scheduler time + all mode decisions of the slot. This is how the simulation actually executes them:
  sequentially.

In both, the timer starts **after** the calibrator and tracker updates of the slot, so their overhead is excluded.

Means over 100 episodes of the per-episode p99; slots over 50 ms use the max metric and are not rounded:

| Pool | Arm | p99, max metric | p99, sum metric | Share of slots > 50 ms: mean (largest episode) | Episodes with any slot > 50 ms |
|---|---|---|---|---|---|
| TD | H\* | 3.7 ms | 3.7 ms | 0 | 0 |
| TD | +mpc | 42.4 ms | **147.6 ms** | 0 | 0 |
| TD | +mpcnc | 43.5 ms | **154.7 ms** | 0 | 0 |
| TD | +mpcng | 43.6 ms | **152.9 ms** | 0.00012 (1.18%) | 1 |
| TD | +mpcrh | 42.3 ms | 137.3 ms | 0.00011 (1.15%) | 1 |
| TD | +mpcrhnb | 37.2 ms | 124.0 ms | 0 | 0 |
| F | +mpc / +mpcng | 42.4 / 43.1 ms | 149.3 / 152.8 ms | 0 / 0.00012 (1.23%) | 0 / 1 |
| V | +mpc / +mpcng | 42.8 / 42.9 ms | 150.3 / 157.0 ms | 0.00036 (2.35%) / 0 | 2 / 0 |
| TD_all | +mpc / +mpcng / +mpcnc | 42.5 / 43.1 / 43.2 ms | 149.0 / 153.7 / 155.5 ms | 0 / 0 / 0.00011 (1.12%) | 0 / 0 / 1 |

**Claims allowed:**
- Each **mode decision** has a 50 ms planning budget, which is checked before each planning stage.
- The **per-slot total** is about 150 ms at p99 if the 5 decisions of a slot run sequentially. It is about 43 ms only
  if they run fully in parallel, an execution architecture that is assumed, not demonstrated.
- No strict real-time compliance is claimed. Tracker and calibrator overhead is not yet measured. It will be
  instrumented in the next phases.

### 7.2 Post-hoc contrasts (from the review; recomputed here from the raw per-episode returns and identical)

These are unadjusted, post-hoc supporting analyses, not pre-registered findings.

| Pool | Matched nominal − H\* | Matched nominal − nominal (no gate / band) |
|---|---|---|
| TD | +7.38 [+5.09, +9.66] | +18.69 [+14.88, +22.50] |
| F | +3.40 [+0.98, +5.83] | +8.12 [+4.44, +11.80] |
| V | +2.11 [+0.14, +4.08] | +10.83 [+7.53, +14.13] |
| TD_all | +8.14 [+5.43, +10.85] | +11.58 [+8.20, +14.97] |

Even without the ACK-calibrated empirical predictor, the error-tracked, gated, band-checked selective MPC improves on
H\* and on the unguarded nominal planner in all four pools.

### 7.3 Wording

- **What is not shown:** an additional return benefit of the ACK-calibrated *empirical rate predictor* on TD, once
  error tracking exists.
- **ACK feedback is still used:** the matched-nominal design relies on observed service feedback through its error
  tracker. "ACK feedback is unnecessary" must therefore not be claimed.
- **Scope for E1–E7:** Phase 2 shows that the earlier nominal-versus-calibrated comparisons confounded the substitution
  of the rate model with the intervention mechanism. It does not identify retrospectively how much each component
  contributed in each E1–E7 setting.
