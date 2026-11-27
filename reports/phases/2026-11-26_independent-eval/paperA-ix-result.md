# Paper A independent evaluation: results (27/11)

**Status: CONFIRMATORY, pre-registered.** Protocol: `paperA-ix-prereg.md` (FINAL v3). All 10,200 registered test
episodes are complete. Everything below comes from one pass of the registered analysis script, with no change to any
data, arm, parameter or analysis after the L2 lock.

**Sources:**
- Analysis: `runs/uav_semcom/paperA_ix/analysis/summary.log` and `analysis/<block>.json`.
- Per-arm rows: `eval/<block>.json`.
- Publication values: `paper/ix_paper_numbers.json`.

## 1. Execution

| Step | When (27/11, KST) | Outcome |
|---|---|---|
| Stage A: profile U_train / U_val, CPU timing, fit, H\* tuning, 9 RL runs, selection | 00:26–14:22 | 9/9 runs completed 2M decisions each; all 14 modes executed in training |
| L2 lock | 17:14 | Hashes of manifest, stage-A profile, CPU estimate, both controller states, H\*, checkpoint selection, 9 checkpoints, 22 code files; `ix_stage.py verify` OK |
| Stage B UAVDT: U_test profile + fidelity, 6 blocks | 17:14–19:49 | 8,800 episodes |
| Stage B X1: profile, 2 blocks | 20:39–22:03 | 1,400 episodes |

**X1 launch history:** a first queued launch failed at start (19:50). The uv interpreter shim could not spawn the base
interpreter (os error 2). This happened before any X1 data was read. It was relaunched at 20:39.

**Integrity (every block):** 0 wrong task counts, 0 seeds with differing images across arms, 0 deliveries outside the
pool.

**Recall on UAVDT originals** (frozen YOLO11s, IoU 0.5, ignore regions removed):

| Mapping | Recall |
|---|---|
| vehicle, class-agnostic | 78.94 |
| car ← car / van | 75.44 |
| car only | 70.85 |

X1 has no public labels, so recall is unavailable there.

## 2. Principal finding

Principal finding: N2 (guarded − H\*) > 0 and A2 (band only − nominal) > 0. Return per episode; paired 95% CI; Holm
within family.

| Block | N2 guarded − H\* | A2 band only − nominal | Verdict |
|---|---|---|---|
| **X1** (principal; VisDrone test-challenge, frozen fd_v1 artifacts) | +2.13 [−0.42, +4.69], p 0.10, not rejected | +19.27 [+16.00, +22.54], rejected | **partially supported** |
| X2-retrained (UAVDT, procedure re-applied) | +7.61 [+4.65, +10.57], rejected | +9.19 [+5.69, +12.69], rejected | supported |
| X2-frozen (UAVDT, fd_v1 artifacts) | +1.17 [+0.18, +2.15], Holm p 0.041, rejected | +11.82 [+8.05, +15.59], rejected | supported |

No N2 or A2 is negative. The two X2 conditions share U_test and are not independent replications.

## 3. Other registered contrasts (return)

| Contrast | X1 | X2-frozen | X2-retrained |
|---|---|---|---|
| N1 nominal − H\* | **−14.69** (rejected) | +1.05 n.s. | −0.26 n.s. |
| N3 calibrated − guarded | **+3.86** (rejected) | **−1.65** (rejected) | −0.43 n.s. |
| A1 gate only − nominal | −0.09 n.s. | −0.03 n.s. | −2.00 n.s. |
| A4 H=4 − H=1 | +1.95 n.s. | **−1.23** (rejected) | **+5.84** (rejected) |
| A4b H=4 − H=0 | **+23.18** (rejected) | **+5.29** (rejected) | **+23.78** (rejected) |
| H1 calibrated − H\* (reference block) | **+5.54** (rejected) | **−0.85** (rejected) | **+6.98** (rejected) |
| H3b calibrated − guarded (reference block) | +0.42 n.s. | **−1.77** (rejected) | +0.21 n.s. |

### X2-retrained network settings

Grid family of 12.

| Setting | G1 guarded − H\* | G2 calibrated − guarded |
|---|---|---|
| K=3 | +0.69 n.s. | **+2.56** |
| K=7 | **+14.08** | **−10.10** |
| M=3 | **+11.51** | +2.37 n.s. |
| M=4 | **+3.06** | **+2.60** |
| B=0.5 | **+5.80** | +1.07 n.s. |
| B=2 | **+6.26** | +1.16 n.s. |

### X2-retrained learned incumbents

RL family of 6; three checkpoints each.

| Incumbent | R1 guarded on RL − RL | R2 calibrated − guarded |
|---|---|---|
| PPO | **+6.88** | **+2.75** |
| QR-DQN | **+22.93** | +1.84 (Holm p 0.069) |
| Discrete SAC | **+37.60** | −0.40 n.s. |

Levels (not a registered contrast): PPO −36.78 vs H\* −31.48.

## 4. Recall (UAVDT only)

System-level vehicle recall, sequence-cluster bootstrap. All three class mappings agree in sign.

| Contrast | X2-frozen | X2-retrained |
|---|---|---|
| N2 guarded − H\* | +1.0 | +4.5 |
| A2 band only − nominal | −7.7 | −1.1 |
| N1 nominal − H\* | +16.5 | +7.3 |

The return–recall trade-off between nominal and banded planning seen on VisDrone recurs on UAVDT.

## 5. Mode utilisation and cost

**Mode utilisation:**
- Generative modes are ≤ 0.7% in all MPC arms.
- Exceptions: the RobustMPC-style predictor without band (21–26%), DSAC alone (9–13%) and QR-DQN alone (5–11%).
- cap_t2i is never compliant.

**Controller time:** same pattern as fd_v1.
- Sequential per-slot p99 is 129–224 ms, and 47–83% of slots exceed 50 ms. Only H=0 fits (16 ms).
- With hypothetical parallel per-UAV decisions, p99 is 31–50 ms at K ≤ 5 and M ≤ 3, but 69–75 ms at K=7 and M=4.
- Returns do not charge controller time.

## 6. Deviations

None from the registered protocol. The only operational event is the failed first X1 launch (§1). No tuning,
retraining, mode removal or added analysis followed the results.
