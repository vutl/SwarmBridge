# Paper A extended evaluation: Phase 3 (learned incumbents) — results (25/11)

**Status:** pre-registered (frozen a1b09178), run to completion.
- Analysis: the pre-committed `fd_next_summary.py rl --name zeroshot | retrained`.
- Supplementary checks: `fd_next_supplement.py ckpts | extras`.

**Execution:**

| Step | Time | Episodes |
|---|---|---|
| Zero-shot | 24/11 20:01–20:50 | 3,600 |
| Training (9 runs) | 20:51 → 25/11 10:13 | — |
| Retrained evaluation | 10:13 → 13:59 | 14,400 |

**Interruption:**
- A host power loss (25/11 about 12:14) interrupted the retrained evaluation during pool V. TD (11:03) and F (11:53)
  were already complete.
- The resume used `--skip-existing` (000251c7, an execution-only change): completed pools were kept and not
  re-simulated. MPC arms are not bit-reproducible, so a re-run would have given a second realisation.
- V and TD_all were run after the restart. The interrupted step log is preserved.

**Checks:**
- Coverage: 18,000 / 18,000 episodes.
- Checkpoints: 9 / 9 `ckpts.json` == `selection.json`, sha256 verified by the runner at the start of evaluation and
  again post-chain.

## 1. Training record (selection on rl_val; weights not committed)

| Run | Selected step | Best val return | Nominal / actual steps | Updates | Wall time |
|---|---|---|---|---|---|
| ppofd s0 / s1 / s2 | 1.4M / 1.6M / 1.6M | −73.02 / −72.67 / −72.87 | 2.0M / 2,027,520 | 3,300 | 0.77 h |
| qrfd s0 / s1 / s2 | 1.6M / 1.4M / 1.8M | −96.61 / −96.24 / −88.75 | 2.0M / = nominal (transition-counted) | 495,000 | 3.74–3.78 h |
| dsfd s0 / s1 / s2 | 2.0M / 1.8M / 1.8M | −100.37 / −105.34 / −95.79 | 2.0M / = nominal | 495,000 | 3.52–3.58 h |

**Off-policy step counts:** the off-policy trainers count environment transitions exactly. The `actual_timesteps`
field exists only for PPO, where rollout blocks overshoot by 27,520 steps.

**dsfd s0 selected its last checkpoint (2.0M).** Its validation return was still improving at the end of the budget.

## 2. Retrained incumbents: confirmatory (Holm over 6 on TD), return

| X | R1 X+ng − X (TD) | Holm | F / V / TD_all | R2 X+cal − X+ng (TD) | Holm | F / V / TD_all |
|---|---|---|---|---|---|---|
| PPO-LA | **+7.38 [+5.99, +8.77]** | rejected (+) | +8.01\* / +6.18\* / +8.54\* | **+3.13 [+1.57, +4.69]** | rejected (+) | +3.51\* / +1.26 (p 0.059) / +1.66\* |
| QR-DQN | **+19.01 [+17.01, +21.02]** | rejected (+) | +20.79\* / +20.75\* / +20.72\* | **+4.68 [+3.02, +6.34]** | rejected (+) | +4.18\* / +3.70\* / +2.57\* |
| Discrete SAC | **+25.73 [+23.79, +27.67]** | rejected (+) | +29.13\* / +25.99\* / +27.48\* | **+6.00 [+4.23, +7.77]** | rejected (+) | +6.73\* / +7.19\* / +5.45\* |

**Wording (pre-registered):**
- The error-aware layer improved incumbents from **3 of 3** learned families.
- The calibrated predictor improved over the matched nominal design for **3 of 3** learned families.
- This contrasts with the heuristic incumbent at the reference setting: Phase 2 H3b, TD −0.10 (n.s.); ablation A7,
  TD −0.75 (n.s.).

**Per training seed:** every R1 and R2 per-seed contrast on TD is positive, with a CI excluding 0. The ppofd s2 R2 CI
has a lower bound of 0.00.
- R1 spread across the three seeds: PPO 6.2–9.3, QR-DQN 15.7–24.8, DSAC 20.1–32.7.
- R2 spread: PPO 2.3–3.6, QR-DQN 3.1–6.2, DSAC 4.4–7.4.
- With 3 training seeds, no inferential claim is made about training-seed variance.

**Supporting:**
- R3 (X+cal − X): +10.5 / +23.7 / +31.7.
- R4 (X+ng − X+nc): +14.0\* / +6.6\* / +4.6\* on TD. On F, DSAC is −1.6 (n.s.).

## 3. Recall (TD, percentage points, seed level)

| X | R1 | R2 | Absolute X / X+nc / X+ng / X+cal (%) |
|---|---|---|---|
| PPO-LA | +2.44\* | −0.40\* | 34.93 / 40.34 / 37.37 / 36.97 |
| QR-DQN | +5.18\* | −0.42\* | 31.98 / 40.51 / 37.15 / 36.73 |
| DSAC | +7.03\* | −0.72\* | 27.37 / 40.29 / 34.40 / 33.68 |

**What the recall shows:**
- The layer raises recall substantially for every learned incumbent.
- Calibration again lowers recall.
- The unguarded nominal planner reaches the highest recall (about 40.3–40.5%) regardless of the incumbent.

## 4. Is RL better than the heuristic? (R5, descriptive; H\* rows from Phase 2, same seeds)

**Return:**

| X | X − H\* TD | F | V | TD_all | X+ng − H\*+ng TD | F | V | TD_all |
|---|---|---|---|---|---|---|---|---|
| PPO-LA | −2.96\* | −3.05\* | −3.08\* | −0.76 | −2.96\* | +1.56 | +0.99 | −0.36 |
| QR-DQN | −22.55\* | −22.83\* | −23.08\* | −21.20\* | −10.91\* | −5.44\* | −4.44\* | −8.62\* |
| DSAC | −34.06\* | −36.61\* | −39.33\* | −33.91\* | −15.71\* | −10.89\* | −15.45\* | −14.57\* |

**Recall, X − H\* (TD):** PPO −2.0\*, QR-DQN −5.0\*, DSAC −9.6\*. Matched with the layer: −1.2\*, −1.4\*, −4.1\*.

**No learned incumbent beat the tuned heuristic on return or recall.** PPO-LA is the closest: on TD_all, and with the
layer on V and F, no difference was detected.

## 5. Zero-shot (old checkpoints, frozen-policy transfer under the fd_v1 workload and observation estimator; supporting; TD)

| | PPO-LA | QR-DQN | DSAC |
|---|---|---|---|
| R1 X+ng − X | +7.59\* | +16.92\* | +22.52\* |
| R2 X+cal − X+ng | +4.31\* | +5.74\* | +7.10\* |
| R5 X − H\* | −4.85\* | −24.16\* | −33.81\* |
| R5 X+ng − H\*+ng | −4.64\* | −14.62\* | −18.67\* |

**Recall:** R1 +1.0\*, +2.8\*, +4.5\*; R2 −0.3\*, −0.4\*, −0.6\*.

**Comparison with retraining:** retraining on fd_v1 changed little. PPO is −2.96 vs −4.85 against H\*; QR-DQN and DSAC
are unchanged within noise.

## 6. Timing (TD, p99, ms)

| Arm | Full-sum (sequential) | Parallel | Sequential share > 50 ms |
|---|---|---|---|
| Incumbents alone | 4.8–16.7 (PPO inference about 3 ms per decision) | — | 0 |
| With any planner | 154–173 | 44–47 | 0.62–0.73 |

The parallel share above 50 ms is ≤ 0.0014.
