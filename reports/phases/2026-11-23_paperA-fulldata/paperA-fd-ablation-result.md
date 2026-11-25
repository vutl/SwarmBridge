# Paper A extended evaluation: mechanism ablation — results (25/11)

**Status:** pre-registered (frozen a1b09178, authorized c4a5e299), run to completion. These results come from the
pre-committed `fd_next_summary.py ablation --name test` and are reported unchanged. Supporting outputs are in
`runs/uav_semcom/paperA_fd/supplement_extras.json`.

**Execution:**
- The first launch (24/11 14:59) died at 15:01, when the WSL VM stopped; it wrote no output.
- The relaunch (18:19, HEAD 899eed2c, run code unchanged) completed at 18:51.
- Coverage check: 2,400 / 2,400 episodes; 52 K deliveries per episode; integrity counters all 0.

**Setting:**
- Reference configuration N5x2 (K = 5, M = 2, 1 MHz); incumbent H\* = fixed_avif_20.
- Fresh environment seeds: TD 24000–24099 (primary), V 25000–25099, F 26000–26099.

**Arms:** all MPC arms use the nominal rate model unless marked calibrated.

| Short name | Arm | What it is |
|---|---|---|
| H | H\* | tuned heuristic, no planner |
| nc | `+mpcnc` | unguarded nominal planner, H = 4 |
| gate | `+mpcngg` | tracker gate only |
| band | `+mpcngb` | robustness band only (cold start unguarded) |
| ng | `+mpcng` | gate + band, H = 4 (the matched design) |
| h1 | `+mpcngh1` | gate + band, H = 1 |
| h0 | `+mpcngh0` | gate + band, H = 0 (screen and planner) |
| cal | `+mpc` | gate + band, H = 4, calibrated |

## 1. Confirmatory family (Holm over A1, A2, A4, A4b; TD)

Return; paired t-CI over 100 environment seeds. An asterisk marks a 95% CI that excludes 0.

| ID | Contrast | TD | Holm | V | F |
|---|---|---|---|---|---|
| A1 | gate − nc | −1.19 [−5.63, +3.25] | not rejected | −2.86 [−7.19, +1.48] | +0.61 [−3.92, +5.13] |
| A2 | band − nc | **+15.15 [+11.41, +18.89]** | **rejected (+)** | +13.36\* | +16.19\* |
| A4 | ng (H4) − h1 | **+5.05 [+2.57, +7.53]** | **rejected (+)** | **−2.51 [−4.84, −0.17]\*** | +0.38 [−2.56, +3.32] |
| A4b | ng (H4) − h0 | **+25.49 [+22.28, +28.71]** | **rejected (+)** | +14.59\* | +21.20\* |

## 2. Supporting contrasts (return)

| ID | Contrast | TD | V | F |
|---|---|---|---|---|
| A3 | ng − nc | +18.05\* | +9.10\* | +14.05\* |
| A5 | ng − gate (adding the band to the gate) | +19.25\* | +11.96\* | +13.45\* |
| A6 | ng − band (adding the gate to the band) | +2.90 [+0.08, +5.73]\* | **−4.26 [−7.07, −1.45]\*** | −2.14 [−6.03, +1.75] |
| A7 | cal − ng | −0.75 [−2.87, +1.38] | +2.48\* | +2.62\* |
| A8 | ng − H\* | +7.44\* | +2.28\* | +5.42\* |
| A9 | h1 − h0 | +20.44\* | +17.10\* | +20.82\* |

## 3. Recall (Qmin-compliant; seed-level paired CI; video-cluster bootstrap in the summary log)

**Recall contrasts (percentage points):**

| ID | TD | V | F |
|---|---|---|---|
| A1 gate − nc | −0.77\* | −1.41\* | −0.96\* |
| A2 band − nc | −0.71\* | −2.41\* | −1.05\* |
| A4 ng − h1 | −0.21 (n.s.) | −0.74\* | −0.66\* |
| A4b ng − h0 | +0.53\* | −0.61\* | +0.63\* |

**Absolute recall on TD (%):**

| H\* | nc | gate | band | ng | h1 | h0 | cal |
|---|---|---|---|---|---|---|---|
| 37.12 | **40.50** | 39.73 | 39.79 | 38.69 | 38.90 | 38.15 | 37.59 |

**The unguarded nominal planner has the highest recall of all arms**, although its return is the lowest of the H4
planners. The return-improving mechanisms (band, gate) lower recall relative to it. The calibrated arm has the lowest
recall of the planners. Return and recall are not aligned here (see the consolidated report, §3).

## 4. Controller timing (TD; ms; p99 per slot)

| Arm | Full-sum (sequential, primary) | Max (parallel, hypothetical) | Sequential share of slots > 50 ms | Parallel share > 50 ms (episodes with any) | Update p99 |
|---|---|---|---|---|---|
| H | 3.8 | 3.8 | 0 | 0 (0) | — |
| nc | 152.8 | 43.6 | 0.727 | 0.0002 (2) | — |
| gate | 148.9 | 43.6 | 0.602 | 0.0013 (7) | 0.63 |
| band | 164.0 | 45.1 | 0.768 | 0.0023 (13) | 0.63 |
| ng | 159.9 | 43.7 | 0.626 | 0.0011 (6) | 0.60 |
| h1 | 111.5 | 30.6 | 0.494 | 0 (0) | 0.58 |
| h0 | **16.0** | 6.6 | **0** | 0 (0) | 0.57 |
| cal | 145.4 | 40.1 | 0.593 | 0 (0) | 0.40 |

**Under sequential execution, every planner except H0 exceeds the 50 ms slot in about half to three quarters of the
slots.** Only the H0 variant fits sequentially, and it is the worst planner in return.

**Planning aborts per episode** (wall-clock budget): band 0.94, gate 0.41, ng 0.19, nc 0.12, h1 / h0 / cal 0.
Abort categories were not logged and are unavailable.

**Band quantiles** (lower edge, q10 / q50 / q90): ng 0.36 / 0.46 / 0.55; cal 0.47 / 0.64 / 0.74. The band-only arm's
lower quantile is 1.0, because of its unguarded cold start. **Gate share of decisions:** about 50 / 260 in every gated
arm.

## 5. Pre-registered wording

**The band is necessary:**
- Adding the robustness band to the unguarded nominal planner improved return (A2, Holm-rejected; replicated on V and F).
- Adding the tracker gate alone produced no detected difference (A1), and it lowered recall.

**The gate is not shown to be necessary:**
- Gate + band versus band alone (A6, supporting) was +2.9 on TD, −4.3\* on V and −2.1 (n.s.) on F.

**Horizon:**
- H4 beat H1 on TD (A4, Holm-rejected). It was worse on V (−2.5\*), with no detected difference on F. **The benefit of
  H4 over H1 does not replicate.**
- H4 beat H0 everywhere (A4b), and H1 beat H0 everywhere (A9, about +17 to +21). Anticipating at least one future
  arrival matters; more than one is not consistently better.

**Calibration within the matched design (A7):**
- TD −0.75 (n.s.), the same as Phase 2 H3b;
- V +2.5\* and F +2.6\* (replications only; they do not upgrade the TD statement);
- calibration lowers recall.
