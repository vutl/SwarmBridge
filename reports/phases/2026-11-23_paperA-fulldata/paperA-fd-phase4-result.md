# Paper A extended evaluation: Phase 4 (network size and bandwidth) — results (25/11)

**Status:** pre-registered (frozen a1b09178), run 18:51–20:01 on 24/11. These results come from the pre-committed
`fd_next_summary.py grid --name test`.

**Coverage:** 4,800 / 4,800 episodes; integrity counters all 0.

**Setting:**
- H\* per configuration, tuned on rl_train: N3x2, N5x3 and N5x4 use mbt q3; N7x2 and B0.5 use mbt q1; B2 uses
  avif_20.
- TD 20000–20099 (primary), F 22000–22099 (replication).
- Arms: H\*, nc (unguarded nominal), ng (matched error-aware design, nominal model), cal (the same design, calibrated
  model).

## 1. Confirmatory (Holm over 12 on TD), return

| Config | G1 ng − H\* (TD) | Holm | F | G2 cal − ng (TD) | Holm | F |
|---|---|---|---|---|---|---|
| N3x2 | +0.63 [−0.75, +2.00] | not rejected | +2.30\* | +1.34 [+0.08, +2.60] | not rejected (p_Holm 0.19) | +2.50\* |
| N7x2 | **+6.78 [+3.40, +10.15]** | **rejected (+)** | +10.87\* | +1.33 [−2.52, +5.18] | not rejected | +4.21\* |
| N5x3 | **+5.19 [+2.82, +7.57]** | **rejected (+)** | +7.29\* | **+2.79 [+1.00, +4.59]** | **rejected (+)** | +2.77\* |
| N5x4 | **+2.95 [+1.77, +4.12]** | **rejected (+)** | +3.74\* | **+1.63 [+0.47, +2.80]** | **rejected (+)** | +1.66\* |
| B0.5 | **+7.42 [+5.46, +9.39]** | **rejected (+)** | +8.01\* | +0.22 [−1.67, +2.11] | not rejected | −0.41 |
| B2 | **+4.90 [+2.76, +7.05]** | **rejected (+)** | +4.88\* | +2.10 [−0.51, +4.71] | not rejected | +2.85\* |

**Wording (pre-registered rule):**
- The error-aware design improved return over the per-configuration tuned incumbent in **5 of 6** configurations. In
  N3x2 no difference was detected on TD.
- A calibration benefit over the matched nominal design was detected **only in N5x3 and N5x4**.
- In the other four configurations: no detected difference. All six TD point estimates are positive (+0.2 to +2.8).
- No pooled statement is made.

## 2. Supporting (return)

| Config | G3 cal − H\* (TD / F) | G4 ng − nc (TD / F) |
|---|---|---|
| N3x2 | +1.96\* / +4.80\* | +11.53\* / +8.17\* |
| N7x2 | +8.11\* / +15.08\* | +11.44\* / +5.59\* |
| N5x3 | +7.99\* / +10.06\* | +13.96\* / +11.78\* |
| N5x4 | +4.58\* / +5.40\* | +13.62\* / +12.95\* |
| B0.5 | +7.64\* / +7.60\* | +2.83 (p 0.054) / −0.12 |
| B2 | +7.00\* / +7.72\* | +30.37\* / +22.63\* |

## 3. Recall (TD; seed-level CI, percentage points)

| Config | G1 ng − H\* | G2 cal − ng | Absolute recall H\* / nc / ng / cal (%) |
|---|---|---|---|
| N3x2 | −0.13 (n.s.) | −0.38\* | 48.23 / 46.91 / 48.10 / 47.73 |
| N7x2 | +1.35\* | −0.63\* | 30.73 / **35.74** / 32.08 / 31.45 |
| N5x3 | −0.37\* | −0.62\* | 47.43 / 46.00 / 47.07 / 46.44 |
| N5x4 | −0.21\* | −0.63\* | 48.36 / 47.59 / 48.16 / 47.52 |
| B0.5 | +2.29\* | −0.80\* | 31.72 / **38.24** / 34.01 / 33.21 |
| B2 | +1.16\* | −1.02\* | 37.16 / **40.10** / 38.32 / 37.30 |

**The return gains do not translate uniformly into recall:**
- **Calibration lowers recall in all six configurations** (TD; replicated on F except N3x2).
- The error-aware design raises recall over H\* in N7x2, B0.5 and B2. It lowers recall slightly in N5x3 and N5x4. In
  N3x2 no difference was detected.
- The unguarded nominal planner has the highest recall in N7x2, B0.5 and B2.

## 4. Controller timing (TD, p99, ms)

| Config | Scheduler alone | Full-sum ng (sequential) | Max ng (parallel) | Sequential share > 50 ms (ng) | Parallel share > 50 ms (ng; nc) |
|---|---|---|---|---|---|
| N3x2 | 1.3 | 109.8 | 41.3 | 0.436 | 0; 0 |
| N7x2 | **28.6** | 220.4 | **67.0** | 0.727 | **0.307; 0.379** (all 100 episodes) |
| N5x3 | 10.7 | 165.8 | 50.4 | 0.664 | 0.037; 0.018 (96 / 77 episodes) |
| N5x4 | **30.9** | 189.1 | **69.6** | 0.715 | **0.411; 0.429** (all 100 episodes) |
| B0.5 | 3.8 | 156.8 | 43.4 | 0.616 | 0.0002; 0 |
| B2 | 3.8 | 176.5 | 43.7 | 0.584 | 0.0001; 0.0003 |

**No real-time claim is made for N7x2 or N5x4.** Even with hypothetical parallel execution, 31–43% of slots exceed
50 ms there. The (M+1)^K scheduler enumeration alone takes about 29–31 ms p99. Sequentially, every configuration
exceeds the budget in 44–73% of slots.

**Other diagnostics:**
- Update overhead p99 0.3–0.8 ms.
- Aborts per episode ≤ 0.08.
- The per-mode-decision >50 ms share is about 0.
