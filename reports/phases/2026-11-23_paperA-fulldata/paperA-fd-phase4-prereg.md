# Paper A extended evaluation: Phase 4 (network size and bandwidth) pre-registration (24/11)

**Status:** submitted for review. Done so far:
- H\* tuned per configuration on **rl_train** (seeds 0–29);
- a 3-seed rl_val smoke run.

No test seed has been simulated.

**Question:**
1. Does the error-aware selective MPC layer improve the per-configuration tuned incumbent outside the reference
   setting?
2. Does the ACK-calibrated empirical predictor add anything in some regime?

Configurations are **not** searched for a calibration win. All are reported.

## 1. Configurations (the E4/E6 grid on fd_v1; nominal LA; same JA scheduler on pacedB3, not re-optimised)

| Configuration | K | M | Total bandwidth | H\* (tuned on rl_train) | rl_train return of H\* (next best) |
|---|---|---|---|---|---|
| N5x2 (reference, **Phase 2 results reused**, same TD / F seeds) | 5 | 2 | 1 MHz | fixed_avif_20 | −71.2 (mbt q1 −75.8) |
| N3x2 | 3 | 2 | 1 MHz | fixed_mbt2018-mean_q3 | −4.4 (greedy+budget1 −10.8) |
| N7x2 | 7 | 2 | 1 MHz | fixed_mbt2018-mean_q1 | −159.7 (avif_20 −173.3) |
| N5x3 | 5 | 3 | 1 MHz | fixed_mbt2018-mean_q3 | −37.0 (greedy+budget1 −41.1) |
| N5x4 | 5 | 4 | 1 MHz | fixed_mbt2018-mean_q3 | −14.2 (greedy+budget1.25 −20.8) |
| B0.5 | 5 | 2 | 0.5 MHz | fixed_mbt2018-mean_q1 | −90.0 (avif_20 −92.6) |
| B2 | 5 | 2 | 2 MHz | fixed_avif_20 | −62.0 (mbt q1 −65.0) |

**Tuning rule:** the same 22-rule family and the same rule as Phase 2. File: `runs/uav_semcom/paperA_fd/phase4/tune.json`.

## 2. Arms, pools and seeds

**Arms** (incumbent H\* of the configuration):
- H\*;
- `+mpcnc`: unguarded nominal;
- `+mpcng`: error-aware design, nominal model;
- `+mpc`: error-aware design, calibrated model.

**Pools:**
- **TD 20000–20099 (primary)** and **F 22000–22099** (method-unseen replication).
- These are the Phase 2 seeds, so each configuration pairs with the reference.
- V is omitted to bound the cost. This is stated as a limitation.

**Size:** 6 configurations × 4 arms × 2 pools × 100 seeds = 4,800 episodes, about 2 h CPU. They run detached with
nothing else on the machine, and need `phase4/AUTHORIZED`.

## 3. Hypotheses (episode return; paired t over 100 seeds per configuration; TD primary)

For each new configuration c (6 of them):

| ID | Contrast | Family |
|---|---|---|
| **G1_c** | error-aware design (nominal) − H\* | **confirmatory** |
| **G2_c** | calibrated − error-aware design (nominal) | **confirmatory** |
| G3_c | calibrated − H\* | supporting |
| G4_c | error-aware design − unguarded nominal | supporting |

**Multiplicity:** Holm over the 12 confirmatory tests on TD (6 configurations × {G1, G2}), family-wise α = 0.05,
two-sided.

**Wording:**
- Per configuration: "improved / reduced / no detected difference" by Holm rejection and sign.
- A regime-specific calibration benefit (G2_c Holm-rejected and positive) is stated **only for that configuration**,
  with F as its replication.
- No pooled statement across configurations is upgraded from individual ones.
- The reference N5x2 is reported from Phase 2. It is not part of this Holm family.

**Secondary, per configuration and arm:**
- return per image (normalised by 52·K); Qmin-compliant recall (seed-level CI, plus the video-cluster bootstrap as in
  Phase 2);
- on-time, drop, late and Qmin-violation rates; energy; goodput.

**Compute and latency, per configuration and arm:**
- p99 per slot for max (parallel assumed), sum (sequential) and full-sum (including the tracker / calibrator update);
- update overhead p99;
- scheduler time;
- the unrounded share of slots over 50 ms (max and full-sum), and the number of episodes with any;
- planning aborts and truncations; gated, planned and switched decisions; robustness checks and rejects.

**No real-time claim** for configurations where the scheduler alone approaches the budget.

**Validation smoke** (3 rl_val seeds; not interpreted):
- p99 max is about 62–74 ms already at N7x2 and N5x4, because the scheduler enumerates (M+1)^K assignments, against
  40–45 ms elsewhere;
- p99 sum is 160–210 ms.

**Integrity asserts:** as in Phase 2, with K × 52 tasks per episode.

**Analysis script:** committed before the run.
