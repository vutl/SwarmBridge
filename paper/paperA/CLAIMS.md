# Paper A (ICT Express rewrite, 25/11): claims and their sources

**Manuscript:** `main.tex`. Title: "Uncertainty-aware mode selection for multi-UAV semantic image offloading".
- **6-page version** (after the ChatGPT correction: original research articles allow up to 6 pages): 3 figures
  (`fig_fd_system`, `fig_fd_mechanism`, `fig_fd_tradeoff`) and 2 tables (Table I settings, Table II = `table_fd.tex`).
- Earlier versions: the 4-page draft is at `484953df`, the method letter at `1225e287`.

**Numbers added in the 6-page version:**

| Text | Value | Source | Status |
|---|---|---|---|
| Guarded MPC switches 54 per episode, drops 2.0% | 54.37; 0.0198 | ablation `TD.levels.ng` | S |
| Band median ℓ = 0.46 | 0.463 | `supplement_extras.json` ablation TD `+mpcng` `band_lo_q[1]` | S |
| Robustness check rejects 63% of the switches that pass the paired test | 93.59 / 147.96 | ablation `TD.levels.ng` (`robust_rejects` / `robust_checks`; counted only when the best mode differs from the incumbent and its gain is resolvable; `a9_mpc.py` around line 894) | S |
| Band only − nominal CI; gate only − nominal CI | [+11.4, +18.9]; [−5.6, +3.3] | `N.ablation.TD.contrasts.{A2,A1}.ci` | C |
| Calibrated − H\* replicated on F, V, TD_all | +7.37\*, +4.05\*, +8.30\* | `N.phase2.{F,V,TD_all}.ret.H1` | S |
| Calibrated − nominal MPC | +18.6 | `N.phase2.TD.ret.H3` | S |
| Compliant-image PSNR guarded vs nominal | 30.3 vs 32.1 dB | `N.decomp["ablation\|TD\|N5x2\|ng-nc"].modes.psnr_ok` | P |
| Guarded raises recall over H\* in 4 of 7 heuristic settings | ref, K7, B0.5, B2 (both CIs) | Table II recall column | S |
| ng − H\* gain from fewer below-Qmin deliveries (TD / V / F) | viol +7.9 / +7.0 / +10.7 | `decomp` `ng-H*` `diff.viol` | P |
| Same recall-split pattern on V and F | Fig. 3c | `decomp` `ng-nc` `recall_split` for V, F | P |
| Robustness check = gain > 1 s.e. also at scales ℓ and h | | `a9_mpc._resolvable` (margin_z = 1) | — |
| Table I parameters | | `env.py`, `env_la`, `ax2_eval.py` docstring, protocol | — |

**Sources** (all stored results; no new simulation):
- `N` = `runs/uav_semcom/paperA_fd/paper/paper_numbers.json`, written by
  `swarmbridge/experiments/uav_semcom/paperA_fd_numbers.py` from the phase summaries, `supplement_extras.json` and
  `posthoc/{decomp,r5}.json`.
- `table_fd.tex` (Table 1) and Fig. 2 are generated from `N` (`paperA_fd_figures.py`). Fig. 1 is a diagram drawn by
  the same script; its flow was checked against `ax2_eval.py` and `a9_mpc.py`.
- Paired 95% t-CIs over 100 environment seeds. Recall CIs are seed-level, plus a video-cluster bootstrap (B 2000).

## Evidence status

| Status | Meaning |
|---|---|
| **C** | Pre-registered confirmatory (Holm within the family) |
| **S** | Pre-registered supporting / replication (unadjusted) |
| **P** | Post-hoc, done after the consolidated report (`paperA-fd-posthoc-result.md`) |

## Model and method (Sections 2–3)

| Claim | Source |
|---|---|
| Utility (1): 0.5 / 0.5 weights, span 15 dB, −1 for a violation, −1 for a drop | `env.py` `utility`, `_drop`; `q_span=15`, `xi_q = xi_d = 0.5` |
| Return (2): energy 0.02 per kJ, time 0.3 per UAV-second | `env.py` `w_energy=0.02`; `w_time=0.3` from `a5_common.REF`. Identity checked on all 27,600 episodes: the time residual equals 0.3·K·mission_s within 2e-13 (`fd_posthoc.py verify`) |
| UAV active until its queue is empty | `env.py` step (6): a sortie ends when the survey is done and the queue is empty |
| LA parameters (2-block report delay, 1 dB error, OLLA 10%, logistic BLER, ARQ, slot-long interference) | `env_la` (unchanged since A-X5 / A-X6; described in the earlier letter) |
| Rate models, tracker, gate (20 s airtime), band formula | `a9_mpc.py` (`CalRates`, `RateErrTracker`, `MIN_AIR = 20`); band rule as in the previous letter |
| Nominal model = genie CQI rate × (1 − BLER target) on statistical fading | `ax2_eval.py` docstring (`+mpcnc`) |
| Guarded MPC = nominal model + the same gate and band; gate-only = band fixed [1, 1]; band-only = no cold-start gate | `ax2_eval.py` lines 160–170 (`+mpcng`, `+mpcngg`, `+mpcngb`, `+mpcngh1`, `+mpcngh0`) |
| Screen at scales [1, ℓ] one image ahead; N = 8; H = 4; 1-s.e. paired test; endpoint check at ℓ and h; 50 ms budget | `ax2_eval.py` `+mpc` docstring (np-any, N 8, H 4, Z 1, H1 1, 50 ms); `a9_mpc.py` `screen` / selective controller |
| 14 modes | `a5_common.MODES` |

## Setup (Section 4)

| Claim | Source |
|---|---|
| Pools: training 1,523, validation 374, TD 582 / 61 sequences, F 168 / 18, V 548; ≤ 30 frames per sequence; 49 TD sequences and V used in development; F used for detector checkpoint selection | `paperA-full-data-protocol.md` §1–2, §8 |
| H\*: best of 22 rules (AVIF-20 at the reference) | `phase2/tune.json` (22 candidates, `H`) |
| Re-tuned H\* per setting | `phase4/tune.json` |
| RL: 3 seeds × 2M decisions; checkpoint chosen on validation; fixed hyper-parameters | `paperA-fd-phase3-result.md` §1; `phase3/selection.json` |
| Holm families 2 / 4 / 12 / 6 | Phase 2, ablation, Phase 4 and Phase 3 pre-registrations |
| 27,600 episodes | `posthoc/run_verify.log` |

## Results (Section 5, Table 1, Fig. 2)

| Text | Value | Source | Status |
|---|---|---|---|
| Nominal MPC − H\* (TD / V / F) | −10.6 [−14.8, −6.4] / −6.8 / −8.6 | `N.ablation.{TD,V,F}.contrasts["nc-H"].ci` (= `decomp.json` `ablation\|*\|N5x2\|nc-H*`) | P (a derived difference of two registered arms) |
| Median bias −0.38 (tracker in the guarded arm, TD) | −0.384 | `supplement_extras.json` `ablation\|test\|TD\|N5x2\|h=fixed_avif_20+mpcng` `bias_q[1]` | S |
| Nominal MPC: 180 switches per episode; drops 8.1% vs 2.1% | 180.0; 0.0807 / 0.0214 | `ablation/summary_ablation_test.json` `TD.levels.{nc,H}` | S |
| Calibrated − H\* (reference) | +7.3 [+5.0, +9.5] | `N.phase2.TD.ret.H1` | C |
| Guarded − H\* (ablation seeds) | +7.4 [+5.1, +9.8] | `N.ablation.TD.contrasts.A8.ci` | S |
| Band only − nominal | +15.2 (Holm); V +13.4, F +16.2 | `A2` | C (TD), S (V / F) |
| Gate only − nominal | −1.2 | `A1` | C (not rejected) |
| Gate + band − band (TD / V / F) | +2.9 / −4.3 / −2.1 | `A6` | S |
| H = 1 − H = 0 | +17 to +21 (20.4 / 17.1 / 20.8) | `A9` | S |
| H = 4 − H = 1 (TD / V / F) | +5.1 (Holm) / −2.5 / +0.4 | `A4` | C (TD), S |
| Guarded − H\* per setting: 5 / 6 rejected, K = 3 n.s. | Table 1 | `N.phase4.TD.<cfg>.G1` | C |
| Learned incumbents: X + guarded − X | +7.4 / +19.0 / +25.7 | `N.phase3.TD.<x>.R1` | C |
| Calibrated − guarded, reference | −0.1 [−2.0, +1.8] | `N.phase2.TD.ret.H3b` | C (not rejected) |
| Calibrated − guarded, V / F (ablation replications) | +2.5 / +2.6 | `A7` V / F | S |
| Calibrated − guarded per setting: rejected at M = 3, M = 4 | Table 1 | `N.phase4.TD.<cfg>.G2` | C |
| Calibrated − guarded, learned | +3.1 / +4.7 / +6.0 | `N.phase3.TD.<x>.R2` | C |
| X − H\* (PPO / QR-DQN / DSAC) | −3.0 [−4.9, −1.0] / −22.5 / −34.1 | `N.phase3.TD.<x>.R5.alone.ret` | S (R5 descriptive) |
| PPO + cal − H\* + cal | +0.3 [−1.6, +2.1] | `N.phase3.TD.ppofd.R5.calibrated.ret` | P |
| Recall levels nominal / guarded / calibrated / H\* | 40.5 / 38.7 / 37.6 / 37.1 | `N.ablation.TD.levels.<arm>.recall` | S |
| Nominal return 18.1 below guarded | 18.05 | `A3` / `decomp` `ng-nc` `ret` | S |
| Calibration lowers recall in every row: −0.4 to −1.1 | Table 1 | `A7` recall, `G2` recall, `R2` recall | S |
| Decomposition guarded − nominal: payload 194 vs 303 kbit; drops +15.8; below-Qmin −15.7; SCE +9.7; delay −1.2 s; time +8.1 (mission −5.4 s); recall −1.8 = −1.2 (both compliant) − 0.6 (set) | as stated | `N.decomp["ablation\|TD\|N5x2\|ng-nc"]` (`modes.kbit`, `diff.*`, `recall_split`) | P |
| 53 contrasts: 44 with a positive mean return difference, 41 of them with smaller payloads; the exceptions are ng − H\* in TD / V / F | counts | computed from `posthoc/decomp.json` (see `paperA-fd-writing-report.md`) | P |
| Zero-shot PPO: recall +3.1, return −4.9 vs H\*; larger payloads (234 vs 175 kbit) | | `N.zeroshot_R5.ppola.alone`; payload from the post-hoc ad-hoc check (`paperA-fd-posthoc-result.md` §2.4) | P |
| Timing: sequential p99 103–229 ms, 42–82% of slots over 50 ms (all H ≥ 1 planner arms, all pools) | 102.8–228.9; 0.418–0.820 | ablation / Phase 4 / Phase 3 summary `levels` (`ms_full_sum_p99`, `full_sum_over50`) | S |
| Timing: parallel ≤ 4.1% (K ≤ 5, M ≤ 3); 31% (K = 7) and 41% (M = 4) for the guarded MPC on TD; scheduler about 30 ms | 0.041; 0.307; 0.411; 28.6 / 30.9 ms | Phase 4 `levels.<cfg>.ng` (`full_max_over50`, `ms_slot_p99`) | S |
| H = 0: 16 ms | 15.9 | ablation `levels.h0.ms_full_sum_p99` | S |
| Not bit-reproducible | | `paperA-fd-phase3-result.md` (resume note); E7 rerun (earlier letter) | — |

## Table 1 columns

| Column | Source |
|---|---|
| Guarded − base, heuristic rows | Reference: ablation A8 (supporting). Other rows: Phase 4 G1 (confirmatory) |
| Calibrated − guarded, heuristic rows | Reference: A7. Other rows: G2 |
| Learned rows | R1 and R2 (confirmatory) |
| Recall | Each contrast's `recall.seed` / `recall.cluster`. `*` = both CIs exclude 0; `°` = seed-level only |
| Timing | `levels.ng` (or `X+mpcng`): `ms_full_sum_p99` / `full_sum_over50` / `full_max_over50` |
