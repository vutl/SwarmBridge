# Paper A method letter: numerical claims and their sources

The manuscript is `main.tex` (method-letter rewrite of 2026-11-23, 4 pages).

Numbers come from three files under `runs/uav_semcom/paperA_r2/`:

| File | Written by | Holds |
|---|---|---|
| `letter_numbers.json` | `paperA_letter_numbers.py` | Table 1, E5, E2c copy, E7 copy, latency |
| `summary_e7.json` | `paperA_e7_summary.py` | E7 |
| `paper_numbers.json` | `paperA_paper_numbers.py` | old Table 1; used only for the motivation sentence |

The manuscript shows the results as figures. Fig. 1 is drawn by `paperA_letter_figures.py`; Figs. 2–4 are IEEE-style versions drawn by `paperA_ieee_figures.py` (`fig_sweeps.pdf`, `fig_phy.pdf`, `fig_rl.pdf`) from the same data as listed below; Fig. 4a also plots the validation curves in `summary_e7.json["val_curves"]`.
- Fig. 2 `figs/fig_lines.pdf`: return per image (level ÷ 52K) of H\*, +RobustMPC-style, +nominal and +calibrated MPC.
  - x-axes: bandwidth (BW rows plus NET:N5x2), K (NET N3x2/N5x2/N7x2) and M (NET N5x2/N5x3/N5x4).
  - Data: `T1[...]["level"][arm]["ret"]`; bands from `["ret_ci"]`.
  - Panel (d): `letter_numbers.json["E5"][fb]["level"]`.
- Fig. 3 `figs/fig_phy.pdf`: the PHY and TD rows. `cal_minus_nom`, `cal_minus_rh` and `cal_minus_H` (bars + CI), and `recall_cal_minus_H`.
- Fig. 4 `figs/fig_e7bars.pdf`: E7 B1, B2, B1_recall and B2_recall, for test and test-dev (`summary_e7.json` `B` / `H_ref`).
  - `B2_recall` was added on 23/11 from the same delivery files; no new runs.
- Figures drawn earlier but no longer in the PDF: `fig_robust.pdf` (forest plot), `fig_e7.pdf` (dot plot) and `fig_recall.pdf`.

The tables `table_robust.tex` and `table_e7.tex` (from `paperA_letter_numbers.py`) carry the same numbers. They are
not in the PDF.

**Notation:**
- `T1[g:c]` = `letter_numbers.json["table1"]["<group>:<config>"]`.
- `cal_minus_nom`, `cal_minus_rh`, `cal_minus_H` and `recall_cal_minus_H` are `[mean, CI low, CI high]` (paired 95% t-CI over 100 environment seeds).
- `n.s.` means the CI includes 0.

**Seed sets and incumbents:**
- PHY rows: E1, seeds 9900–9999.
- TD rows: E3, seeds 12000–12099.
- NET rows: E4, seeds 13000–13099.
- BW rows: E6, seeds 13000–13099.
- E7: test seeds 13000–13099, test-dev seeds 14000–14099.
- H\*: frozen fixed_mbt2018-mean_q3 (AVIF-50 under Shannon) in PHY/TD rows. Re-tuned in NET rows (q1 at K=7) and BW rows (AVIF-20 at 0.5 MHz).

## Abstract

| Claim | Source |
|---|---|
| beats nominal rates in all 12 evaluated LA configurations | `letter_numbers.json["counts"]["cal_minus_nom"]` = pos 12 / of 12 (LA rows = all rows except PHY:S and TD:S) |
| beats a RobustMPC-style predictor in 11 | `counts["cal_minus_rh"]` = pos 11 / 12; the exception is NET:N5x3, +1.66 [−0.16, +3.49] |
| improves policies from three RL families | `summary_e7.json` test `B_pass` = 6/6 |
| gains over the policy without MPC depend on the configuration | `counts["cal_minus_H"]` = 10/12; NET:N5x2 +2.66 n.s. vs PHY:L +8.25 |
| higher return is not always higher recall | `counts["recall_cal_minus_H"]` = 4 positive, 8 negative |

## Introduction

| Claim | Source |
|---|---|
| PPO +2.3 under Shannon | `paper_numbers.json["S"]["A_return"]` = +2.25 (PPO-S − T:best, same slot rule, seeds 9900–9999) |
| trails by 22 to 60 under LA | `paper_numbers.json` `A_return` for L (−23.4), L_fade (−39.0), L_bursty (−22.1), L_badfb (−37.9) and TD-L (−59.7) |
| 14 configurations | the 14 rows of Table 1 (12 LA + 2 Shannon) |

## System model and method (design facts)

Each design fact is checked against code.

| Fact | Code |
|---|---|
| 52 images per UAV; 14 modes | `a5_common`, regime `K5_la` |
| 2 dB SNR bins; 20 s minimum observations; cold start; observations pooled over UAVs; slot-weighted observation | `a9_mpc.CalRates` (`BIN_DB`, `MIN_AIR`, `persistent=False`, `slot_weight=True`); bin keys depend on SNR only |
| error tracker: bias and standard error; gate (`stats` is None, so the incumbent is kept) | `a9_mpc.RateErrTracker.stats`; `ax2_eval.episode` (`gated`) |
| band ℓ = max(min(1, 1+b) − 2σ, 0.05), h = max(1, 1+b) + 2σ | `ax2_eval.episode` |
| screen: horizon 1, scales [1, ℓ]; candidates = best nominal + best pessimistic; trigger "any" | `MPCMode.selective(cand="np", trig="any", h1=1)` |
| full plan: N = 8, H = 4, 40 slots, causal rollout, switch threshold margin_z = 1, robustness check at scales [ℓ, h] | `MPCMode("cal", 8, 4, ..., rollout="causal", margin_z=1.0)`; `S_SLOTS = 40`; `MPCMode.__call__` |
| 50 ms budget including incumbent inference, 3 ms margin | `a9_mpc.Deadline(50.0, margin_ms=3.0)`, started before the incumbent's inference |
| nominal MPC: LA mapping on fading samples, no gate, band [1, 1] | `ax2_eval` `nocal` branch; `MPCMode._rate` (phy "la") |
| RobustMPC-style: harmonic mean of the last 5 rates / (1 + max relative error of the last 5) | `a9_mpc.HarmRates` |
| QR-DQN and Discrete SAC hyper-parameters | `offpolicy_rl.py`, `offpolicy_train.py`; `runs/uav_semcom/paperA_rl/selection.json` |

## Results: calibration versus matched planners

| Claim | Source |
|---|---|
| nominal: 5.2 to 36.4 in all 12 LA rows | min TD:L +5.24, max PHY:L_fade +36.43 (`cal_minus_nom`) |
| Shannon: +0.6 n.s. (test), +1.8 (test-dev) | PHY:S +0.57 [−0.92, +2.07]; TD:S +1.79 [+0.34, +3.23] |
| RobustMPC-style 11/12; K=5, M=3: +1.7 n.s. | `counts`; NET:N5x3 `cal_minus_rh` +1.66 [−0.16, +3.49] |
| RobustMPC-style within 0.7 of H\* alone (NET and BW rows) | `cal_minus_H − cal_minus_rh` per row: −0.03, −0.59, +0.55, −0.17, −0.31, −0.10, −0.66 |
| ladder: no single step; pooling +3.9 to +9.0, significant in 4 of 7 | `summary_e2c.json` "pool UAVs": L_fade +3.9*, L_bursty +4.0*, L_badfb +9.0*, TD-S +4.3*; others n.s. |
| SNR binning: no difference in the ladder | `summary_e2c.json` "SNR bins (cal - pl)": all 7 CIs include 0 |
| unbinned model lower in all 5 networks (−0.7 to −1.7; 2 significant) | `summary_e4.log` / `paperA-e4-result.md`: pooled − cal = −1.7*, −1.5, −1.4, −0.7, −1.2* |

## Results: gains over the incumbent alone

| Claim | Source |
|---|---|
| 10 of 12 LA rows | `counts["cal_minus_H"]`; the exceptions are NET:N5x2 and NET:N5x3 |
| +8.2 (seeds 9900–9999) | PHY:L `cal_minus_H` +8.25 [+4.83, +11.66] |
| +2.7 n.s. (seeds 13000–13099) | NET:N5x2 +2.66 [−0.52, +5.84] |
| +16.4 (seeds 14000–14099) | `summary_e7.json` testdev `H_ref.B1` +16.36 [+11.66, +21.06] |
| pre-registered rules failed (3/5, 2/3); nominal advantage held in all | `summary_e4.json["pass"]` (H1 3/5, H3 5/5); `summary_e6.json["pass"]` (H1 2/3, H3 3/3) |

## Results: delayed or lost reports (E5)

| Claim | Source |
|---|---|
| +18.1 to +21.3 vs nominal, all 5 settings | `letter_numbers.json["E5"][fb]["minus_nominal"]`: fb10l0 +18.11 … fb1l0 +21.31, all CIs > 0 |
| no difference from immediate reports detected | `E5[fb]["minus_immediate"]`: all 5 CIs include 0 |
| untested cases | `paperA-e4-result.md`, section E5 |

## Results: RL families (E7, `summary_e7.json`)

| Claim | Source |
|---|---|
| Fig. 3 | `test.levels` and `testdev.levels` (means and `ret_per_seed` ticks) |
| B1 in text, test: +6.2±1.3, +23.9±1.9, +29.2±1.9 | `test.B[...].B1`; ± = half-width of the CI |
| B2 in text, test: +20.2±2.7, +10.6±3.4, +9.0±2.9 | `test.B[...].B2` |
| 6/6 on test | `test.B_pass` = 6 |
| 5/6 on test-dev; Discrete SAC B2 +1.2 n.s. | `testdev.B_pass` = 5; `testdev.B["Discrete SAC"].B2` = +1.19 [−1.42, +3.81] |
| per seed: B1 18/18; B2 not significant in 4 (QR-DQN s2 on both pools; Discrete SAC s0 and s2 on test-dev) | `test.B_per_seed`, `testdev.B_per_seed` |
| weaker than PPO-LA | `test.A["QR-DQN"]["X - PPO-LA"]` −30.85*; Discrete SAC −38.47* (test-dev −22.70*, −37.42*) |
| X+cal − H\*+cal on test: −13.6, −15.9 | `test.A[...]["X+cal - H*+cal"]`: −13.57*, −15.89* |
| PPO-LA − H\*: −4.4 (test), +13.7 (test-dev) | `test.A["PPO-LA"]["X - H*"]` −4.38*; `testdev` +13.68* |
| with MPC: −0.4 n.s. (test), +8.1 (test-dev) | `X+cal - H*+cal`: −0.43 n.s.; +8.05* |

## Results: return and recall

| Claim | Source |
|---|---|
| H\* incumbent: recall lower in 8/12 LA rows (−0.3 to −1.2), higher in 4 (+0.9 to +2.6) | `recall_cal_minus_H` per row: negative −0.31 … −1.21; positive +0.90, +1.08, +1.90, +2.58 |
| learned incumbents: B1 recall +0.6 to +6.1 | `B[...].B1_recall` on test (+0.94, +4.53, +6.13) and test-dev (+0.58, +2.57, +4.14) |
| nominal MPC has higher recall than cal for QR-DQN and Discrete SAC, by 3.5 and 4.4 points (Fig. 4b) | `test.levels`: QR-DQN X+nom 47.1 vs X+cal 43.6; Discrete SAC 47.2 vs 42.8 |

## Results: decision time and reproducibility

| Claim | Source |
|---|---|
| K=5, M=2: per-episode p99 40–64 ms | `letter_numbers.json["E1_latency"]`: p99_min 40.4, p99_max 64.0 (all +mpc arms of E1 and E3) |
| at most 0.5% of slots over 50 ms on average | per-configuration means: E1/E3 rows ≤ 0.1% (`over50_cal`), NET:N5x2 0.5%, BW rows 0.0%, E7 ≤ 0.1% (`summary_e7.json` levels `int_over50`) |
| K=7 / M=4: scheduler p99 35–38 ms; 28% / 40% of slots over 50 ms | `paperA-e4-result.md` (scheduler p99 34.9 / 38.0 ms); `T1[NET:N7x2]` and `T1[NET:N5x4]` `over50_cal` = 0.277 / 0.402 |
| rerun changed 16/100 episodes, mean −0.4 [−1.3, +0.5] | `paperA-e7-result.md` section 7: E7 H\*+cal vs E4 N5x2 H\*+cal, same seeds |

## Changes from the 20/11 version

- **Framing:** from "wireless model and task metric disagree" (evaluation study) to a method letter. The PPO-S Shannon/LA reversal remains as one motivating sentence.
- **Removed claims:**
  - "the mechanism behind the gain over RobustMPC-style is not resolved": now covered by the E2c ladder;
  - "ideal feedback only": E5 now relaxes this partly, and the untested cases are listed;
  - "only PPO": E7 adds QR-DQN and Discrete SAC.
- **Removed references:** hu2025 and zhou2026 (space); pedersen2007 (never verified).
- **Added references:** sun2016 (CS2P), yan2020 (Fugu), akhtar2018 (Oboe), dabney2018 (QR-DQN), christodoulou2019 (Discrete SAC). Their page numbers should be checked before submission.
