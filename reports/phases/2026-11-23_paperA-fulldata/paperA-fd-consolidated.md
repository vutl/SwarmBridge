# Paper A extended evaluation: consolidated claims / evidence matrix and recommendation (25/11)

**Inputs:**
- Phase 1 (profile).
- Phase 2: `paperA-fd-phase2-result.md`.
- Ablation: `paperA-fd-ablation-result.md`.
- Phase 4: `paperA-fd-phase4-result.md`.
- Phase 3: `paperA-fd-phase3-result.md`.
- Novelty audit: `paperA-novelty-audit-ea-smpc.md`.

**Completion checks:** `runs/uav_semcom/paperA_fd/supplement_{coverage,ckpts,extras}.log`.
- 25,200 / 25,200 new evaluation episodes, with the expected pools, seeds, arms, configurations, checkpoint ids and
  52 K deliveries per episode.
- 9 / 9 checkpoints match `selection.json` (sha256).

**No new experiment was run for this report.**

**Notation:**
- H\* = the tuned heuristic incumbent.
- ng = the error-aware selective MPC (tracker gate + robustness band, H = 4) with the **nominal** rate model.
- cal = the same design with the **ACK-calibrated** model.
- nc = the unguarded nominal planner.
- An asterisk marks a 95% CI that excludes 0. "Holm" means rejected within its pre-registered family.

## 1. Claims / evidence matrix

### A. Method effectiveness (does the error-aware layer help?)

| Claim | Evidence | Status |
|---|---|---|
| ng (or cal) improves return over the tuned heuristic at the reference setting | Phase 2: H1 cal − H\* +7.28 (Holm), post-hoc ng − H\* +7.38; replicated on F / V / TD_all. Ablation A8 on fresh seeds: ng − H\* +7.44\* / +2.28\* / +5.42\* (TD / V / F) | **Supported** |
| …in other network sizes and bandwidths | Phase 4 G1: 5 / 6 configurations Holm-rejected positive (N7x2, N5x3, N5x4, B0.5, B2); N3x2 not detected on TD (+0.63), +2.30\* on F | **Supported, 5 / 6** |
| …on top of learned incumbents | Phase 3 R1: 3 / 3 families Holm-rejected positive (+7.4 PPO, +19.0 QR-DQN, +25.7 DSAC); all replications and all per-seed contrasts positive; zero-shot also positive | **Supported** |
| The return gain carries over to detection recall | vs H\*: TD absolute recall 38.69 vs 37.12 (ablation). Phase 4 G1 recall: +1.2 to +2.3\* in N7x2 / B0.5 / B2; −0.2 to −0.4\* in N5x3 / N5x4; n.s. in N3x2. vs learned incumbents: +2.4 to +7.0\* | **Mixed**: positive vs weak incumbents, small or negative vs H\* in 3 of 6 configurations |
| Return and recall agree on ranking | The **unguarded nominal planner nc has the highest recall** in the reference setting (40.5% vs ng 38.7%, cal 37.6%), in N7x2 / B0.5 / B2 and with every learned incumbent, yet its return is lower than ng by 9 to 30 | **Not supported**: the return used for every confirmatory test does not rank arms the same way as recall |

### B. Mechanism necessity (ablation; reference setting)

| Claim | Evidence | Status |
|---|---|---|
| The robustness band is necessary | A2 band − nc +15.15 (Holm), +13.4\* V, +16.2\* F | **Supported** |
| The tracker gate is necessary | A1 gate − nc −1.19 (n.s.; V −2.9, F +0.6), with lower recall (−0.8 to −1.4\*). A6 adding the gate to the band: +2.9\* TD, **−4.3\* V**, −2.1 F | **Not supported**: the band carries the gain |
| A 4-step horizon is needed | A4 H4 − H1 +5.05 (Holm) TD, but **−2.51\* V**, +0.38 F. H1 costs about 70% of H4's compute | **Not supported** (does not replicate) |
| Some look-ahead is needed | A4b H4 − H0 +25.5 (Holm), +14.6\*, +21.2\*. A9 H1 − H0 +17 to +21 everywhere | **Supported** (one anticipated arrival) |
| ACK calibration adds to the matched design | Reference: Phase 2 H3b −0.10 (TD, n.s.); A7 −0.75 (TD, n.s.); F / V +2.5 to +4.0\*. Phase 4 G2: 2 / 6 Holm (N5x3 +2.8, N5x4 +1.6), all estimates positive. Phase 3 R2: **3 / 3 Holm** (+3.1, +4.7, +6.0) | **Regime- and incumbent-dependent**: return gain with learned incumbents and two configurations, not at the reference heuristic |
| Calibration improves recall | **Lower recall in every setting** with a detectable difference: −0.4 to −1.0\* (Phase 4 all 6, Phase 3 all 3, ablation) | **Contradicted** (consistently negative) |

### C. Wireless generality

| Claim | Evidence | Status |
|---|---|---|
| Benefit across K ∈ {3, 5, 7}, M ∈ {2, 3, 4}, bandwidth ∈ {0.5, 1, 2} MHz | G1 5 / 6; the benefit is smallest at K3 | **Supported for return**, with the K3 exception |
| Real-time feasibility (50 ms slot) | Sequential full-sum p99 is 103–229 ms in every H ≥ 1 planner arm and configuration. 44–82% of slots exceed 50 ms sequentially. With hypothetical parallel decisions, p99 is about 41–50 ms at K ≤ 5 / M ≤ 3, but **31–43% of slots exceed 50 ms at N7x2 and N5x4** (the scheduler alone is about 29–31 ms). Only the H0 variant fits sequentially (16 ms), and it is the worst planner. Simulated returns do **not** charge controller time | **Not supported** as a general claim. Supported only as "per-decision ≤ 50 ms with one core per decision at K ≤ 5, M ≤ 3" |

### D. Policy generality / RL

| Claim | Evidence | Status |
|---|---|---|
| The layer is incumbent-agnostic | R1 positive for PPO / QR-DQN / DSAC, retrained and zero-shot | **Supported** |
| Learned incumbents beat the tuned heuristic | R5: PPO −3.0\* (TD), −0.8 (TD_all, n.s.); QR-DQN about −22\*; DSAC about −34\*. Recall −2 to −10\*. With the layer: PPO n.s. on F / V / TD_all; others negative. **Post-hoc (§3 of the post-hoc report):** PPO + cal − H\* + cal: TD +0.27 [−1.58, +2.12], n.s. in all 4 pools; recall −0.73\* (cluster CI excludes 0) | **Not supported on return.** Recall: the retrained incumbents are at or below H\* (cluster bootstrap). **Correction (post-hoc §4): the zero-shot PPO-LA has HIGHER recall than H\* (+3.1, cluster CI [+1.6, +4.8]) with lower return (−4.9)** |
| Training-seed robustness | All 18 per-seed R1 / R2 TD contrasts positive. Spreads: R1 PPO 6.2–9.3, QR 15.7–24.8, DSAC 20.1–32.7 | Consistent in sign; magnitude varies; 3 seeds only |

### E. Novelty

| Claim | Evidence | Status |
|---|---|---|
| The algorithmic elements are new | Audit: MPC-based ABR / RobustMPC, error-aware gating and empirical calibration are prior art | **Not supported** |
| Defensible contribution | (i) The application to multi-UAV image delivery with codec / mode selection under scheduler-aware LA. (ii) The matched-design finding: a robustness band removes most of the nominal-vs-calibrated gap, and calibration then adds return only in some regimes / with learned incumbents, while lowering recall. (iii) The negative RL-vs-heuristic finding | Empirical / application contribution |

## 2. Positive, negative and inconclusive results (one line each)

**Positive:**
- ng > H\* (reference + 5 / 6 configurations).
- ng > every learned incumbent alone.
- band > nc.
- H ≥ 1 > H0.
- cal > ng with learned incumbents and in N5x3 / N5x4.

**Negative:**
- Gate alone ≈ nc, with lower recall.
- cal lowers recall everywhere.
- Learned incumbents < H\*.
- Sequential timing exceeds the slot in most slots.
- Parallel timing exceeds the slot at N7x2 / N5x4.
- The highest-recall arm is the unguarded nominal planner.

**Inconclusive:**
- H4 vs H1 (TD +, V −, F 0).
- cal vs ng with H\* at the reference (TD 0, F / V +).
- G1 at N3x2.
- Adding the gate to the band (TD +, V −, F 0).

## 3. The return / recall issue

**What the evidence shows:**
- Every confirmatory test is on episode return.
- The arms that raise return over nc (the band, the gate, calibration) **lower** Qmin-compliant recall relative to nc.
  nc's recall is higher than H\* by 3.4 points at the reference and by 6.5 points at B0.5.
- This repeats the A-X9 observation ("return not recall-aligned").
- The return combines delivery, quality and energy terms. Recall counts detected objects.
- The Qmin-compliance rate is about equal for nc and ng (0.866 vs 0.867 at the reference), so the return gap is not
  explained by compliance.
- **Post-hoc decomposition (added 25/11, [paperA-fd-posthoc-result.md](paperA-fd-posthoc-result.md) §2).** ng sends
  smaller payloads than nc (194 vs 303 kbit per compliant image).
  - **Return:** fewer drops (+15.8) and about 15.7 more below-Qmin deliveries roughly cancel, because each costs −1.
    The net gain comes from the SCE terms (+9.7) and from mission time (+8.1: 5.4 s shorter per UAV at 0.3 per
    UAV-second).
  - **Recall:** −1.8. About two thirds comes from a lower detection yield on the same compliant images (lower PSNR);
    one third from which images end up compliant.
  - **Generality:** the same direction holds in all 53 contrasts examined (pools, configurations, RL families; band,
    gate and calibration steps too).

**Consequence:** a paper claim such as "improves perception" cannot be based on the return results. Recall must be
reported as a separate outcome, with the trade-off stated.

## 4. Recommendation

**Recommended: (C) reposition Paper A as an application and empirical contribution.**

**Why C:**
1. **The method-level evidence is real but narrow.**
   - The layer reliably improves return over a tuned heuristic (reference + 5 / 6 configurations) and over every
     learned incumbent.
   - It is incumbent-agnostic. This is enough for an empirical claim.
2. **The mechanism claim behind the multi-step controller fails.**
   - The gain comes from the robustness band.
   - Neither the tracker gate nor H = 4 is shown to be necessary: A1 n.s., A6 and A4 do not replicate.
   - A paper presenting "error-aware selective multi-step MPC" as the contribution would be describing components that
     the pre-registered ablation does not support. **(A) is therefore not recommended.**
3. **The real-time claim fails as stated.**
   - Sequential execution overruns the 50 ms slot in about 60–80% of slots.
   - Parallel execution overruns at K = 7 and M = 4.
   - A controller claim would need a timing-aware simulation, which is not done here.
4. **No novel algorithmic element** (audit).
5. **Recall:**
   - Calibration lowers recall in every setting.
   - Return does not rank the arms the same way as recall does.
   - These are reportable empirical findings, not method wins.

**What a C paper states:**
- Under scheduler-aware LA and persistent interference, model-based mode switching needs a robustness check against
  rate-prediction error.
- With it, the nominal model performs about as well as an empirically calibrated one at the reference setting.
- Calibration helps learned incumbents and two larger-M configurations on return, but lowers recall.
- Learned mode-selection incumbents (PPO, QR-DQN, DSAC; mode selection only) do not beat a tuned heuristic.
- The evaluation is pre-registered across 3 learned families, 7 network configurations and 4 image pools.
- Sequential controller timing is reported as a limitation.

**About (B), a simpler short-horizon controller (band, H = 1, no gate):**
- The ablation points to it: A2 band − nc +15; H1 is within ±5 of H4 at about 70% of the compute; it has no gate and
  an H1 timing.
- But building it is a new study: new arm combinations and new seeds, chosen after seeing these results.
- It would need its own pre-registration. It cannot be claimed from the present data, and it would still not fit
  sequential timing (H1 p99 about 112 ms, 49% of slots over 50 ms).
- Recommended only as a separately registered follow-up if a method paper is still wanted. It is not needed for C.

**About (D):** not needed. The evidence base (25,200 new episodes, all pre-registered and complete) supports a
scoped empirical paper.

**About the manuscript:** no rewrite until this report is reviewed, as agreed.

## 5. Execution record

| Time | Event |
|---|---|
| 24/11 14:59 | Chain launched. **WSL VM stopped at 15:01**; no output was written. It was noticed at 18:18 (the earlier status report had relied on the log line, not a process check) |
| 24/11 18:19 | Relaunched unchanged (899eed2c) |
| 24/11 | Ablation 18:51, Phase 4 20:01, zero-shot 20:50 complete |
| 25/11 10:13 | Training complete (9 runs) |
| 25/11 ~12:14 | **Host power loss** during retrained pool V; TD and F were already complete |
| 25/11 12:21 | Resumed with `--skip-existing` (000251c7, execution-only; completed pools kept, not re-simulated) |
| 25/11 13:59 | Chain exit 0 |

Post-chain coverage, checkpoint and extras checks all passed. Abort *categories* were never logged; only the counts
are reported.
