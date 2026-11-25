# SwarmBridge: handoff brief for a new reviewer session (25/11/2026)

**For:** a new ChatGPT session that takes over as the external reviewer. The previous session ran out of length.
**Read this file first.** It is self-contained. For details, use the files linked in section 9.
**Repo (public):** https://github.com/vutl/SwarmBridge (branch `main`). Latest relevant commit: `bc4d2736`.

## 0. Roles and rules

**Roles:**
- **Student:** a Master's student at UCLab. They make the decisions.
- **Claude Code:** runs the experiments and writes the reports in the repo.
- **You:** the external reviewer. You design and review protocols, check claims and recommend decisions.

**Rules that have been agreed:**
1. **Check before you assert.** Before saying "X was never tried" or proposing a direction, check
   `reports/HISTORY.md` and the phase reports. Earlier reviews made factual errors of exactly this kind (HISTORY §6).
2. **Pre-register confirmatory tests.** Analysis code is committed before a run; Holm correction applies within a
   pre-registered family. Post-hoc analyses are labelled as post-hoc.
3. **Report numbers neutrally.** Never explain a result by saying the scenario "favours heuristics".
4. **Keep negative results.** Corrections to earlier conclusions go in the corrections table (`reports/README.md` §3).
5. **The student's goal** is a publishable paper. The original ambition was a Q1 paper with an algorithmic
   contribution (see §1).

## 1. Project in one paragraph

**Original thesis (29/10):** multi-UAV + semantic communication + swarm intelligence. Can a UAV swarm send only the
complementary evidence that reduces the receiver's task uncertainty?

**How the problem narrowed, over 48 review rounds:**
- Cooperation was dropped (06/11), then movement was frozen, then the scheduler was frozen.
- What remained is **Paper A:** independent image offloading from K UAVs to a base station. A controller chooses a
  compression mode for each image under realistic link adaptation.
- **Paper 2** (cooperative perception on UAV3D + realistic LA + receiver state) is designed but not started. It waits
  until Paper A is submitted.
- **Paper B** (JEPA semantic codec) is paused, because inter-frame video codecs win at ≤ 12 kbit.

## 2. Paper A setup (what the simulator does)

**Network:**
- K UAVs (default 5) survey disjoint sectors at 100 m altitude. Each UAV offloads 52 images per episode to a base
  station.
- The UAVs share M channels (default 2) and B = 1 MHz at 2.4 GHz.
- A joint-assignment (JA) scheduler enumerates all (M+1)^K channel assignments each 1 s slot.

**PHY (block-level LA):**
- Each 0.1 s block uses one LTE CQI MCS. It is chosen from an SNR report delayed by 2 blocks with 1 dB error, plus an
  outer-loop offset (10% BLER target).
- Bits count only when ACKed. Failed blocks are retransmitted (ARQ).
- Co-channel interference lasts for the whole slot.
- All LA parameters are assumptions, not measurements.

**Images:**
- VisDrone frames, each sent in one of **14 modes**: AVIF, the MBT2018 learned codec and JPEG2000 at several
  qualities, a thumbnail, and 3 generative modes.
- Each mode has a measured payload, PSNR and encode time.
- Each image has a deadline D_max ~ U(4, 12) s and a quality floor Q_min ~ U(22, 30) dB.

**Return (the confirmatory metric):**
- Per image: u = 0.5·clip((q − Q_min)/15, −1, 1) + 0.5·max(D_max − D, 0)/D_max − 1[q < Q_min or D > D_max].
- A dropped image scores −1.
- Return = the sum of the utilities, minus energy and mission-time costs.

**Recall (the secondary task metric):** YOLO11s recall at IoU ≥ 0.5, over images delivered on time with PSNR ≥ Q_min.

**Incumbents (the mode policy without the layer):**
- **H\***: the best of 22 fixed or backlog-budget rules, tuned on training seeds.
- **PPO**, **QR-DQN** and **Discrete SAC**: each trained with 3 seeds × 2M decisions.

**The method layer (EA-SMPC: "error-aware selective MPC"):**
- For each image, the MPC may **override** the incumbent's mode.
- It plans over rate paths sampled from a rate model. The model is either **nominal** (the statistical channel model)
  or **ACK-calibrated** (empirical per-SNR-bin rates from ACKed bits; cold start each episode).
- Two guards:
  - **Tracker gate:** no planning until 20 s of prediction-error data exists.
  - **Robustness band [ℓ, h]:** built from the tracked bias ± 2 SE. A switch must remain better with all paths scaled
    by ℓ and by h.
- **Horizon:** H = 4 future images by default.
- **Budget:** 50 ms per decision.

**Arm names used below:**

| Name | Meaning |
|---|---|
| nc | Unguarded nominal planner |
| ng | Nominal model + gate + band, H = 4 (the "matched design") |
| cal | ACK-calibrated model + gate + band, H = 4 |
| gate, band | ng with only one of the two guards |
| h1, h0 | ng with H = 1 or H = 0 |

## 3. Timeline of Paper A claims (short)

| Date | Version | Main claim | What happened to it |
|---|---|---|---|
| 18–20/11 | Evaluation letter | Realistic PHY reverses RL-vs-heuristic rankings; return and recall disagree | Replaced by a method framing |
| 23/11 | Method letter "ACK-calibrated selective MPC" (`paper/paperA/main.tex`, **current file**) | Calibrated > nominal in 12/12 LA configurations; > RobustMPC-style in 11/12; improves 3 RL families | **The comparison was confounded.** "Nominal" there was the *unguarded* planner. Against a nominal planner with the same gate + band, the difference vanishes (Phase 2 H3b) |
| 23–25/11 | Extended full-data evaluation (protocol `fd_v1`) | Pre-registered re-test on new data with a matched design, an ablation, network sizes and retrained RL | **Done** (section 4) |

**Data in fd_v1:**
- Sequence-disjoint pools, at most 30 frames per sequence:
  - `rl_train` (1,523 frames) and `rl_val` (374) from DET-train;
  - **TD** = DET-test-dev, 582 frames, 61 sequences: the primary test pool. 49 of its sequences were seen in earlier
    rounds.
  - **F** = YOLO hold-out, 168 frames: method-unseen, but used to select the YOLO checkpoint.
  - **V** = DET-val, 548 frames: historical, used in development.
  - **TD_all** = test-dev uncapped, 1,610 frames.
- Statistics: 100 environment seeds per pool; paired 95% t-CI; Holm within each family; recall at seed level plus a
  video-cluster bootstrap.

## 4. Full-data results (25,200 new episodes; coverage and checkpoint checks pass)

An asterisk (\*) marks a 95% CI that excludes 0. Values are return differences on TD unless stated.

**Phase 2 (reference: K5, M2, 1 MHz):**
- cal − H\* = +7.28 (Holm-rejected); it replicates on F, V and TD_all.
- **cal − ng = −0.10 [−1.95, +1.76], n.s.** (F +3.97\*, V +1.94, TD_all +0.16).
- cal − nc = +18.6\*.
- **Conclusion:** the gate + band design, not the calibration, carries the gain at the reference.

**Ablation (fresh seeds; Holm over 4):**

| ID | Contrast | TD | V | F | Result |
|---|---|---|---|---|---|
| A2 | band − nc | +15.15 | +13.4\* | +16.2\* | Holm-rejected; **the band is necessary** |
| A1 | gate − nc | −1.19 | — | — | n.s.; recall −0.8\* |
| A6 | gate + band − band | +2.9\* | −4.3\* | −2.1 | **The gate is not shown necessary** |
| A4 | H4 − H1 | +5.05 | −2.51\* | +0.38 | Holm-rejected on TD; **does not replicate** |
| A4b / A9 | H4 − H0 / H1 − H0 | +17 to +25 everywhere | | | One look-ahead image is needed |
| A7 | cal − ng | −0.75 | +2.5\* | +2.6\* | n.s. on TD |
| A8 | ng − H\* | +7.44\* | +2.28\* | +5.42\* | |

**Phase 4 (network size and bandwidth; H\* re-tuned per configuration; Holm over 12):**

| Configuration | G1 ng − H\* | G2 cal − ng |
|---|---|---|
| N3x2 | +0.63, n.s. (F +2.30\*) | +1.34, not rejected after Holm |
| N7x2 | +6.78, Holm-rejected | not rejected |
| N5x3 | +5.19, Holm-rejected | +2.79, Holm-rejected |
| N5x4 | +2.95, Holm-rejected | +1.63, Holm-rejected |
| B 0.5 MHz | +7.42, Holm-rejected | not rejected |
| B 2 MHz | +4.90, Holm-rejected | not rejected |

- G1 is rejected in 5 of 6 configurations; G2 in 2 of 6. All G2 point estimates are positive (+0.2 to +2.8).

**Phase 3 (retrained learned incumbents; Holm over 6):**
- R1, X + ng − X: PPO +7.4, QR-DQN +19.0, DSAC +25.7 (3/3 rejected).
- R2, X + cal − X + ng: +3.1, +4.7, +6.0 (3/3 rejected).
- All 18 per-training-seed contrasts are positive.
- **R5:** no learned incumbent beats H\*. PPO − H\* = −2.96\* (TD), −0.76 n.s. (TD_all); QR-DQN ≈ −22\*;
  DSAC ≈ −34\*.
- Zero-shot (old checkpoints) gives the same signs.

**Recall (the main weakness):**
- Absolute recall on TD (ablation seeds): H\* 37.1, **nc 40.5**, gate 39.7, band 39.8, ng 38.7, h1 38.9, h0 38.2,
  cal 37.6 (%).
- **The unguarded planner nc has the highest recall** at the reference, in N7x2, B0.5 and B2, and with every learned
  incumbent. Yet its return is 9–30 points lower than ng.
- **Calibration lowers recall** in every setting with a detectable difference (−0.4 to −1.0 points\*).
- ng − H\* recall: +1.2 to +2.3\* in N7x2 / B0.5 / B2; −0.2 to −0.4\* in N5x3 / N5x4; n.s. in N3x2.
- The cause has not been decomposed. The Qmin-compliance rates of nc and ng are equal (0.866 vs 0.867). A post-hoc
  return-term decomposition on stored rows is possible without simulation.

**Controller timing (slot = 1 s; the 50 ms budget is per decision):**
- **Sequential** (how the simulator runs; includes scheduler and updates): p99 103–229 ms for every H ≥ 1 planner.
  44–82% of slots exceed 50 ms.
- **Hypothetical parallel** (one core per mode decision): p99 about 41–50 ms at K ≤ 5, M ≤ 3. At N7x2 and N5x4, 31–43%
  of slots still exceed 50 ms; the (M+1)^K scheduler alone takes about 29–31 ms.
- Only H0 fits sequentially (16 ms), and it is the worst planner.
- **Simulated return does not charge controller time.**
- The 23/11 letter's statement of "p99 39–48 ms" was per decision / parallel.
- MPC arms are not bit-reproducible, because they run on a wall-clock budget.

**Novelty audit** (`paperA-novelty-audit-ea-smpc.md`):
- MPC with online model-error estimation, RobustMPC-style conservatism, uncertainty-gated override of a learned policy
  (e.g. OSAP, HotNets 2020) and empirical calibration are all prior art.
- The combination is "partially novel in this application" only.
- The most defensible new content is an evaluation finding: **under gate + band, nominal ≈ calibrated**.

## 5. Consolidated status of claims (from `paperA-fd-consolidated.md`)

**Supported:**
- The layer improves return over a tuned heuristic (reference + 5/6 configurations).
- It improves return over 3/3 learned families, with or without retraining.
- The band is necessary.
- At least one image of look-ahead is needed.

**Not supported or contradicted:**
- The gate is necessary.
- H = 4 is needed.
- Calibration helps at the reference with H\*.
- Calibration improves recall (it lowers recall everywhere).
- Return and recall rank the arms the same way.
- Real-time feasibility in general.
- Learned incumbents beat the tuned heuristic.
- Algorithmic novelty.

**Regime-dependent:** calibration helps on return with learned incumbents and at N5x3 / N5x4.

**Claude's recommendation: C.** Reposition Paper A as an application and empirical contribution.
- **Not A** (keep the multi-step gated controller as the method): the ablation removes its mechanism claim.
- **B** (a simpler band + H1 controller without the gate) only as a *separately pre-registered* study, because it was
  chosen after seeing the data. H1 still exceeds 50 ms sequentially (p99 about 112 ms, 49% of slots).
- **Not D** (drop the paper): the evidence base is large and complete.

## 6. Manuscript state

- `paper/paperA/main.tex` is the **23/11 method letter**: ICT Express template, currently 6 pages with IEEE-style
  figures, 4 figures, 15 references. Its title and abstract claim that ACK-calibrated selective MPC beats nominal MPC
  in 12/12 configurations.
- **This is now outdated.** The full-data results contradict its headline (matched nominal ≈ calibrated). It also
  presents the gate and H = 4 as parts of the method, and it uses per-decision latency.
- Agreed process: **no rewrite until the reviewer accepts a direction** (C / B / D) and a venue.
- Open TODOs: authors, affiliation, funding. The venue / page budget is undecided: ICT Express allows 4 pages
  including references, with a fee for extra pages; the alternative is a longer venue.

## 7. Claude's honest view on "is it strong enough to submit?"

**As the current manuscript: no.** Its main claim is contradicted by the pre-registered matched comparison.

**After a rewrite under C:**
- It is submittable as a **letter or conference-level empirical paper**. The evidence is large, pre-registered and
  internally consistent.
- It is **not** a Q1 algorithmic-contribution paper.
- **Expected reviewer objections:**
  1. limited novelty (all components are prior art);
  2. simulator-only evaluation with assumed LA parameters;
  3. the confirmatory metric (return) disagrees with the task metric (recall), which is awkward for a "semantic
     communication" paper;
  4. controller time is not charged, and the sequential timing exceeds the slot;
  5. a single dataset family (VisDrone).
- What would most strengthen it **without new simulation**: the post-hoc return-term decomposition explaining nc's
  higher recall but lower return. This could become a finding ("return-optimal guards suppress recall-useful
  switches") instead of an unexplained weakness.

## 8. Decisions requested from the reviewer

1. Accept recommendation C, or argue for B or D.
2. Request (or not) the post-hoc return-term decomposition nc vs ng (stored rows only, no simulation).
3. Venue and page budget (ICT Express 4-page letter vs a longer venue). This determines how much of section 4 fits.
4. If C: an outline of the rewritten letter (title, the 3 claims, which figures), before any LaTeX edit.
5. Confirm that Paper 2 (cooperative UAV3D + LA + receiver state; `phases/2026-11-18_design-reset/`) starts after
   submission.

## 9. Constraints and where things are

**Constraints:**
- **Data:**
  - Data and weights are never committed.
  - MDMT has verbal permission only: never publish its images, crops or embeddings.
  - CrossTrack-UAV needs a signed agreement.
- **Code and branches:**
  - `runs/active_recovery/` belongs to another agent and is off-limits.
  - PPO runs on CPU; GPU jobs go through a lock.
- **Used test seeds** (do not reuse): 7000–7099, 8000–8999, 9000–9099, 9200–9299, 9600–9699, 9800–9999, 12000–14099,
  20000–26099.

**Key files** (paths from the repo root; prefix `https://github.com/vutl/SwarmBridge/blob/main/`):

| File | What it holds |
|---|---|
| `reports/phases/2026-11-23_paperA-fulldata/paperA-fd-consolidated.md` | Claims / evidence matrix + recommendation |
| `reports/phases/2026-11-23_paperA-fulldata/paperA-fd-{phase2,ablation,phase4,phase3}-result.md` | Per-phase results |
| `reports/phases/2026-11-23_paperA-fulldata/paperA-full-data-protocol.md` and `*-prereg.md` | Protocol and pre-registrations |
| `reports/phases/2026-11-23_paperA-fulldata/paperA-novelty-audit-ea-smpc.md` | Novelty audit |
| `paper/paperA/main.tex`, `paper/paperA/CLAIMS.md` | Current (outdated) manuscript and its number sources |
| `reports/HISTORY.md` | Full history, every phase, newest first |
| `reports/README.md` §3 | Corrections table |
| `reports/phases/2026-11-18_design-reset/swarmbridge-design-review-2026-11-18.md` | Paper 2 design |

**Already tried, so do not re-propose without new reasons:**
- **Schrödinger Bridge:** latent completion (≤ a simple mapping).
- **Joint control with PPO:** movement + channel + power + mode (A3.2 / A4 / A5; movement adds nothing because image
  content does not depend on position).
- **Learned channel scheduling:** the JA heuristic ≥ every RL scheduler.
- **Per-image task-value prediction:** does not transfer to test-dev.
- **Content representations:** TOKEN / CROP / FRAME semantic modes; JEPA codec.
- **MDMT cooperative reporting and view selection:** the good view is fixed, so gains are small.
- **Songdo:** the channel is not constrained.
- **RL value over a tuned heuristic** on this problem: none found (A-X9, E7, Phase 3 R5).
