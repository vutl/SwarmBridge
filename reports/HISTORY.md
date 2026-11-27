# SwarmBridge: full research history (for reviewers)

Last updated 2026-11-25. A shorter, self-contained brief for starting a new reviewer session is
`reports/CHATGPT_HANDOFF.md`; read it first. This file is written for an external reviewer (e.g. ChatGPT) who needs to know everything that has been
tried before proposing anything.
- Sections 1–3 are the state of the project.
- Section 4 is every phase, **newest first**, with the question, what was built and run, the key numbers, the
  conclusion, and later corrections.
- Sections 5–6 are assets and process lessons.

Detailed reports: `reports/phases/<date>_<name>/` (one folder per phase, each with a README). Corrections table:
`reports/README.md` section 3. Short version: `reports/CONTEXT.md`.

**Reviewer rule (agreed 18/11):** before proposing a direction or saying "X was never tried", check this file and the
phase reports. Two recent review rounds made factual errors that this file answers (section 6).

## 1. Goal and original thesis

- **Goal:** multi-UAV + semantic communication + swarm intelligence, aiming at a Q1 paper with an algorithmic
  contribution.
- **Original thesis** (29/10, `pdf/Swarm-Conditioned Schrödinger Bridge cho Task-Oriented Multi-UAV Semantic
  Communications.pdf`): *Can a distributed UAV swarm communicate only the complementary evidence required to reduce
  the receiver's task uncertainty?*
  - Several UAVs have complementary views.
  - Message value is conditional on what the receiver already knows: I(Y; M_i | C), oracle Δ = L(C) − L(C, M_i), then
    a learned value predictor.
  - Wireless cost.
  - Closed loop: uncertainty → request evidence → completion.
  - A conditional Schrödinger Bridge (SB) was the candidate completion model, with an early kill criterion against
    mapping / OT-CFM.
  - MARL was explicitly deferred; active flight (AirSim) was a later stage.

## 2. Current status (20/11; full-data update 25/11)

- **Extended full-data evaluation of Paper A (23–25/11, `phases/2026-11-23_paperA-fulldata/`, namespace paperA_fd,
  protocol fd_v1).** All phases were pre-registered and run to completion: 25,200 new episodes, coverage and
  checkpoint checks pass. Consolidated in `paperA-fd-consolidated.md`.
  - **Positive:**
    - the error-aware layer (ng) beats the tuned heuristic at the reference and in 5 / 6 network / bandwidth
      configurations;
    - it beats 3 / 3 learned incumbents (PPO, QR-DQN, DSAC);
    - the robustness band is the necessary component.
  - **Negative / inconclusive:**
    - the tracker gate is not necessary; H4 vs H1 does not replicate;
    - calibration adds return only with learned incumbents and at N5x3 / N5x4, and **lowers recall everywhere**;
    - the unguarded nominal planner has the highest recall (return and recall are not aligned);
    - sequential controller timing exceeds the 50 ms slot in 44–82% of slots (parallel too at K7 / M4);
    - no learned incumbent beats the heuristic.
  - **Recommendation C:** reposition Paper A as an application and empirical contribution. B (band + H1 controller)
    only as a separately registered study. No manuscript rewrite before review.
  - **Execution incidents:**
    - 24/11: the WSL VM stopped 90 s after launch, unnoticed for 3 h.
    - 25/11: host power loss; resumed with `--skip-existing`; completed pools were kept.
  - **Manuscript status:** `paper/paperA/main.tex` (the 23/11 method letter) is **outdated**. Its headline,
    "calibrated > nominal in 12/12", compared against the unguarded nominal planner; against the matched nominal
    design the difference is −0.10 n.s. It also presents the gate and H = 4 as parts of the method and reports
    per-decision latency. It must not be submitted as it is. The rewrite waits for the reviewer's decision on C / B / D
    and on the venue.
  - **Decided (new ChatGPT session, 25/11):** C; venue ICT Express (4 pages including references); Paper A only until
    submission.
  - **Post-hoc analyses on stored data** (`paperA-fd-posthoc-result.md`; no simulation):
    - **Fail-fast verification:** 27,600 / 27,600 episodes OK; a negative test exits 1.
    - **Return decomposition:** ng sends smaller payloads than nc (194 vs 303 kbit). Drops fall by 15.8 and
      below-Qmin deliveries rise by 15.7; these cancel. The net return gain comes from SCE (+9.7) and mission time
      (+8.1). Recall falls by 1.8: two thirds from a lower detection yield (lower PSNR), one third from which images
      end up compliant. Of 53 contrasts, 44 have a positive mean return difference and 41 of those send smaller
      payloads (exceptions: ng − H\* in three pools).
  - **Manuscript rewritten (25/11, ChatGPT GO for ICT Express):**
    - title "Uncertainty-aware mode selection for multi-UAV semantic image offloading";
    - 4 pages; 2 figures + 1 table, all from stored fd_v1 results (`paperA_fd_numbers.py`, `paperA_fd_figures.py`);
    - claims → sources in `paper/paperA/CLAIMS.md`;
    - writing report `phases/2026-11-23_paperA-fulldata/paperA-fd-writing-report.md`;
    - not submitted; awaiting final review.
  - **6-page version (0a27d406)**, then the **final revision (26/11)**:
    - title "Predictive mode selection under wireless rate uncertainty for multi-UAV semantic image offloading";
    - vector Fig. 1;
    - RobustMPC-style comparison: guarded − RobustMPC-style +6.4 on TD (post-hoc);
    - Phase 2 confirmatory row in Table II;
    - fixed-policy weight sensitivity: F2 robust, F1 depends on the mission-time cost (V, w = 0: nominal − H\*
      +4.8\*);
    - wording fixes;
    - snapshot script (tag `paperA-v1.1`).
    The next step is the final pre-submission review, then the professor review.
    - **PPO + cal − H\* + cal:** +0.27 n.s. on TD (n.s. in all 4 pools); recall −0.73\*.
    - **R5 recall with the cluster bootstrap:** the retrained incumbents are at or below H\*. Correction: the
      zero-shot PPO-LA has recall +3.1\* above H\*, with return −4.9\*.
  - **Next:** the reviewer re-assesses novelty, significance and the target; then the 4-page outline.

- **Decision (18/11, after review #48):**
  - Write Paper A now as an evaluation study + calibrated planning baseline, from existing data
    (`phases/2026-11-18_paperA-writeup/paperA-outline.md`).
  - The cooperative multi-UAV direction (tracking on UAV3D + realistic LA; ChatGPT's post-#48 proposal) is Paper 2,
    started after Paper A is submitted.
- **Robustness round E1–E3 + E1b (19/11, `phases/2026-11-18_paperA-writeup/paperA-r2-robustness-result.md`):**
  - Rank reversal PPO-S/T:best holds on 4 LA variants and on test-dev.
  - The calibrated MPC beats a RobustMPC-style history-only predictor on every LA configuration (+3.4..+17.8).
    E2b ablation: removing the SNR bins changes nothing (11/12 CIs include 0), so the gain is NOT from SNR
    conditioning (an earlier claim of ours, now corrected).
  - MPC p99 decision latency 39–48 ms (episode max up to 64 ms; <= 0.31 % slots over 50 ms).
  - On test-dev, video-level (seed x video) recall CIs: PPO-LA + MPC − H* + MPC −2.5 recall despite +7.4 return.
  - The best fixed mode changes with LA conditions (q3 -> q1). PPO-LA transfers better than the frozen heuristic
    (+61 / +51) but trails the re-tuned heuristic (−31 / −29; PPO not retrained).
  - On test-dev content, PPO + MPC beats train-tuned heuristic + MPC (+7.4 L, +8.0 S).
- **4-page rewrite (20/11, `paper/paperA/main.tex`, ICT Express letter):** title "When wireless models and task
  metrics disagree: Revisiting multi-UAV semantic image offloading"; main findings = PHY-dependent return ranking +
  return/recall misalignment; Table 1 (contrasts A-F over 7 configurations, generated by `paperA_paper_numbers.py` +
  `paperA_table.py`), Fig. 1 system, Fig. 2 misalignment. No general "PPO transfers better" claim; frozen vs re-tuned
  kept separate; SNR-conditioning claim withdrawn (calibrator SNR spans only 10.8-18.9 dB); no hard 50 ms claim.
  Status: draft for ChatGPT's final review, not yet a GO for submission.
- Other branches are closed or paused.
- **R1 done (18/11, seeds 9800–9899, `phases/2026-11-18_paperA-writeup/paperA-r1-result.md`):** the same frozen
  policies on Shannon vs LA.
  - Designs made for Shannon lose on LA: tuned heuristic −98.9, PPO-S −22.8, A5 slot rule below −150.
  - ACK calibration of the MPC: +19.8 (vs the same MPC uncalibrated).
  - On LA, PPO-LA + MPC vs H* + MPC: +1.51 [−1.55, +4.58].
  - On Shannon, with the literature-style setup, PPO-S beats the tuned heuristic by +2.1.
- **Open:** a problem reset (`phases/2026-11-18_design-reset/swarmbridge-design-review-2026-11-18.md`).
  - Proposal: realistic link-adaptation (LA) wireless stack from Paper A + UAV3D cooperative pipeline + receiver state
    (track uncertainty / age).
  - Movement in stage 2, where observation value depends on position.
  - The same report (section 5c) lists what UAV3D already showed against simple receiver-state rules.
- **Waiting for the user to decide:**
  1. the stage-1/2 framing;
  2. a cheap "greedy receiver-state rule vs lookahead oracle" property check on UAV3D mini;
  3. whether full UAV3D can be obtained;
  4. keep or drop SB.
- **Paper A** (independent image offloading; DRL / MPC mode and scheduling control): closed at review #47.
- **Paper B** (JEPA semantic codec on VisDrone-VID): paused since review #35.

## 3. Facts that stand after all corrections

1. **Cooperation helps when views are complementary, and it is communication-sensitive.**
   - UAV3D mini (5 synchronized UAVs, 3 dev scenes, no CI):
     - ego mAP 0.512 → box fusion 0.665 (AMOTA 0.594);
     - Kalman (KF) track messages with velocity: AMOTA 0.542 → 0.690;
     - 2-frame delay removes almost all of the gain; 50 % loss costs 5.5 AMOTA.
   - MDMT (2 drones): the good view is almost always the same drone, so cooperative gains are small.
   - Town3 was run with one source per region (N = 1), so cooperation was never tested there.
2. **Simple receiver-state rules were already tested on UAV3D dev** (synthetic independent links, no interference,
   no LA):
   - "Send only what the receiver lacks" is wrong for tracking. Value is mostly confirmation (about 65 vs 7.5 cars per
     message), and dropping already-seen objects pushes AMOTA below ego.
   - Receiver-certainty feedback is no better than inter-helper de-duplication once the feedback bits are charged.
   - Progressive deadline-bounded messages ≈ a channel oracle.
   - A content/order headroom of +0.025..0.044 AMOTA remains. Two simple causal estimators did not capture it; a
     stronger estimator on 17 train scenes was never run.
3. **The PHY assumption changes conclusions.**
   - A policy trained on a Shannon channel loses 16.8 return under realistic LA (delayed CQI, OLLA at 10 % BLER, ARQ,
     persistent co-channel interference, scheduler-aware LA).
   - Channel-reuse gains vanish under realistic LA.
4. **Baselines must be tuned.**
   - Untuned literature-style rules lose by 10..170 to tuned heuristics.
   - The JA heuristic scheduler matches or beats every RL scheduler we trained.
   - This is evidence from the specific papers and baselines we reproduced, not a claim about the field.
5. **Independent image offloading is close to myopic.**
   - PPO trained on the correct PHY + calibrated MPC vs heuristic + the same MPC: −0.14 [−2.83, +2.56] (pre-registered
     A-X9).
   - Joint movement + channel + power + mode PPO was tried (A3.2 / A4 / A5): small gains. RL-chosen movement adds
     nothing in A4, because image content does not depend on position (frames are drawn from a pool).
6. **Online ACK-based calibration fixes a mismatched planner model** (+17 vs nominal MPC). Calibrated,
   compute-bounded MPC already exists in video streaming (RobustMPC / FastMPC, Oboe, Fugu).
7. **Per-image task value is neither predictable from cheap features nor transferable.**
   - The object-count oracle gives +1.53 recall on same-source content and −0.97 on held-out VisDrone test-dev.
   - The per-mode detector-outcome oracle gives +2.17 but cannot be approximated.
8. **With a fixed detection task at the receiver, sending box lists usually beats sending features or latents**
   (hybrid pilot, LIF, Codex gate A, UAV3D, Town3 map baseline).
9. **Novelty landscape:**
   - DRL selecting SemCom mode + resources (+ trajectory) for UAVs exists: Si et al. Globecom 2023, Huang et al. JSAC
     2025, Li et al. JSAC 2026.
   - "Who/what to send" for collaborative perception exists: Where2comm, CoSDH, CodeFilling, How2comm, SchedCP, semantic
     HARQ for CP, CCSRA; most assume a Shannon channel or perfect CSI.
   - "Send when value > cost" exists: GOSC, AoII.
   - SB for semantic communication exists (SBGSC 2026).
   - Combining these is not by itself novel.

## 4. Phases, newest first

Format: **dates (review numbers)** | question | what was done | key numbers | conclusion | later corrections.

### 4.0b Paper A independent evaluation: 26/11, STAGE 1 PROPOSAL (awaiting review)

`phases/2026-11-26_independent-eval/independent-eval-audit-proposal.md`. Submission paused.
- **Trigger:** TD / V were used in earlier development, so there is no fully untouched test set.
- **Audit:**
  - TD reuse is 146/582 frames and 49/61 sequences; the 26/11 discussion said 403/582, which is wrong.
  - rl_train, rl_val and F were never touched.
  - F is method-unseen (used only for YOLO checkpoint selection) and already replicates the return findings:
    nominal − H\* −8.6, band +16.2, guarded − H\* +5.4.
  - UAVDT, DroneVehicle and DET test-challenge were never downloaded.
- **Proposal:**
  - X1: DET test-challenge, frozen artifacts, return endpoints only (no public labels). Includes a prefix-overlap
    exclusion rule.
  - X2: UAVDT in two arms. X2-frozen tests zero-shot transfer. X2-retrained re-applies the procedure on UAVDT
    train, including RL. Recall uses the frozen YOLO with mapped classes.
  - 3 × 7,400 episodes, about 2–3 days.
  - Retraining on VisDrone alone removes nothing (fd_v1 already trained on clean splits).

**26/11 later: stage 1 review approved in principle** (10,200 episodes). Done:
- downloaded VisDrone DET test-challenge (1,580 images) and the UAVDT GT/toolkit;
- UAVDT images are blocked by the gdown quota, so the user downloads them in a browser;
- X1 hash audit (`ix_audit.py`): 0 prefix / sha matches, 1 near-duplicate sequence (0000332) excluded, leaving
  52 sequences and 631 capped frames;
- UAVDT: 50 sequences, official test = 20 sequences (from RES_DET), classes car/truck/bus with no van;
- fd_v1 14-mode audit (`ix_modes.py`): the generative modes are 0.1–0.2% of frames, cap_t2i is never compliant;
- revised prereg draft `paperA-ix-prereg-draft.md`: A2 = band only − nominal; replication families as registered;
  new family N1–N3 (nc−H\*, guarded−H\*, cal−guarded); lock L4 = retrained checkpoints hashed before any test episode;
  frozen loader spec + tests.

**26/11 latest: draft approved in principle; final candidate v2** (`paperA-ix-prereg.md`).
- UAVDT images downloaded by the user (sha256 e82675c9…, 40,735 frames).
- Full cross audit `ix_audit.py full`: 0 UAVDT~VisDrone, 0 X1~UAVDT and 0 UAVDT train~test links; only X1 0000332
  is excluded.
- UAVDT frames are 1024×540 / 960×540. M0207 annotations end at frame 571, so only annotated frames are used.
- Manifest `manifest_ix_v1.json` (734ea49b…): U_train 26 sequences / 780 frames, U_val 4 / 120, U_test 20 / 600,
  X1 52 / 631.
- `ix_common.py` frozen / refit loader, pinned fd_v1 state hash 39018855…; `test_ix_common.py` 11/11 pass,
  including deterministic-budget behavioural equivalence and the real timing path.
- Order corrected: test profiling only after the L2 lock.
- Principal finding: N2 and A2 in X1.
- Recall: vehicle class-agnostic primary; GT inside ignore regions removed; all occlusion levels count.

**27/11: design APPROVED (ChatGPT, e964ad41); drivers and tests done** (prereg FINAL v3 §10).
- New drivers: `ix_profile`, `ix_cpu`, `ix_stage` (fit / tune / train / select / lock), `ix_eval`, `ix_analysis` and
  `ix_actlog`. Trainers get opt-in flags `--ix` and `--log-actions`; `ax2_eval` gets ppoix / qrix / dsix.
- Frozen and retrained states are serialised. Evaluation loads them with every fitting function disabled. L2 pins
  the data, states, H\*, 9 checkpoints and 22 code files, and stage B refuses to start without it.
- Tests: 561 in the full suite pass, plus 2 GPU tests (`IxPipe` reproduces the stored fd_v1 rows).
  - `ix_analysis` reproduces the stored fd_v1 ablation / Phase 2 summaries.
  - Action logging leaves PPO and QR-DQN parameters bit-identical.
  - End-to-end smoke test on the sandbox.
- Launcher: `runs/uav_semcom/paperA_ix/scripts/run_stage.ps1 -Stage A|uavdt|x1`.
- X1 waits for the VisDrone answer.

**27/11: stage A DONE and L2 locked** (run 00:26–14:22; `L2_lock.json` created 17:14, verify OK).
- Profiling U_train + U_val: 48 min.
- MBT2018 CPU frame-weighted estimates (s): q1 0.232, q3 0.228, q5 0.463, q8 0.461.
- Retrained state on U_train (780 frames): 8fd86b50….
- H\* on U_train:
  - N5x2, B0.5, B2: mbt2018 q3
  - N3x2, N5x3, N5x4: avif_50
  - N7x2: avif_20
  - fd_v1 was avif_20.
- RL: 9/9 runs completed the full 2M steps, all exit 0. Selected checkpoints (best step, best U_val return):

  | Algorithm | s0 | s1 | s2 |
  |---|---|---|---|
  | PPO | 2.0M, −1.13 | 1.0M, −2.86 | 2.0M, −0.80 |
  | QR-DQN | 1.8M, −35.7 | 1.6M, −28.5 | 1.8M, −26.7 |
  | DSAC | 0.8M, −63.1 | 1.4M, −53.3 | 1.4M, −50.2 |

  These are validation numbers only.
- Training action histograms: all 14 modes are executed in every run.
  - PPO mostly picks mbt q3 (33–39%), avif_50 (14–22%) and mbt q1 (12–15%); generative modes are ≤1.8% each.
  - The `--heur` flag is identical to fd_v1 and inactive (heur_mode = 0).
- No hyperparameters, budgets or selection rules changed.
- Stage B uavdt launched 17:14.
- 27/11 18:19: the user authorised running X1 now. `scripts/after_uavdt_x1.ps1` starts stage x1 after the uavdt
  launcher exits. The two run sequentially because the MPC arms use a wall-clock budget.

**27/11: stage B UAVDT DONE** (17:14–19:49; commit 757388aa; `paperA_ix/analysis/summary.log`).
- All 6 blocks are complete (8,800 episodes). Integrity: 0 wrong task counts, 0 differing images, 0 deliveries
  outside the pool. Original-image recall: veh 78.94, vancar 75.44, car 70.85.
- **Principal verdict (N2 & A2):**
  - **X2-frozen: supported.** N2 +1.17 [+0.18, +2.15], Holm p 0.041; A2 +11.82 [+8.05, +15.59].
  - **X2-retrained: supported.** N2 +7.61 [+4.65, +10.57]; A2 +9.19 [+5.69, +12.69].
  - No negative N2 / A2.
- **X2-frozen secondary results:**
  - Calibration hurts: H1 (cal − H\*) −0.85 and H3b / N3 (cal − guarded) −1.77 / −1.65, all Holm-rejected negative.
  - A4 (H4 − H1) −1.23, Holm-negative. A4b +5.29.
  - N1 (nominal − H\*) +1.05, not significant (fd_v1: nominal lost). A1 (gate only) −0.03, not significant.
- **X2-retrained secondary results:**
  - H1 +6.98. H3b / N3 (cal − guarded) about 0, not significant. N1 −0.26, not significant. A1 −2.00, not
    significant. A4 +5.84, A4b +23.78.
- **Grid:**
  - G1 (guarded − H\*) is positive and Holm-significant in 5 of 6 settings; N3x2 +0.69 is not significant.
  - G2 (cal − guarded): +2.56 at N3x2 and +2.60 at N5x4; **−10.10 at N7x2**; not significant at N5x3, B0.5, B2.
- **RL:**
  - R1 (guarded on top of RL − RL): PPO +6.88, QR-DQN +22.93, DSAC +37.60, all Holm-significant.
  - R2 (cal − guarded): PPO +2.75 Holm-significant; QR-DQN +1.84 (Holm p 0.069); DSAC −0.40.
  - Level (not tested): PPO alone −36.78 vs H\* −31.48.
- **Recall (veh, system level):**
  - N2: +1.0 (frozen), +4.5 (retrained).
  - A2 (band − nominal): −7.7 (frozen), −1.1 (retrained).
  - N1 (nominal − H\*): +16.5 (frozen), +7.3 (retrained).
  - So the return–recall trade-off between nominal and banded planning reappears.
- **Utilisation:** generative modes ≤0.7% in the MPC arms. Exceptions: RobustMPC without band about 21%, DSAC alone
  9–13%, QR-DQN alone 5–11%. cap_t2i is never compliant.
- **Cost:** same pattern as fd_v1.
  - Sequential full-sum p99 is 139–224 ms. 47–83% of slots exceed 50 ms; only H0 fits (16 ms).
  - Hypothetical parallel max p99 is 39–50 ms at K ≤ 5 / M ≤ 3, but 69–75 ms at N7x2 / N5x4.
- **X1:**
  - The queued launch failed at start: at 19:50 the `.venv` uv trampoline could not find the base interpreter (works
    again at 20:40). No X1 profile or episode exists.
  - The relaunch waited for the user.

**27/11: stage B X1 DONE** (the user said go; 20:39–22:03; L2 verified). The X2 analysis is unchanged by the rerun.
- 1,400 episodes. Integrity is clean. Recall is unavailable by design.
- **Principal block X1: PARTIALLY SUPPORTED.**
  - N2 (guarded − H\*) +2.13 [−0.42, +4.69], Holm p 0.10, not rejected.
  - A2 (band − nominal) +19.27 [+16.00, +22.54], rejected.
- **Other results:**
  - N1 (nominal − H\*) −14.69, rejected (nominal loses, as in fd_v1).
  - N3 (cal − guarded) +3.86, rejected.
  - A1 −0.09, not significant. A4 +1.95, not significant. A4b +23.18.
  - Reference block: H1 (cal − H\*) +5.54, rejected; H3b +0.42, not significant; H2 vs RobustMPC +5.94.
- **Utilisation:** generative modes ≤0.5%, except RobustMPC without band at 26%.
- **Cost:** as on UAVDT. Sequential p99 129–160 ms, 47–75% of slots over 50 ms; parallel max p99 31–44 ms with
  0 slots over 50 ms.
- **All three verdicts:** X1 partially supported, X2-retrained supported, X2-frozen supported.

**27/11 late: publication artifacts FROZEN; research complete.**
- ChatGPT approved `ca6f8099`:
  - Fig. 1 `fig_fd_system`;
  - Fig. 2 `fig_ix_ablation` (vertical; horizontal kept as `_h`);
  - Fig. 3 `fig_ix_tradeoff`;
  - Table 1 `table_ix_results.tex`;
  - supplementary CSVs in `paperA_ix/paper/`. The levels CSV now uses the union of fields: 7 MPC diagnostic columns
    had been dropped.
- Synced to vutl/uav-selective-mpc (57050fe):
  - ix code and tests;
  - `runs/uav_semcom/paperA_ix`;
  - the 9 selected checkpoints, with sha256 matching L2;
  - protocol and result notes (`paperA-ix-result.md`);
  - figures and table.
  No images, profiles or delivery logs of the new datasets are included. In the dedicated repo, the 46 ix tests pass
  and the table and CSVs are reproduced byte-identically.
- `main.tex` is untouched. ChatGPT now rewrites the manuscript.
- Open items:
  - the user's local Fig. 1 edit (uncommitted, with photos) is not synced;
  - VisDrone test-challenge permission before publication.

### 4.0a Paper A extended full-data evaluation: 23–25/11, COMPLETE (awaiting review)

`phases/2026-11-23_paperA-fulldata/` (README + consolidated report); results in `runs/uav_semcom/paperA_fd/`.

- **Why:** the user and ChatGPT approved re-testing the 23/11 method letter on more data, with a pre-registered
  protocol.
- **Protocol (fd_v1):**
  - New sequence-disjoint VisDrone pools (DET-train → `rl_train` / `rl_val`; test pools TD = test-dev, F = YOLO
    hold-out, V = DET-val, TD_all), after a leakage audit.
  - Re-profiled 14 modes; H\* re-tuned; analysis code frozen before each run.
- **Phases:**
  - **Phase 2 (reference):** cal − H\* +7.28 (Holm); **cal − matched nominal (ng) −0.10 n.s.**; cal − unguarded nominal
    +18.6.
  - **Ablation:**
    - band − nc +15.15 (Holm);
    - gate − nc −1.19 n.s.;
    - H4 − H1 +5.05 on TD but −2.51\* on V;
    - H4 − H0 +25.5.
  - **Phase 4** (N3x2 / N7x2 / N5x3 / N5x4 / B0.5 / B2): ng − H\* Holm-rejected in 5/6 (not N3x2); cal − ng in 2/6
    (N5x3, N5x4).
  - **Phase 3** (PPO / QR-DQN / DSAC retrained, 3 seeds each):
    - X + ng − X: +7.4 / +19.0 / +25.7;
    - X + cal − X + ng: +3.1 / +4.7 / +6.0;
    - no learned incumbent beats H\*.
- **Recall:**
  - Calibration lowers recall everywhere.
  - The unguarded planner nc has the highest recall (TD 40.5 vs ng 38.7, H\* 37.1).
- **Timing:** sequential full-sum p99 is 103–229 ms; 44–82% of slots exceed 50 ms.
- **Conclusion:** the return gain is robust and comes from the robustness band. The multi-step gated controller and
  the calibration are not supported as the source. Recommendation C: an application / empirical paper.
- **Commits:** 3cf4fd62, 000251c7, bc4d2736.

### 4.0 Paper A write-up and robustness rounds: 18–23/11, SCOPE FROZEN

`phases/2026-11-18_paperA-writeup/`, manuscript `paper/paperA/` (claims → sources in `paper/paperA/CLAIMS.md`).

- **Reframing (21–23/11):** from an evaluation study ("wireless model and task metric disagree") to a method letter.
  - Method: ACK-calibrated selective MPC on top of any incumbent mode policy.
- **Rounds:**
  - E1–E3 (PHY variants, test-dev);
  - E2c ladder: no single factor; pooling over UAVs is the most consistent step;
  - E4 network size: rule FAIL, cal > H\* in only 3/5;
  - E5 feedback delay/loss: holds vs nominal in all 5 settings; E5 tracker bug found and fixed first;
  - E6 bandwidth: rule FAIL, cal > H\* in 2/3;
  - E7 RL families (QR-DQN, Discrete SAC): RQ-B 6/6 test, 5/6 test-dev. The new families were weak, so X+cal is
    still below H\*+cal. PPO-LA vs H\* flips sign between pools.
- **Robust claims:**
  - calibrated > nominal in 12/12 LA configurations;
  - calibrated > RobustMPC-style in 11/12.
- **Condition-dependent:** gain over the incumbent alone, 10/12.
- **Return vs recall:** recall falls in 8/12 configurations with H\*.
- **Also reported:**
  - JA latency limits at K=7 / M=4;
  - MPC results are not bit-reproducible (wall-clock budget).
- **Operational lesson:** the first E7 launch died with the Claude session (10 h lost). Long jobs are now launched
  detached via WMI.
- **Stop rule:** no further algorithm rounds for Paper A. Next: final review of the 4-page draft, then Paper 2.

### 4.1 Design reset: 18/11 (#47–#48), OPEN

- **Question:** where did the project drift from the original thesis, and can it be restored without rebuilding?
- **Done:** read the original proposals in `pdf/` and traced every phase.
- **Findings:**
  - The 06/11 pivot to the Huang-style setting dropped cooperation and receiver state.
  - Movement, power and channel were never "forgotten": they were RL actions in A3–A5.
  - Cooperation was tested only on weak data (Town3 N = 1, MDMT 2 drones) or briefly (UAV3D, 1 day, dev only).
- **Proposal:** LA stack + UAV3D + receiver state; movement in stage 2. Hypotheses:
  - H1: receiver-state-conditioned allocation beats unconditioned rules at equal airtime (simple forms already negative
    on UAV3D dev);
  - H2: lookahead beats greedy-conditioned;
  - H3: Shannon-designed policies lose under LA.
- **Biggest risk:** if neither realistic LA nor lookahead creates a difference, cooperation on UAV3D gives a
  message-representation contribution, not a scheduling one.

### 4.2 Paper A, scheduling + realistic LA + task utility: 17–18/11 (#37–#47), CLOSED

`phases/2026-11-17_paperA-scheduling/`. Setting: 5 UAVs, 2 channels, BS; images from a VisDrone pool; 14 codec modes
(AVIF, mbt2018, JPEG2000, thumbnails, 3 caption-generative modes); deadlines 4–12 s; Qmin 22–30 dB.

| Round | Question | Key result |
|---|---|---|
| A-X2 (#37) | Learned channel scheduler (C-only PPO) | +2.5 / +3.1 vs tuned heuristic on Shannon. Learning channel + mode unstable. Literature-style untuned rules lose by 10..170 |
| A-X2b (#38) | C-only on MCS with genie (interference-aware) LA | +11.3; with partial LA knowledge λ = 0.75 → −32 |
| A-X5 (#39) | Block-level realistic LA (`env_la`: delayed CQI, OLLA, ARQ) + randomised training | No-reuse heuristic best in all 7 conditions; randomised C-only stuck at 0.86 reuse (−42.9) |
| A-X6 (#41) | PHY audit; scheduler-aware LA; opportunity decomposition | Atomic-block fix changes A-X5 by ≤ 2. Causal joint assignment (JA) +8.2 vs least-slack; reuse −5..+6; non-causal channel oracle +12.5. JA without reuse = A-X2b RL |
| A-X7 (#42) | Information value; correlated channels; mode controller on LA | The oracle gap is slot-frozen fast fading (LoS-only oracle ≈ 0); SRS probing +3. Calibrated PPO–MPC +12..17 vs zero-shot PPO; an initial "+19 interaction" was a movement confound |
| A-X8 (#43) | Fair baseline (heuristics retuned with JA and movement) | PPO–MPC − best heuristic = −1.25 n.s. MPC seeded with a heuristic incumbent is best (PPO incumbent −5.0). JA +8..16 robust |
| A-X9 (#44) | Pre-registered final DRL test (seeds 8300–8399) | LA-trained PPO + MPC vs heuristic + MPC −0.14 [−2.83, +2.56]; recall −0.031; gate FAIL 4/5 → **DRL line stopped** |
| Novelty + A-T1 (#45) | Novelty matrix; per-image detector-utility oracle inside the network | No defensible algorithmic gap. Oracle cross-checked +1.71 [1.29, 2.14] recall vs q3 + MPC (gate PASS); ≈ object-count knowledge via congestion relief; violates Qmin (−2.15 compliant recall) |
| Step C (#46) | Causal object-count predictor on held-out test-dev (49 videos, seeds 9600–9699) | Selected light-detector predictor +0.20 [−0.94, +1.63], macro recall −3.0 → gate FAIL. Count oracle itself −0.97 on new content (fewer objects per frame: 46.9 vs 68.6). Detector oracle still +2.17 |
| #47 | — | Independent-offloading line closed; LA stack kept as an asset |

### 4.3 Two tracks (Paper A DRL + MPC under model mismatch; Paper B JEPA): 15–17/11 (#26–#36), CLOSED / PAUSED

`phases/2026-11-15_two-track/`.

**Paper A rounds:**
- **A-E1:** PPO costs 1.2–1.7 ms per decision vs 40–60 µs for a heuristic.
- **A-E2/E3:** correct-model MPC +2.7 vs PPO at 30–100× the cost; Shannon MPC −8.6.
- **A-E4:** non-clairvoyant rollout ≈ PPO; ACK calibration fixes model mismatch; PPO + correct MPC +2.0.
- **A-E5:** slot-weighted calibration; error gate; +1.69 / +1.86.
- **A-E6:** decision-aware gate +2.82 / +2.23 (bias robustness: Shannon +1.30 instead of −8.67).
- **A-E7:** two-stage screen at 20–25 ms.
- **A-E8:** 50 ms budget held (p99 40–44 ms).
- **A-E9:** fresh seeds +2.18 / +1.91; robust to feedback delay or loss thanks to warm start.
- **A-E10:** mismatched warm start −6 / −12.
- **A-E11:** a 30-observation window fixes it.
- **A-X1:** channel × mode coupling exists at K5.
- **Corrected later:** all A-E gains were measured **against PPO**, not against heuristic + MPC. Under realistic LA
  (A-X8/A-X9) PPO ≈ tuned heuristic.

**Paper B rounds:**
- **B-E0:** the DINOv2 token pipeline works; raw tokens cost about as much as AVIF.
- **B-E1/E2:** PCA codec; learned / JEPA-style predictors ≤ "hold".
- **B-E3:** learned codec beats AVIF; JEPA-32 +0.02 mAP.
- **B-E4:** 3 seeds, confidence intervals include 0.
- **B-E5:** smoothing alone ≥ JEPA, so the JEPA gain = temporal stability.
- **B-E6:** a rate-only surrogate +0.03 mAP.
- **B-E7:** inter-frame HEVC/AV1 → DINOv2 beats the semantic codec at ≤ 12 kbit, so Paper B as designed is not viable.
- **B-E8/E9:** task-aware ROI-QP ceiling ≈ +0.01 AP.

### 4.4 Songdo + JEPA proposal: 14–15/11, CLOSED

UAV command-and-control (C&C) state streams on Songdo Traffic: fleet maximum 11.2 kbit/s vs ≥ 60 kbit/s for the C&C
channel alone, so communication is not constrained. NO-GO.

### 4.5 Novelty review + RQ1 on MDMT: 13–14/11, CLOSED

- **Novelty:** PPO / MAPPO choosing TOKEN / CROP / FRAME messages is already covered (CCSRA, HyComm, SchedCP,
  Where2comm, CodeFilling, CoSDH).
- **RQ1** (cross-view association as the reason to transmit):
  - Phase 0, geometry only: NOT IDENTIFIABLE (GT-free registration works on only 3/8 sequences).
  - RQ1-E (DINOv2 zero-shot re-identification on real frames): E0 FAIL 0/8.

### 4.6 A7 / A8 / B1: 13–14/11, CLOSED

- **A7-S0** (TOKEN / CROP / FRAME families in the A5 env): information upper bound +6..11 % return, +8..10 recall.
  The headroom is content-driven, not link-driven; the causal heuristic is unstable.
- **A7-S1:** the headroom is scene-level GT knowledge (previous-frame values keep 48–83 %).
- **A7-S2:** a label-free BS proxy ranks scenes well, but gives false positives in dense scenes → NO-GO.
- **A8** (model-mismatch ladder): **INVALID**, because the compared policies had no co-channel interference.
- **B1** (view choice on MDMT, 8 sequences):
  - NO-GO for the view-choice rule;
  - the label-free "most detections" rule picks the same view 8/8;
  - view-aware allocation +2.5 F1 vs alternating;
  - the good view is drone 2 in 7/8 sequences.
- **Data:** MDMT rights are verbal only; M3OT is CC BY but only 3/9 sequences pass the ID screen; CrossTrack-UAV needs a
  signed agreement.

### 4.7 H1 on MDMT (cooperative unique-object reporting): 11–13/11, CLOSED

- **Gate 0A:**
  - token headroom only near budget ≈ token stream (+3..6 points);
  - crop headroom +4..9 at 100–500 kbit/epoch with perfect crops.
- **Audit:** token NO-GO, crop conditional GO.
- **Pilot 0B** (real AVIF + YOLO11n/s):
  - crop +1..3 F1, 70–99 % of it from one sequence;
  - full frame +8 F1 but over 1 s CPU.
- Not stable enough for RL.

### 4.8 R0 → R1b on VisDrone: 10–11/11, CLOSED

- **G-L2** (per-image task value predictable?): FAIL.
  - Static oracle headroom +1..2 points, cross-checked +0.95 / +0.21 / −0.40.
  - Causal predictors did not beat a payload-aware rule.
- **R1a** (token / hybrid / ROI vs image by bit budget): INCONCLUSIVE.
- **R1b-0a** (semantic feedback loop): H1 FAIL, no headroom (CPU-bound, the trigger never skipped).

### 4.9 Huang-style UAV image delivery (Direction A, A2–A6-L1): 06–10/11, CLOSED

`phases/2026-11-06_uav-semcom-a/`. The basis of Paper A.

- **Environment:** UAV camera → onboard encoder → air-to-ground link → BS. Per-UAV action:
  - move (11 options: hover, 8 compass directions, next waypoint, return to BS);
  - channel, power level;
  - mode vector.
- **A3 correction:** a slot bug (one image per UAV per slot) created a fake "pacing" story.
- **A3.2** (3 seeds): joint PPO +2.6 vs tuned heuristic. Movement +1.0..1.9 (detours instead of hovering), channel +
  power ≈ +1, mode 0.
- **A4** (native-resolution survey, 20 m/s): joint PPO +4.9.
  - Mode adapted to the queue (+4.2) + 3 UAVs on 2 channels.
  - **RL-chosen movement and power contribute nothing.**
  - Two rules extracted from PPO get within 0.5.
- **A5:** joint PPO loses to mode-only PPO in all 3 regimes, so mode-only RL from then on. Mode-only +3..+12 in some
  regimes (later explained by weak baselines, A-X8).
- **A5.5:** robustness audit of A5.
- **A6-L1:** a task reward (TP − 0.5 FP) only moved the policy to a cheaper fixed mode.
- **Novelty audit:** Li et al. JSAC 2026 (Huang modes + trajectory + WPT + DSAC) and Si 2023 are close; "Huang + Si"
  is not novel.

### 4.10 Huang JSAC step 0 + proposal A/B/C: 04–06/11, CLOSED

- In Huang's MDP, mode choice is a table lookup: no ≥ 0.02 headroom for RL over the best heuristic, and knowing future
  tasks did not help.
- FloodNet and UAVBench dropped: questions are generated from labels, and a compact summary answers 94–99 %.
- `reports/wireless-abc.md`: synthesis of directions A/B/C (06/11).

### 4.11 UAV3D cooperative perception: 04/11, FROZEN by the 06/11 pivot

`phases/2026-11-04_uav3d/`.
- **Data:** UAV3D v1.0-mini, 20 scenes × 20 frames, 5 UAVs, synthetic (CARLA + AirSim), fixed trajectories; 3 dev scenes used.
- **Pipeline:** real codec, causal ledger, 3-frame cache, noisy-OR fusion, KF tracking.
- **Results:** see facts 1–2 in section 3.
  - Velocity in the message: +0.07 AMOTA even without delay.
  - Receiver-side delay compensation keeps only 28–33 % of the mAP gain and 0 % of the AMOTA gain at 2-frame delay.
- **The stronger-estimator test was never run** (it needed a GPU detection dump on 17 train scenes).

### 4.12 Schrödinger Bridge on CoPerception-UAV Town3: 30/10–03/11, CLOSED

`phases/2026-10-30_sb-town3/`.
- **SB decoder vs mapping** (3 seeds): no SB variant passes. At 15 % sending:
  - mapping 0.610 mIoU;
  - best SB 0.602.
- **Temporal memory** at 15 %: U-Net 0.685 vs per-frame mapping 0.609; simple memory fill 0.679.
- **Block selection at 15 % budget:**
  - UAV-side replica of the receiver 0.760 mIoU vs random 0.676;
  - GT receiver state 0.763;
  - with all ACKs lost 0.744.
- **Map baseline:** about 0.771 mIoU at about 0.35 kbit/frame (ideal code length) vs about 8.4 kbit for latents.
- **Setup limit:** one source per region (N = 1); the complementary-evidence question was never tested.

## 5. Assets and constraints

**Code** (`swarmbridge/experiments/`):
- `uav_semcom/`:
  - `env.py`, `env_la` (block LA PHY), `ja_sched.py`;
  - `a9_mpc.py` (calibrated selective MPC);
  - A-X / A-T1 / step C runners;
  - 14-mode table with measured payload / PSNR / detector TP.
- `uav3d_comm/`: box / KF messages, fusion, ledger, AMOTA.
- `jepa_semcom/`.

**Results:** `runs/uav_semcom/`, `runs/jepa_semcom/`.

**Local data (not in git):**
- UAV3D mini (`data/v1.0-mini`, 7.1 GB);
- CoPerception-UAV Town3 (20 GB);
- VisDrone DET val / test-dev / VID;
- MDMT (verbal permission only, unofficial mirror: never publish images, crops or embeddings);
- CrossTrack-UAV requires a signed agreement.

**Operations:**
- The repo is public.
- PPO runs on CPU; GPU jobs go through a lock.
- `runs/active_recovery/` is another agent's branch and is off-limits.

**Used test seeds** (do not reuse for new tests): 7000–7099, 8000–8999, 9000–9099, 9200–9299, 9600–9699,
9800–9999, 12000–12099, 13000–13099, 14000–14099, 20000–26099 (full data).

## 6. Process lessons (agreed with the user)

1. **The problem was narrowed step by step.**
   - Cooperation was dropped (06/11), then movement frozen (A5), then the scheduler frozen (A-X9).
   - Each narrowed sub-problem was then gated with pre-registered tests.
   - The resulting "no RL advantage" results hold only for a near-myopic problem.
   - Rule from now on: reduce scale, never the mechanism under study.
2. **Exploration stage:** prototype with a few meaningful baselines. Only fatal flaws block (leakage, using the future,
   unfair comparisons). Publication-level gates are for the validation stage.
3. **Reviewer errors to avoid:**
   - Review #48 first switched the topic to target tracking, then retracted it.
   - It then claimed joint mobility + communication + compression "was never tried", based on the A-X2 report only.
     It was tried in A3.2 / A4 / A5 (section 4.9).
   - The design review itself first called UAV3D "frozen without negative results"; section 5c of that review corrects
     this.
   - Check claims against the phase reports first.
4. Keep all negative results and the corrections table. Report in Vietnamese to the user, explaining the flow before
   the numbers.
