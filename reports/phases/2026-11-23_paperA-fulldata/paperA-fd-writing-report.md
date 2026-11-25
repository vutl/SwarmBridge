# Paper A: manuscript rewrite for ICT Express (25/11)

**Authorisation:** ChatGPT writing authorisation after reviewing 18ddc9e7. No new simulation, training or tuning
was run. The previous method letter stays in git at `1225e287`.

## 1. Output

| Item | Value |
|---|---|
| Manuscript | `paper/paperA/main.tex` → `paper/paperA/main.pdf` |
| Pages | **4** (A4, elsarticle 3p two-column ICT Express template); font size and margins unchanged; no LaTeX warnings |
| Title | "Uncertainty-aware mode selection for multi-UAV semantic image offloading" |
| Abstract | 100 words (limit 75–100) |
| Keywords | 5, alphabetical |
| Floats | Fig. 1 system / controller; Fig. 2 (a) ablation, (b) return vs recall, (c) post-hoc decomposition; Table 1 network settings + learned incumbents + timing |
| References | 13 |
| Figure / table code | `swarmbridge/experiments/uav_semcom/paperA_fd_numbers.py` (writes `runs/uav_semcom/paperA_fd/paper/paper_numbers.json` and `table_fd.tex`) and `paperA_fd_figures.py` (writes `figs/fig_fd_system.pdf`, `figs/fig_fd_results.pdf`); stored results only |
| Claims → sources | `paper/paperA/CLAIMS.md`, each number marked C (confirmatory), S (supporting) or P (post-hoc) |
| Build | `paper/paperA/build.sh` (WSL pdflatex) |

**Structure**, following the authorised outline:
1. Introduction, with related work and contributions C1–C3.
2. System model and problem formulation, with the objective J written out.
3. Uncertainty-aware mode selection.
4. Experimental setup.
5. Results and discussion: F1, F2, F3, decision time.
6. Conclusions with limitations.

## 2. Reporting corrections applied (from the review)

| Requested correction | How it was applied |
|---|---|
| "53 contrasts, same direction" was too strong | Manuscript and reports now say: 44 / 53 with a positive mean return difference, 41 of them with smaller payloads, exceptions ng − H\* (TD / V / F). Verified from `decomp.json` |
| Decomposition is not an independent exact reconstruction | Reports state that it uses stored aggregates and that the time term is a residual. That residual equals 0.3·K·mission time within 2e-13 in all 27,600 episodes, which is tighter than the reviewer assumed |
| SCE parts cannot be separated | Stated in the Fig. 2 caption and the reports. The manuscript reports the mean-delay difference (−1.2 s) only as a supporting observation |
| R5 is the absence of a detected advantage, not equivalence | "No learned incumbent showed a detected return advantage"; PPO + cal vs H\* + cal "+0.3 [−1.6, +2.1] (not evidence of equivalence)" |
| Keep the zero-shot PPO-LA recall exception | In Section 5 (+3.1 recall, −4.9 return) |
| Separate yield from end-to-end recall | Section 5 splits recall into yield on images compliant under both arms vs a change in the compliant set |
| Mark confirmatory vs supporting vs post-hoc | Table 1 marks (†, \*); the Setup paragraph names the post-hoc analyses; CLAIMS.md gives a status per number |
| No causal generality, deployment readiness or real-time guarantee | "In this simulator, with the weights of (2)"; the limitations sentence; the decision-time paragraph states uncharged time and the overruns |

## 3. Corrections found while writing (checked against code / data)

1. **The "gate" is a cold-start gate.** In `ax2_eval.py`, planning starts only once the error tracker holds 20 s of
   airtime. Earlier user-facing summaries described it as "pausing when the error is large"; that was wrong. The
   manuscript describes it as implemented.
2. **"44–82% of slots over 50 ms sequentially"** (consolidated report) covers TD only. Across all pools it is
   **42–82%**. The manuscript uses 42–82%.
3. **"At most 4%" parallel at K ≤ 5, M ≤ 3** is **4.1%** (F, N5x3). The manuscript says 4.1%.
4. **The nominal model's bias −0.35** came from Phase 2. On the ablation seeds used in F1 it is −0.38, measured by
   the tracker in the guarded arm; the nominal arm has no tracker. The manuscript says this.
5. **The nominal MPC does not have the lowest return of the H = 4 planners.** The gate-only arm is lower (−58.2 vs
   −57.0). The manuscript says "18.1 below the guarded MPC" instead.

## 4. Remaining issues for the final review

1. **Authors, affiliation, e-mail, funding:** still `\todo` (red).
2. **Repository URL in the manuscript** (`github.com/vutl/SwarmBridge`). The repo is public and named after the
   original project. The user should decide whether to cite it, cite an archived snapshot, or remove it. It does not
   contain MDMT data.
3. **ICT Express Guide for Authors not verified.** The official page returned 403 to the reviewer. Unconfirmed: the
   4-page rule, the declaration heading (we use "Conflict of interest", as in the template), a data-availability
   statement, the reference style (the bibliography is manual, in elsarticle-num format, with "et al." after three
   authors to fit 4 pages).
4. **Fig. 1 text is small** (5.2–6.4 pt at full width). It is readable in the PDF but below the usual 7–8 pt.
5. **The reference row of Table 1** uses the ablation seeds (A8 / A7, supporting), so that it matches Fig. 2. The
   confirmatory reference values (Phase 2: +7.3 cal − H\*, −0.1 cal − guarded) are in the text. A reviewer may
   prefer the Phase 2 values in the table.
6. **Parameters taken from the earlier letter** (40 s plan horizon, LA details) were checked against the code
   docstrings. They were not re-derived from first principles in this round.
7. **YOLO11 citation:** software citation (Jocher and Qiu, 2024), with no paper.
8. **No supplementary material** is referenced except the repository.
