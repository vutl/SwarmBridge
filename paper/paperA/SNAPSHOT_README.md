# Reproducibility snapshot: "Predictive mode selection under wireless rate uncertainty for multi-UAV semantic image offloading"

This archive is a **versioned, paper-only extract** of the project repository. It is built by
`paper/paperA/make_snapshot.sh <tag>` with `git archive`, so its content is exactly the tagged commit, restricted to
the paths below.

## Contents

| Path | What it is |
|---|---|
| `paper/paperA/` | Manuscript (`main.tex`, `main.pdf`), Table II (`table_fd.tex`), figures (`figs/fig_fd_*.pdf`) and the claim-to-source map `CLAIMS.md` |
| `swarmbridge/experiments/uav_semcom/` | Simulator (`env.py`, `env_la`), scheduler, selective MPC (`a9_mpc.py`), evaluation (`ax2_eval.py`, `fd_next.py`), pre-registered analyses (`fd_phase2_summary.py`, `fd_next_summary.py`), post-hoc analyses (`fd_posthoc.py`), and table / figure generators (`paperA_fd_numbers.py`, `paperA_fd_figures.py`) |
| `runs/uav_semcom/paperA_fd/` | Stored results of all 27,600 evaluation episodes (per-episode rows and delivery lists), manifest, profile tables, pre-registered summaries, post-hoc outputs, `paper/paper_numbers.json` |
| `runs/uav_semcom/{ax9,paperA_rl,a5,a6l1}/` (selected files) | Small inputs read by the code: frozen scheduler, checkpoint path lists, detector split / hashes, earlier manifest |
| `reports/phases/2026-11-23_paperA-fulldata/` | Protocol, pre-registrations, phase results, consolidated report, post-hoc report, writing report |

**Not included:**
- VisDrone images. Obtain VisDrone2019-DET from the official source under its own terms; the image lists are in the
  manifest.
- Detector and RL weights.
- Anything unrelated to this paper.

## Regenerating the paper's numbers, table and figures (no simulation)

From `swarmbridge/experiments/uav_semcom/`, in an environment with numpy, scipy, matplotlib and the project
dependencies:

```
python fd_posthoc.py verify        # fail-fast coverage / integrity check of the 27,600 episodes
python fd_posthoc.py decomp        # post-hoc return / recall decomposition
python fd_posthoc.py r5            # post-hoc learned-vs-heuristic comparisons
python paperA_fd_numbers.py        # paper_numbers.json + paper/paperA/table_fd.tex
python paperA_fd_figures.py ../../../paper/paperA/figs
```

`fd_posthoc.py verify` checks the sha256 of the manifest and profile first.

**Note on reruns:** re-running the episodes themselves (`fd_next.py`, `fd_phase2.py`) requires the images, the
profiling stage and the trained checkpoints. MPC arms use a wall-clock budget, so reruns are not bit-reproducible.
