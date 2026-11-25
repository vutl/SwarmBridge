#!/usr/bin/env bash
# Build the paper-only reproducibility snapshot from a git tag (default: paperA-v1.0).
# Output: paper/paperA/dist/paperA-snapshot-<tag>.zip (+ .sha256). Content = exactly the tagged commit, restricted to
# the paths listed in SNAPSHOT_README.md. Upload to an archive (e.g. Zenodo) is a manual step by the authors.
set -euo pipefail
TAG="${1:-paperA-v1.0}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || { echo "tag $TAG not found"; exit 1; }
mkdir -p paper/paperA/dist
OUT="paper/paperA/dist/paperA-snapshot-$TAG.zip"
PATHS=(
  paper/paperA
  swarmbridge/__init__.py
  swarmbridge/experiments/__init__.py
  swarmbridge/experiments/uav_semcom
  runs/uav_semcom/paperA_fd
  runs/uav_semcom/ax9/sched.json
  runs/uav_semcom/ax9/ppo_la_ckpts.json
  runs/uav_semcom/paperA_rl/ckpts.json
  runs/uav_semcom/a5/manifest.json
  runs/uav_semcom/a5/detector
  runs/uav_semcom/a6l1/fidelity/per_image_val.jsonl
  reports/phases/2026-11-23_paperA-fulldata
)
git archive --format=zip --prefix="paperA-snapshot-$TAG/" -o "$OUT" "$TAG" -- "${PATHS[@]}"
sha256sum "$OUT" | tee "$OUT.sha256"
echo "files: $(unzip -l "$OUT" | tail -1)"
