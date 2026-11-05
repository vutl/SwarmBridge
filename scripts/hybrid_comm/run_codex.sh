#!/usr/bin/env bash
# Complete resumable matched pilot. CPU tracking workers never share a job.
set -euo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
OUT="$REPO/runs/hybrid_comm/codex_m2"
X="$REPO/swarmbridge/experiments/hybrid_comm"
PY="$HOME/envs/wireless-abc/bin/python"
mkdir -p "$OUT"
cd "$X"
"$PY" -B -m pytest test_core.py test_receiver.py test_tilecodec.py -q -p no:cacheprovider
"$PY" -B reproduce.py "$REPO" "$OUT/reproduction"
bash "$REPO/scripts/hybrid_comm/run_matched.sh" prepare
bash "$REPO/scripts/hybrid_comm/run_matched.sh" train
bash "$REPO/scripts/hybrid_comm/run_matched.sh" infer
"$PY" -B "$X/calibrate_merge.py" "$REPO" "$OUT"
cd "$HOME/work/uav3d_track"
export TRACK_DT=0.2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
jobs=()
for part in 0 1 2 3; do
  "$HOME/envs/uav3d/bin/python" -u "$X/track_bundle.py" "$REPO" "$OUT/tracking_manifest.json" "$OUT/tracking" \
    --part "$part" --parts 4 > "$OUT/tracking_worker_$part.log" 2>&1 &
  jobs+=("$!")
done
status=0
for pid in "${jobs[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]] || exit 1
"$PY" -B "$X/summarize_matched.py" "$REPO" "$OUT"
