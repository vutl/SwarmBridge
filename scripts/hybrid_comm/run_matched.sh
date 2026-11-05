#!/usr/bin/env bash
# Each GPU phase holds the same Windows/WSL lock; does not alter old environments.
set -euo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
ENV="$HOME/envs/uav3d"
export CUDA_HOME="$HOME/cuda-11.3"
export PATH="$HOME/envs/gcc9bin:$HOME/cuda-11.3/bin:$ENV/bin:/usr/local/bin:/usr/bin:/bin"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
X="$REPO/swarmbridge/experiments/hybrid_comm"
OUT="$REPO/runs/hybrid_comm/codex_m2"
mkdir -p "$OUT"
cd "$HOME/work/uav3d_collab"
case "${1:?prepare|train|infer|profile}" in
  prepare) shift; task=prepare_matched.py ;;
  train) shift; task=train_matched.py ;;
  infer) shift; task=infer_matched.py ;;
  profile) shift; task=profile_inputs.py ;;
  *) exit 2 ;;
esac
limit=14400
[[ "$task" == train_matched.py ]] && limit=2700
python "$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py" "$REPO" -- \
  timeout --signal=INT --kill-after=5s "$limit" python -u "$X/$task" "$REPO" "$OUT" "$@" \
  2>&1 | tee -a "$OUT/${task%.py}.log" | tr '\r' '\n' \
  | grep --line-buffered -E '^GPU|^PREP|^TRAIN|^VALIDATION|^BENCHMARK|^MATCHED|^INFER|^INPUT_PROFILE|^SKIP|Traceback|Error|line [0-9]+'
