#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
ENV="$HOME/envs/active-recovery"
export YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR="$ENV/yolo-config"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$REPO/swarmbridge/experiments/active_recovery${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
"$ENV/bin/python" "$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py" "$REPO" -- \
  timeout --signal=INT --kill-after=10s 1800 "$ENV/bin/python" -u \
  "$REPO/swarmbridge/experiments/active_recovery/run_gate_a.py" "$REPO" "$@" \
  2>&1 | tee "$REPO/runs/active_recovery/gate_a.log"
