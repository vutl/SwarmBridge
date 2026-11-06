#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
ENV="$HOME/envs/active-recovery"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR="$ENV/yolo-config"
cd "$REPO"
timeout --signal=INT --kill-after=10s 1800 "$ENV/bin/python" -u \
 "$REPO/swarmbridge/experiments/active_recovery/paired_cpu.py" "$REPO" "$@" \
 2>&1 | tee -a "$REPO/runs/active_recovery/paired_cpu.log"
