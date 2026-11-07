#!/bin/bash
# Usage: run_ev.sh OUT_DIR STEPS SEED CFG_JSON [extra train_ev.py args...]   (reference 512^2 A1 mode table)
set -e
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
R=$REPO/runs/uav_semcom
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
MODES=cap_t2i,thumb64_up,cap_thumb64_i2i_t0.6,cap_thumb128_i2i_t0.3,avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25
OUT=$1; STEPS=$2; SEED=$3; CFG=$4; shift 4
/home/uservutl/envs/semcom-a/bin/python train_ev.py "$OUT" --profiles $R/a11/profile.jsonl $R/a12/profile.jsonl \
  --cpu-file $R/a11/cpu_learned.jsonl --modes $MODES --steps "$STEPS" --seed "$SEED" --cfg "$CFG" "$@"
