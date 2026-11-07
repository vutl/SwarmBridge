#!/bin/bash
# A4: two-timescale MaskablePPO on the photogrammetry survey workload, native-resolution mode table (CPU only).
# Usage: run_a4.sh OUT_DIR STEPS SEED V_MAX RL_PART W_TIME [extra train_ev.py args...]
set -e
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
N=$REPO/runs/uav_semcom/a4_native
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
MODES=cap_t2i,thumb64_up,cap_thumb64_i2i_t0.6,cap_thumb128_i2i_t0.3,avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25
OUT=$1; STEPS=$2; SEED=$3; V=$4; PART=$5; WT=$6; shift 6
/home/uservutl/envs/semcom-a/bin/python train_ev.py "$OUT" --env survey --rl-part "$PART" \
  --profiles $N/a11/profile.jsonl $N/a12/profile.jsonl --cpu-file $N/a11/cpu_learned.jsonl --modes $MODES \
  --steps "$STEPS" --seed "$SEED" --cfg "{\"mode_info\":\"predicted\",\"v_max\":$V,\"w_time\":$WT}" "$@"
