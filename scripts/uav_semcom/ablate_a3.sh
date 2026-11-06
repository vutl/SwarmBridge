#!/bin/bash
# Usage: ablate_a3.sh MODEL.zip OUT_JSON HEUR [CFG_JSON]
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
R=$REPO/runs/uav_semcom
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
MODES=cap_t2i,thumb64_up,cap_thumb64_i2i_t0.6,cap_thumb128_i2i_t0.3,avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25
/home/uservutl/envs/semcom-a/bin/python ablate_policy.py "$1" "$2" --heur "$3" --profiles $R/a11/profile.jsonl $R/a12/profile.jsonl \
  --cpu-file $R/a11/cpu_learned.jsonl --modes $MODES --cfg "${4:-{\}}"
