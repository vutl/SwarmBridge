#!/bin/bash
# A4: tune the heuristic family on train images / train seeds of the survey workload (native table)
# Usage: tune_a4.sh OUT_JSON V_MAX W_TIME
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
N=$REPO/runs/uav_semcom/a4_native
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
MODES=cap_t2i,thumb64_up,cap_thumb64_i2i_t0.6,cap_thumb128_i2i_t0.3,avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25
nice -n 10 /home/uservutl/envs/semcom-a/bin/python tune_ev.py "$1" --env survey --profiles $N/a11/profile.jsonl $N/a12/profile.jsonl \
  --cpu-file $N/a11/cpu_learned.jsonl --modes $MODES --cfg "{\"mode_info\":\"predicted\",\"v_max\":$2,\"w_time\":$3}"
