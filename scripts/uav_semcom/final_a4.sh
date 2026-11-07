#!/bin/bash
# A4 confirmation on fresh test seeds 5000..5099 (final_a4.py). Usage: final_a4.sh OUT_JSON [extra final_a4.py args]
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
N=$REPO/runs/uav_semcom/a4_native
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
MODES=cap_t2i,thumb64_up,cap_thumb64_i2i_t0.6,cap_thumb128_i2i_t0.3,avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25
OUT=$1; shift
/home/uservutl/envs/semcom-a/bin/python final_a4.py "$OUT" --profiles $N/a11/profile.jsonl $N/a12/profile.jsonl \
  --cpu-file $N/a11/cpu_learned.jsonl --modes $MODES --cfg '{"mode_info":"predicted","v_max":20,"w_time":0.3}' "$@"
