#!/bin/bash
# A4: behaviour + return decomposition of the joint PPO (each seed) vs the tuned heuristic, 10 test seeds
# Usage: analyze_a4.sh V_MAX
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
N=$REPO/runs/uav_semcom/a4_native
W=/home/uservutl/work
PY=/home/uservutl/envs/semcom-a/bin/python
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
MODES=cap_t2i,thumb64_up,cap_thumb64_i2i_t0.6,cap_thumb128_i2i_t0.3,avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25
v=$1
H=$($PY -c "import json; print(json.load(open('$N/tune_v${v}_wt0.3.json'))['tuned'])")
for s in 0 1 2; do
  nice -n 10 $PY analyze_ev.py $W/a4_v${v}_joint_s$s/ppo_s$s.zip $N/v${v}_joint_s$s/analysis.json --env survey --heur $H \
    --profiles $N/a11/profile.jsonl $N/a12/profile.jsonl --cpu-file $N/a11/cpu_learned.jsonl --modes $MODES \
    --cfg "{\"mode_info\":\"predicted\",\"v_max\":$v,\"w_time\":0.3}" &
done
wait
echo ANALYZE_DONE
