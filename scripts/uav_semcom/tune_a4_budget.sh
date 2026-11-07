#!/bin/bash
# A4: tune the backlog-budget mode heuristic family (mode_heur.py: slot heuristic x budget_mode(tau)) on train seeds,
# both speeds; test-seed returns of every member are stored for paired comparison with PPO.
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
N=$REPO/runs/uav_semcom/a4_native
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
MODES=cap_t2i,thumb64_up,cap_thumb64_i2i_t0.6,cap_thumb128_i2i_t0.3,avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25
for v in 20 10; do
  nice -n 10 /home/uservutl/envs/semcom-a/bin/python tune_ev.py $N/tune_budget_v${v}_wt0.3.json --family budget --procs 3 \
    --env survey --profiles $N/a11/profile.jsonl $N/a12/profile.jsonl --cpu-file $N/a11/cpu_learned.jsonl --modes $MODES \
    --cfg "{\"mode_info\":\"predicted\",\"v_max\":$v,\"w_time\":0.3}"
done
echo TUNE_BUDGET_DONE
