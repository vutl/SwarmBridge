#!/bin/bash
# A3.2: tune the heuristic family on train seeds and compare with each training seed's PPO (paired, test seeds)
R=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom/tune_a3.sh
CFG='{"mode_info":"predicted","w_time":0.3}'
for s in 0 1 2; do
  nice bash $S $R/a32_predicted_wt0.3_s$s/tune.json $R/a32_predicted_wt0.3_s$s/eval_s$s.json "$CFG" \
    > /home/uservutl/work/tune_a32_s$s.log 2>&1 &
done
wait
echo TUNE_DONE
