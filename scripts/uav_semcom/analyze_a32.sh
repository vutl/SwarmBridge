#!/bin/bash
# A3.2: behaviour analysis of each training seed's PPO next to the tuned heuristic (pacedB1.5), 10 test seeds
R=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom/analyze_a3.sh
CFG='{"mode_info":"predicted","w_time":0.3}'
for s in 0 1 2; do
  nice bash $S /home/uservutl/work/a32_predicted_wt0.3_s$s/ppo_s$s.zip $R/a32_predicted_wt0.3_s$s/analysis.json "$CFG" pacedB1.5 \
    > /home/uservutl/work/analyze_a32_s$s.log 2>&1 &
done
wait
echo ANALYZE_DONE
