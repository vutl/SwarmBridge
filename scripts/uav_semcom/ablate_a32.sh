#!/bin/bash
# A3.2: component-wise mix of each training seed's PPO with the train-tuned heuristic (pacedB1.5)
R=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom/ablate_a3.sh
CFG='{"mode_info":"predicted","w_time":0.3}'
for s in 0 1 2; do
  W=/home/uservutl/work/a32_predicted_wt0.3_s$s
  nice bash $S $W/ppo_s$s.zip $R/a32_predicted_wt0.3_s$s/ablate.json pacedB1.5 "$CFG" > /home/uservutl/work/ablate_a32_s$s.log 2>&1 &
done
wait
echo ABLATE_DONE
