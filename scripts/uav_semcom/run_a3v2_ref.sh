#!/bin/bash
# A3 v2, reference scenario: MaskablePPO (seed 0, 2M steps) for mode_info {predicted (main), oracle (reference)}
# x w_time {0, 0.3}; CPU only.
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom/run_a3.sh
W=/home/uservutl/work
for wt in 0.0 0.3; do
  for mi in predicted oracle; do
    bash $S $W/a3v2_${mi}_wt${wt}_s0 2e6 30 0 "{\"mode_info\":\"$mi\",\"w_time\":$wt}"
  done
done
echo BATCH_DONE
