#!/bin/bash
# A3.2 on env v2.1: MaskablePPO, predicted mode info, w_time 0.3, training seeds 0 1 2 (2M steps each, CPU only)
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom/run_a3.sh
for s in 0 1 2; do
  bash $S /home/uservutl/work/a32_predicted_wt0.3_s$s 2e6 30 $s '{"mode_info":"predicted","w_time":0.3}'
done
echo BATCH_DONE
