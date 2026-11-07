#!/bin/bash
# A4 batch 1: per speed, tune the heuristic family (train seeds), then joint two-timescale MaskablePPO, seeds 0 1 2,
# 2M steps each (predicted info, w_time 0.3). Ablations (movement-only / mode-only) follow in batch 2.
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom
R=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/a4_native
W=/home/uservutl/work
for v in 20 10; do
  bash $S/tune_a4.sh $R/tune_v${v}_wt0.3.json $v 0.3 > $W/tune_a4_v$v.log 2>&1
  for s in 0 1 2; do
    O=$W/a4_v${v}_joint_s$s
    bash $S/run_a4.sh $O 2e6 $s $v joint 0.3 > $W/a4_v${v}_joint_s$s.log 2>&1
    mkdir -p $R/v${v}_joint_s$s; cp $O/eval_s$s.json $R/v${v}_joint_s$s/
    tail -n 3 $W/a4_v${v}_joint_s$s.log
  done
done
echo BATCH_DONE
