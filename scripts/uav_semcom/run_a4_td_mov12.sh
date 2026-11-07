#!/bin/bash
# Corrected batch, movement-only seeds 1-2 re-run (objective identical to the preliminary runs, verified bit-identical
# on seed 0) only to get the per-eval checkpoints and experience log needed for the matched-experience comparison.
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom
R=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/a4_native
W=/home/uservutl/work
for s in 1 2; do
  O=$W/a4td_v20_movement_s$s
  bash $S/run_a4.sh $O 2e6 $s 20 movement 0.3 --slot-heur pacedB1.5 --time-discount > $O.log 2>&1
  mkdir -p $R/td_v20_movement_s$s; cp $O/eval_s$s.json $R/td_v20_movement_s$s/
  grep '^ppo ' $O.log
done
echo MOV12_DONE
