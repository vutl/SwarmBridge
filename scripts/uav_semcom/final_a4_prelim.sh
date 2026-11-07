#!/bin/bash
# Fresh-seed (5000..5099) evaluation of the PRELIMINARY (per-step discount) A4 models + all heuristics.
W=/home/uservutl/work
R=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/a4_native
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom
M=""
for s in 0 1 2; do M="$M --model joint_s$s=$W/a4_v20_joint_s$s/ppo_s$s.zip:joint --model movement_s$s=$W/a4_v20_movement_s$s/ppo_s$s.zip:movement"; done
bash $S/final_a4.sh $W/final_prelim.json --slot-heur pacedB1.5 --procs 2 $M > $W/final_prelim.log 2>&1
mkdir -p $R/final; cp $W/final_prelim.json $R/final/prelim_joint_movement_heur.json; echo FINAL_PRELIM_DONE
