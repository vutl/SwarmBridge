#!/bin/bash
# Fresh-seed (5000..5099) evaluation of the CORRECTED (time-discount) A4 models and the preliminary mode-only models.
# Matched-experience checkpoints: first checkpoint whose training episodes >= mean final episodes of the corrected
# joint runs (6366): mode-only ckpt_1000000 (6498 episodes), movement-only ckpt_1200000 (s0: 7295).
# Usage: final_a4_td.sh OUT_NAME SEEDS_FOR_MOVEMENT (e.g. "0" or "1 2") [joint]
W=/home/uservutl/work
R=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/a4_native
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom
M=""
for s in $2; do
  M="$M --model td_movement_s$s=$W/a4td_v20_movement_s$s/ppo_s$s.zip:movement"
  M="$M --model td_movement_matched_s$s=$W/a4td_v20_movement_s$s/ckpt_1200000.zip:movement"
done
if [ "$3" == "joint" ]; then
  for s in 0 1 2; do
    M="$M --model td_joint_s$s=$W/a4td_v20_joint_s$s/ppo_s$s.zip:joint"
    M="$M --model td_mode_s$s=$W/a4td_v20_mode_s$s/ppo_s$s.zip:mode"
    M="$M --model td_mode_matched_s$s=$W/a4td_v20_mode_s$s/ckpt_1000000.zip:mode"
    M="$M --model mode_s$s=$W/a4_v20_mode_s$s/ppo_s$s.zip:mode"
  done
fi
bash $S/final_a4.sh $W/$1.json --slot-heur pacedB1.5 --procs 2 --no-heur $M > $W/$1.log 2>&1
cp $W/$1.json $R/final/$1.json; echo "$1 DONE"
