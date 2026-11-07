#!/bin/bash
# A5 CPU queue (resumable; each stage skipped if its output exists):
#   tests -> heuristic tuning on the train pool for every regime -> validation sweep (heuristics only) ->
#   pre-registered regime selection (a5_select.py) -> PPO training (joint + mode-only, seeds 0, 1, 2 in that order
#   across the selected regimes; mode-only slots = slot part of the regime's best tuned heuristic, T:best). Final test evaluation is a separate step (run_a5_final.sh).
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
A5=$REPO/runs/uav_semcom/a5
W=/home/uservutl/work/a5
export CUDA_VISIBLE_DEVICES=""
mkdir -p $W
cd $REPO/swarmbridge/experiments/uav_semcom
until grep -q PROFILE_A5_DONE /home/uservutl/work/a5_profile_val.log 2>/dev/null; do sleep 60; done
$PY -m pytest -q test_env.py test_env_ev.py test_env_survey.py test_timeppo.py test_a5.py 2>&1 | tail -3
REGS="ref K2 K4 K5 v10 v15 env_suburban env_dense env_highrise fade_k3 fade_k0 dl_tight dl_loose q_low q_high wt0 wt0.1 wt1 phy_gap3 phy_mcs phy_mcs_blind"
for r in $REGS; do
  [ -f $A5/tune/$r.json ] || $PY a5_eval.py tune --regimes $r --procs 6 || exit 1
done
echo "tuning done $(date)"
POL="T:fixed,H:greedy,T:base,T:budget,T:reuse,H:pacedB1.5+budget1.5,H:greedy+budget1+reuse,T:reuse@oracle"
if [ ! -f $A5/sweep_val.json ]; then
  $PY a5_eval.py eval $A5/sweep_val.json --regimes $(echo $REGS | tr ' ' ,) --pool val --seeds 3000:3050 \
    --policies $POL > $W/sweep_val.log 2>&1 || exit 1
fi
$PY a5_select.py $A5/sweep_val.json $A5/selected.json
R3=$($PY -c "import json; print(json.load(open('$A5/selected.json'))['R3'])")
echo "selected ref K5 $R3 $(date)"
for s in 0 1 2; do
  for r in ref K5 $R3; do
    for part in joint mode; do
      N=${r}_${part}_s$s; O=$W/$N
      if [ -f $A5/train/$N/train.json ]; then continue; fi
      $PY a5_train.py $O --regime $r --rl-part $part --seed $s --heur T:best > $O.log 2>&1 || { echo "TRAIN FAILED $N"; exit 1; }
      mkdir -p $A5/train/$N; cp $O/train.json $A5/train/$N/
      echo "$N $(tail -n 2 $O.log | head -n 1) $(date)"
    done
  done
done
echo A5_CPU_DONE
