#!/bin/bash
# After the main CPU queue: final evaluation (run_a5_final.sh), then the EXTENDED-TRAINING experiment (decided
# 2026-11-08 02:00 after K5 joint s0's validation curve was still rising at 2M; separate from the pre-registered
# main runs): K5 joint seeds 0-2 continued from their 2M checkpoint to 5M steps (a5_train.py --resume), checkpoint
# chosen by the same best-validation rule over the whole 5M log, evaluated on the test pool (paired with the rest).
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
A5=$REPO/runs/uav_semcom/a5
W=/home/uservutl/work/a5
export CUDA_VISIBLE_DEVICES=""
bash $REPO/scripts/uav_semcom/run_a5_final.sh || exit 1
cd $REPO/swarmbridge/experiments/uav_semcom
for s in 0 1 2; do
  N=K5_joint_long_s$s
  [ -f $A5/train/$N/train.json ] && continue
  $PY a5_train.py $W/$N --regime K5 --rl-part joint --seed $s --steps 5e6 --resume $W/K5_joint_s$s > $W/$N.log 2>&1 \
    || { echo "TRAIN FAILED $N"; exit 1; }
  mkdir -p $A5/train/$N; cp $W/$N/train.json $A5/train/$N/
  echo "$N $(tail -n 2 $W/$N.log | head -n 1) $(date)"
done
P=""
for s in 0 1 2; do
  B=$($PY -c "import json; print(json.load(open('$A5/train/K5_joint_long_s$s/train.json'))['best_steps'])")
  F=$W/K5_joint_long_s$s/ckpt_$B.zip; [ -f $F ] || F=$W/K5_joint_s$s/ckpt_$B.zip
  P="$P,P:joint_long_best_s$s=$F:joint:T~best:PPPP,P:joint_long_last_s$s=$W/K5_joint_long_s$s/ckpt_5000000.zip:joint:T~best:PPPP"
done
[ -f $A5/final/test_ppo_K5_long.json ] || $PY a5_eval.py eval $A5/final/test_ppo_K5_long.json --regimes K5 --pool test \
  --seeds 7000:7100 --policies ${P#,} > $A5/final/test_ppo_K5_long.log 2>&1
echo A5_AFTER_DONE
