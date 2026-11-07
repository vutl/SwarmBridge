#!/bin/bash
# A4 batch 2 (ablation): movement-only RL (modes by event-greedy) and mode-only RL (slot decisions by the heuristic
# tuned on train seeds in batch 1), seeds 0 1 2, 2M steps, per speed; same config as batch 1.
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom
R=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/a4_native
W=/home/uservutl/work
PY=/home/uservutl/envs/semcom-a/bin/python
for v in 20 10; do
  H=$($PY -c "import json; print(json.load(open('$R/tune_v${v}_wt0.3.json'))['tuned'])")
  for part in movement mode; do
    for s in 0 1 2; do
      O=$W/a4_v${v}_${part}_s$s
      bash $S/run_a4.sh $O 2e6 $s $v $part 0.3 --slot-heur $H > $W/a4_v${v}_${part}_s$s.log 2>&1
      mkdir -p $R/v${v}_${part}_s$s; cp $O/eval_s$s.json $R/v${v}_${part}_s$s/
      tail -n 2 $W/a4_v${v}_${part}_s$s.log
    done
  done
done
echo BATCH2_DONE
