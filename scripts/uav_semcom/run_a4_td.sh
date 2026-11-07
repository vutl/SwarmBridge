#!/bin/bash
# A4 corrected runs: time-consistent (semi-MDP) discounting, gamma = 0.99 and GAE lambda = 0.95 PER SECOND of env time
# (train_ev.py --time-discount). 20 m/s only, seeds 0 1 2, joint / mode-only / movement-only, same config, steps and
# hyper-parameters as batch 1/2; mode-only slot heuristic = the one tuned on train seeds in batch 1.
# Movement-only RL steps are exactly 1 s apart, so its objective is unchanged by the fix (tested bit-identical GAE):
# seed 0 is re-run as a check; if its test returns equal the preliminary run's, seeds 1-2 are reused, else re-run.
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom
R=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/a4_native
W=/home/uservutl/work
PY=/home/uservutl/envs/semcom-a/bin/python
v=20
H=$($PY -c "import json; print(json.load(open('$R/tune_v${v}_wt0.3.json'))['tuned'])")
run() {   # part seed
  O=$W/a4td_v${v}_$1_s$2
  bash $S/run_a4.sh $O 2e6 $2 $v $1 0.3 --slot-heur $H --time-discount > $O.log 2>&1
  mkdir -p $R/td_v${v}_$1_s$2; cp $O/eval_s$2.json $R/td_v${v}_$1_s$2/
  echo "$1 s$2"; tail -n 2 $O.log | head -n 1; grep '^ppo ' $O.log
}
for s in 0 1 2; do run joint $s; done
for s in 0 1 2; do run mode $s; done
run movement 0
SAME=$($PY -c "
import json
a=json.load(open('$R/td_v${v}_movement_s0/eval_s0.json'))['results']['ppo']
b=json.load(open('$R/v${v}_movement_s0/eval_s0.json'))['results']['ppo']
print(int(max(abs(x['return']-y['return']) for x,y in zip(a,b)) < 1e-9))")
echo "movement s0 identical to preliminary: $SAME"
if [ "$SAME" != "1" ]; then run movement 1; run movement 2; fi
echo TD_DONE
