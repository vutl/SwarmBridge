#!/bin/bash
# A6.0-L1 CPU queue (reports/archive/uav-semcom-a/uav-semcom-a6-l1.md section 0). Waits for the GPU fidelity stage and the A5.5 queue.
# Resumable: every output is skipped if present.
#   1. tests (no skips allowed); 2. tune all heuristic families under the task reward (train, 3 regimes);
#   3. PPO mode-only, task reward, 3 seeds x 3 regimes (2M steps, best-val by task return);
#   4. final evaluation, fresh seeds 9000..9099, test + test-dev; 5. video cluster bootstrap of the main comparisons.
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
O=$REPO/runs/uav_semcom/a6l1
A5=$REPO/runs/uav_semcom/a5
W5=/home/uservutl/work/a5
W6=/home/uservutl/work/a6l1
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
until [ -f $O/fidelity/sha256.txt ]; do sleep 60; done
until [ -f $REPO/runs/uav_semcom/a55/boot/testdev_phy_mcs_blind.json ]; do sleep 60; done
echo "inputs ready $(date)"
$PY -m pytest -q -rs test_a6l1.py test_a55.py test_a5.py > $O/tests.log 2>&1 || { echo "TESTS FAILED"; tail -30 $O/tests.log; exit 1; }
grep -q -i skipped $O/tests.log && { echo "TESTS SKIPPED (inputs missing)"; tail -10 $O/tests.log; exit 1; }
echo "tests ok $(date)"
SEL="ref K5 phy_mcs_blind"
need=""
for r in $SEL; do [ -f $O/tune/$r.json ] || need="$need,$r"; done
if [ -n "$need" ]; then
  $PY a6l1_eval.py tune --regimes ${need#,} > $O/tune.log 2>&1 || { echo "TUNE FAILED"; tail -20 $O/tune.log; exit 1; }
fi
echo "tune done $(date)"
mkdir -p $O/train
for s in 0 1 2; do
  for r in $SEL; do
    N=${r}_s$s
    [ -f $O/train/$N/train.json ] && continue
    $PY a6l1_train.py $W6/$N --regime $r --seed $s > $W6/$N.log 2>&1 || { echo "TRAIN FAILED $N"; tail -20 $W6/$N.log; exit 1; }
    mkdir -p $O/train/$N; cp $W6/$N/train.json $O/train/$N/
    echo "$N $(tail -n 2 $W6/$N.log | head -n 1) $(date)"
  done
done
best6() { $PY -c "import json; print(json.load(open('$O/train/$1/train.json'))['best_steps'])"; }
best5() { $PY -c "import json; print(json.load(open('$A5/train/$1/train.json'))['best_steps'])"; }
ppo() {   # $1 regime, $2 = 1 to add last checkpoints
  local r=$1 P=""
  for s in 0 1 2; do
    P="$P,P:task_best_s$s=$W6/${r}_s$s/ckpt_$(best6 ${r}_s$s).zip:mode:T6~best:PPPP"
    P="$P,P:psnr_best_s$s=$W5/${r}_mode_s$s/ckpt_$(best5 ${r}_mode_s$s).zip:mode:T5~best:PPPP"
    [ "$2" = 1 ] && P="$P,P:task_last_s$s=$W6/${r}_s$s/ckpt_2000000.zip:mode:T6~best:PPPP"
  done
  echo ${P#,}
}
ev() { local out=$1; shift; [ -f $out ] || $PY a6l1_eval.py "$@" > ${out%.json}.log 2>&1 || { echo "FAILED $out"; tail -20 ${out%.json}.log; exit 1; }; echo "done $out $(date)"; }
HP="T6:best,T6:tv,T6:reuse,T6:budget,T6:fixed,T5:best,T5:fixed,H:greedy+budget1+reuse"
mkdir -p $O/final $O/boot
for r in $SEL; do
  ev $O/final/test_$r.json eval $O/final/test_$r.json --regimes $r --pool test --policies $HP,$(ppo $r 1)
  ev $O/final/testdev_$r.json eval $O/final/testdev_$r.json --regimes $r --pool testdev --policies $HP,$(ppo $r 0)
done
for r in $SEL; do
  for pool in test testdev; do
    ev $O/boot/${pool}_$r.json boot $O/boot/${pool}_$r.json --regimes $r --pool $pool --policies T6:best,T5:best,$(ppo $r 0)
  done
done
echo A6L1_DONE
