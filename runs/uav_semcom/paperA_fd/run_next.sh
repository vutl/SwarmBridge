#!/usr/bin/env bash
# Sequential chain (frozen pre-registrations a1b09178): ablation -> Phase 4 -> Phase 3 zero-shot -> Phase 3 training ->
# Phase 3 retrained evaluation, each followed by its pre-registered analysis. RL training never overlaps an MPC benchmark.
# Stops at the first failure. Resumable at the step level: a step whose summary log exists is skipped.
cd /mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom || exit 1
PY=/home/uservutl/envs/semcom-a/bin/python
OUT=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_fd
LOG=$OUT/next_chain.log
step() {   # name, done-marker file, command...
  local name=$1 marker=$2; shift 2
  if [ -f "$marker" ]; then echo "[$(date '+%F %T')] skip $name (done)" >> $LOG; return 0; fi
  echo "[$(date '+%F %T')] start $name" >> $LOG
  "$@" > $OUT/chain_$name.log 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] end $name exit $rc" >> $LOG
  if [ $rc -ne 0 ]; then echo "CHAIN STOPPED at $name" >> $LOG; exit 1; fi
}
echo "[$(date '+%F %T')] chain start at commit $(git -C /mnt/c/Users/USER/Documents/SwarmBridge rev-parse --short HEAD)" >> $LOG
step ablation_test     $OUT/ablation/test_F.json                 $PY fd_next.py ablation test --procs 6
step ablation_summary  $OUT/ablation/summary_ablation_test.log   $PY fd_next_summary.py ablation --name test
step grid_test         $OUT/phase4/test_F.json                   $PY fd_next.py grid test --procs 6
step grid_summary      $OUT/phase4/summary_grid_test.log         $PY fd_next_summary.py grid --name test
step rl_zeroshot       $OUT/phase3/zeroshot_TD.json              $PY fd_next.py rl zeroshot --procs 6
step rl_zs_summary     $OUT/phase3/summary_rl_zeroshot.log       $PY fd_next_summary.py rl --name zeroshot --pools TD
step rl_training       $OUT/phase3/selection.json                $PY fd_phase3_queue.py
step rl_retrained      $OUT/phase3/retrained_TD_all.json         $PY fd_next.py rl retrained --procs 6 --skip-existing
step rl_rt_summary     $OUT/phase3/summary_rl_retrained.log      $PY fd_next_summary.py rl --name retrained
echo "[$(date '+%F %T')] CHAIN EXIT 0" >> $LOG
