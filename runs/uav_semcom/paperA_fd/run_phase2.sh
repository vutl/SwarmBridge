#!/usr/bin/env bash
# Phase 2 test run (frozen pre-registration 66dd28c4): 6 arms x 4 pools x 100 seeds, then the pre-registered summary.
cd /mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom || exit 1
PY=/home/uservutl/envs/semcom-a/bin/python
OUT=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_fd
{
  echo "== phase 2 test start $(date '+%F %T') commit $(git -C /mnt/c/Users/USER/Documents/SwarmBridge rev-parse --short HEAD)"
  $PY fd_phase2.py test --procs 6 || { echo "EXIT 1 test failed $(date +%T)"; exit 1; }
  echo "== summary $(date +%T)"
  $PY fd_phase2_summary.py --name test || { echo "EXIT 1 summary failed $(date +%T)"; exit 1; }
  echo "EXIT 0 $(date '+%F %T')"
} > $OUT/phase2_test.log 2>&1
