#!/usr/bin/env bash
# Phase 1 full profiling (fd_v1), resumable: rerunning skips every (image, mode, seed, config) already written.
# Pools in manifest order; recall pools get detector fidelity in the same pass. Then CPU timing of learned codecs.
cd /mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom || exit 1
PY=/home/uservutl/envs/semcom-a/bin/python
OUT=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_fd
{
  echo "== phase 1 start $(date '+%F %T')"
  for P in rl_train rl_val F TD TD_extra V; do
    echo "== pool $P $(date +%T)"
    $PY fd_profile.py run $OUT/profile --pools $P || { echo "FAILED pool $P"; echo "EXIT 1 $(date +%T)"; exit 1; }
  done
  echo "== cpu timing $(date +%T)"
  $PY fd_profile.py cpu $OUT/profile --k 10
  echo "== coverage $(date +%T)"
  $PY fd_profile.py run $OUT/profile
  echo "EXIT 0 $(date '+%F %T')"
} >> $OUT/phase1.log 2>&1
