#!/usr/bin/env bash
# Phase 1 smoke tests (review of 50ce9022): 5 DET-val images reproduced with the old A5 seeds; 20 rl_train + 5 F
# images for timing (profile only / profile + detector). Separate output dirs; the main cache is not touched.
cd /mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom || exit 1
PY=/home/uservutl/envs/semcom-a/bin/python
OUT=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_fd
{
  echo "== repro $(date +%T)"
  $PY fd_profile.py repro $OUT/smoke_repro --n 5
  echo "== timing rl_train $(date +%T)"
  $PY fd_profile.py run $OUT/smoke_timing --pools rl_train --limit 20
  echo "== timing F (with detector) $(date +%T)"
  $PY fd_profile.py run $OUT/smoke_timing --pools F --limit 5
  echo "== resume check (must be 'nothing to do') $(date +%T)"
  $PY fd_profile.py run $OUT/smoke_timing --pools rl_train --limit 20
  echo "EXIT $? $(date +%T)"
} > $OUT/smoke.log 2>&1
