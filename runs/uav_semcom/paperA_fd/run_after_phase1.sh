#!/usr/bin/env bash
# After the GPU profiling job (phase1.log ends with 'EXIT'): A5-protocol CPU timing of the learned codecs
# (fd_cpu_timing.py, 6 threads, no concurrent GPU job), then the Phase 1 summary. Launched detached (WMI).
OUT=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_fd
cd /mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom || exit 1
PY=/home/uservutl/envs/semcom-a/bin/python
{
  echo "== waiting for phase1.log EXIT $(date '+%F %T')"
  until grep -q "^EXIT" $OUT/phase1.log 2>/dev/null; do sleep 60; done
  grep "^EXIT" $OUT/phase1.log
  if ! grep -q "^EXIT 0" $OUT/phase1.log; then echo "phase 1 failed; not continuing"; exit 1; fi
  while pgrep -f "fd_profile.py" > /dev/null; do sleep 10; done
  echo "== cpu timing $(date +%T)"
  $PY fd_cpu_timing.py --k 10
  echo "== summary $(date +%T)"
  $PY fd_phase1_summary.py
  echo "AFTER_EXIT $? $(date '+%F %T')"
} >> $OUT/after_phase1.log 2>&1
