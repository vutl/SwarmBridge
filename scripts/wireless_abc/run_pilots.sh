#!/usr/bin/env bash
# Full pilot runs on trace_v1. Usage: bash run_pilots.sh a1|a1long|b|battn|c
set -uo pipefail
R=/mnt/c/Users/USER/Documents/SwarmBridge
PY=~/envs/wireless-abc/bin/python
T=$R/data/wireless_abc/trace_v1
cd "$R/swarmbridge/experiments/wireless_abc"
case "$1" in
  a1) $PY ../uav3d_comm/gpulock_run.py "$R" -- $PY a1_bench.py "$T" "$R/runs/wireless_abc/a1" --seeds 0 1 2 --device cuda ;;
  a1long) $PY ../uav3d_comm/gpulock_run.py "$R" -- $PY a1_bench.py "$T" "$R/runs/wireless_abc/a1_long" --seeds 0 1 2 --device cuda --steps 15000 --nps 4 ;;
  battn) $PY ../uav3d_comm/gpulock_run.py "$R" -- $PY b_attn.py "$T" "$R/runs/wireless_abc/b_attn" --seeds 0 1 2 --device cuda ;;
  b)  $PY b_coop.py "$T" "$R/runs/wireless_abc/b" --seeds 0 1 2 ;;
  b_et) $PY b_coop.py "$R/data/wireless_abc/trace_etoile" "$R/runs/wireless_abc/b_etoile" --seeds 0 1 2 ;;
  c_et) $PY c_feas.py "$R/data/wireless_abc/trace_etoile" "$R/runs/wireless_abc/c_etoile" --first_ep 96 ;;
  c)  $PY c_feas.py "$T" "$R/runs/wireless_abc/c" --first_ep 96 ;;
esac
