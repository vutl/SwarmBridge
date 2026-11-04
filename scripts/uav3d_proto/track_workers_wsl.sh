#!/usr/bin/env bash
# Start N shared-queue tracking workers inside WSL (see track_worker.py). Usage: NJOBS=3 bash track_workers_wsl.sh DIR...
set -uo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
export PATH="$HOME/envs/uav3d/bin:/usr/local/bin:/usr/bin:/bin" TRACK_DT=0.2 TRACK_ROOT=$REPO/data/v1.0-mini/
[ -f "$REPO/runs/uav3d_proto/track_priority.txt" ] && export TRACK_PRIORITY=$REPO/runs/uav3d_proto/track_priority.txt
cd "$HOME/work/uav3d_track"
for w in $(seq 1 "${NJOBS:-3}"); do
  python -u "$REPO/scripts/uav3d_proto/track_worker.py" "$REPO" "wsl$w" "$@" 2>&1 | tr '\r' '\n' \
    | grep --line-buffered -E '^(DONE|FAIL|WORKER_END)|Traceback|Error' &
done
wait
echo RUN_DONE wsl_workers
