#!/usr/bin/env bash
# Protocol v1: track every box_proto prediction (dev scenes) with dt = 0.2 s simulated time. CPU only.
set -uo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
export PATH="$HOME/envs/uav3d/bin:/usr/local/bin:/usr/bin:/bin"
export TRACK_DT=0.2
cd "$HOME/work/uav3d_track"
B="$REPO/runs/uav3d_proto/box"
args=()
for f in "$B"/dets/*/*/results_nusc.json; do
  rel=${f#"$B/dets/"}; name=${rel%/results_nusc.json}; args+=("${name//\//__}=$f")
done
python -u "$REPO/swarmbridge/experiments/uav3d_comm/track_run.py" "$B/tracking" "${args[@]}" 2>&1 \
  | grep --line-buffered -E '^TRACK|Error|Traceback|SystemExit' | sed 's/.*TRACK/TRACK/'
echo RUN_DONE track_box_proto
