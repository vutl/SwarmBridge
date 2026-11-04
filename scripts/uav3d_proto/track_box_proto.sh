#!/usr/bin/env bash
# Protocol v1: track every box_proto prediction (dev scenes) with dt = 0.2 s simulated time. CPU only.
# ONLY_NEW=1: skip predictions whose tracking/<name>/input.json already records the same sha256.
set -uo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
export PATH="$HOME/envs/uav3d/bin:/usr/local/bin:/usr/bin:/bin"
export TRACK_DT=0.2
cd "$HOME/work/uav3d_track"
B="${BOX_DIR:-$REPO/runs/uav3d_proto/box}"   # BOX_DIR: another protocol output dir (e.g. runs/uav3d_proto/budget)
args=()
for f in "$B"/dets/*/*/results_nusc.json; do
  rel=${f#"$B/dets/"}; name=${rel%/results_nusc.json}; name=${name//\//__}
  if [ "${ONLY_NEW:-0}" = 1 ] && [ -f "$B/tracking/$name/metrics_summary.json" ]; then
    old=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['det_sha256'])" "$B/tracking/$name/input.json")
    new=$(sha256sum "$f" | cut -d' ' -f1)
    [ "$old" = "$new" ] && continue
  fi
  args+=("$name=$f")
done
python -u "$REPO/swarmbridge/experiments/uav3d_comm/track_run.py" "$B/tracking" "${args[@]}" 2>&1 \
  | grep --line-buffered -E 'TRACK |Error|Traceback|SystemExit' | sed 's/.*TRACK/TRACK/'
echo RUN_DONE track_box_proto
