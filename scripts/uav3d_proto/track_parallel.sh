#!/usr/bin/env bash
# Protocol v1.1: track predictions of one or more protocol output dirs in parallel (CPU only, dt = 0.2 s).
# Usage (inside WSL): NJOBS=3 bash track_parallel.sh DIR [DIR ...]   (DIR = e.g. runs/uav3d_proto/box, repo-relative)
# Skips any prediction whose DIR/tracking/<name>/input.json already records the same sha256 (and has metrics).
set -uo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
export PATH="$HOME/envs/uav3d/bin:/usr/local/bin:/usr/bin:/bin"
export TRACK_DT=0.2
NJOBS=${NJOBS:-3}
cd "$HOME/work/uav3d_track"
LIST=$(mktemp)
for d in "$@"; do
  B="$REPO/$d"
  for f in "$B"/dets/*/*/results_nusc.json; do
    rel=${f#"$B/dets/"}; name=${rel%/results_nusc.json}; name=${name//\//__}
    if [ -f "$B/tracking/$name/metrics_summary.json" ] && [ -f "$B/tracking/$name/input.json" ]; then
      old=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['det_sha256'])" "$B/tracking/$name/input.json")
      new=$(sha256sum "$f" | cut -d' ' -f1)
      [ "$old" = "$new" ] && continue
    fi
    echo "$B $name=$f" >> "$LIST"
  done
done
echo "QUEUED $(wc -l < "$LIST") predictions on $NJOBS workers"
for w in $(seq 0 $((NJOBS - 1))); do
  (
    awk -v n="$NJOBS" -v w="$w" 'NR % n == w' "$LIST" | while read -r B arg; do
      python -u "$REPO/swarmbridge/experiments/uav3d_comm/track_run.py" "$B/tracking" "$arg" 2>&1 \
        | tr '\r' '\n' | grep --line-buffered -E 'TRACK |Error|Traceback|SystemExit' | sed 's/.*TRACK/TRACK/'
    done
  ) &
done
wait
rm -f "$LIST"
echo RUN_DONE track_parallel
