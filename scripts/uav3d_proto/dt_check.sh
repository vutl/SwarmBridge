#!/usr/bin/env bash
# Protocol v1 check (Codex review): AMOTA sensitivity to the tracker time step, timestamps (~2.7 s) vs 0.2 s,
# on the same prediction files. CPU only.
set -uo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
export PATH="$HOME/envs/uav3d/bin:/usr/local/bin:/usr/bin:/bin"
cd "$HOME/work/uav3d_track"
R="$REPO/runs/uav3d_comm"
args=("ego=$R/cap500/ego_s2/results_nusc.json" "feat=$R/cap500/feat_s2/results_nusc.json"
      "wnor=$R/box_stress_ft500_wnor/dets/clean/results_nusc.json" "boxft=$R/cap500/boxft_s0/results_nusc.json")
for dt in ts 0.2; do
  if [ "$dt" = ts ]; then unset TRACK_DT; else export TRACK_DT=$dt; fi
  python -u "$REPO/swarmbridge/experiments/uav3d_comm/track_run.py" "$R/tracking_dtcheck/$dt" "${args[@]}" 2>&1 \
    | grep --line-buffered -E 'TRACK|Error|Traceback|SystemExit'
done
echo RUN_DONE dt_check
