#!/usr/bin/env bash
# A0: Fesl DM checkpoints on <= 1024 published test channels (CPU). Usage: bash run_a0.sh
set -uo pipefail
R=/mnt/c/Users/USER/Documents/SwarmBridge
cd /tmp
for m in 3gpp quadriga_LOS; do
  ~/envs/wireless-abc/bin/python -I "$R/swarmbridge/experiments/wireless_abc/a0_fesl.py" \
    "$R/pdf/GithubModels/benediktfesl__Diffusion_channel_est" "$R/data/wireless_abc/fesl/bin" "$R/runs/wireless_abc/a0" \
    --model "$m" --n 1024 2>&1 | grep -av '%|'
done
