#!/usr/bin/env bash
# Hybrid object-feature pilot, milestone 0/1a: tile codec + feature receiver sanity (legacy UAV3D env, GPU lock).
# Usage: bash run_m0.sh OUT_SUBDIR [--frames N]
set -euo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
ENV="$HOME/envs/uav3d"
export CUDA_HOME="$HOME/cuda-11.3" PATH="$HOME/envs/gcc9bin:$HOME/cuda-11.3/bin:$ENV/bin:/usr/local/bin:/usr/bin:/bin"
cd "$HOME/work/uav3d_collab"
CFG=configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/lowerbound/default.yaml
R="$REPO/runs/uav3d_comm"
OUT="$REPO/runs/hybrid_comm/$1"; shift
mkdir -p "$OUT"
python "$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py" "$REPO" -- python "$REPO/swarmbridge/experiments/hybrid_comm/m0_receiver.py" \
  "$CFG" "$REPO/external/uav3d_ckpt/lowerbound.pth" "$R/lowerbound/infos_all.pkl" "$HOME/work/bev_heldout" \
  "$R/fusion/feat_s1/decoder.pt" "$R/fusion/ego_s2/decoder.pt" "$OUT" --cap 500 "$@"
