#!/usr/bin/env bash
# Matched separated-source controls; one GPU job at a time via shared Windows lock.
set -euo pipefail
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
ENV="$HOME/envs/uav3d"
export CUDA_HOME="$HOME/cuda-11.3" PATH="$HOME/envs/gcc9bin:$HOME/cuda-11.3/bin:$ENV/bin:/usr/local/bin:/usr/bin:/bin"
cd "$HOME/work/uav3d_collab"
CFG=configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/lowerbound/default.yaml
X="$REPO/swarmbridge/experiments/uav3d_comm"
R="$REPO/runs/uav3d_comm"
OUT="$REPO/runs/uav3d_separate_pilot"
mkdir -p "$OUT"
for seed in 0 1 2; do
  for mode in noage age; do
    destination="$OUT/feat_separate_${mode}_s${seed}"
    if [[ -f "$destination/decoder.pt" && -f "$destination/delay2/results_nusc.json" ]]; then
      echo "SKIP completed $destination"
      continue
    fi
    flags=()
    if [[ "$mode" == age ]]; then flags+=(--age); fi
    echo "START seed=$seed mode=$mode $(date -Is)"
    python "$X/gpulock_run.py" "$REPO" -- python "$X/train_separate.py" \
      "$CFG" "$REPO/external/uav3d_ckpt/lowerbound.pth" "$HOME/work/bev_train" "$HOME/work/bev_heldout" \
      "$R/lowerbound/infos_all.pkl" "$R/lowerbound" "$R/feat/drops.json" "$OUT" \
      --arm feat --seed "$seed" --robust --cap 500 --tag "_separate_${mode}" "${flags[@]}"
  done
done
python "$X/eval_dir.py" "$REPO/data/v1.0-mini" "$OUT"
echo "PILOT_DONE $(date -Is)"
