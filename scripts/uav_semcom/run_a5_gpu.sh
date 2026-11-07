#!/bin/bash
# A5 GPU queue (one GPU job at a time, shared lock): after the val-pool profile -> fidelity smoke test ->
# detector training (VisDrone-train minus val-overlapping sequences) -> test-dev profile -> task fidelity (val with
# generative modes on the first 100 frames, test-dev). Each stage is skipped if its output exists (resumable).
# Test-dev latencies are NOT used (the A5 table always takes encode/decode times from the val-pool profile), so the
# CPU load from concurrent PPO training does not matter for that profile.
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
LOCK=$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py
A5=$REPO/runs/uav_semcom/a5
W=/home/uservutl/work
VAL=$REPO/data/visdrone/VisDrone2019-DET-val
TD=/home/uservutl/data/visdrone_dl/VisDrone2019-DET-test-dev
WEI=$W/a5_det/yolo11s_visdrone_960/weights/best.pt
X=$REPO/swarmbridge/experiments/uav_semcom
cd $W   # Ultralytics downloads yolo11s.pt / yolo11n.pt (AMP check) into the cwd: keep them out of the repo
until grep -q PROFILE_A5_DONE $W/a5_profile_val.log 2>/dev/null; do sleep 60; done
echo "val profile done $(date)"
if [ ! -f $W/a5_fid_smoke/coco_val.json ]; then
  $PY $LOCK $REPO -- $PY $X/a5_fidelity.py $W/a5_fid_smoke --pool val --image-dir $VAL/images --ann-dir $VAL/annotations \
    --weights yolo11s.pt --n 3 --gen 1 > $W/a5_fid_smoke.log 2>&1 || { echo "FIDELITY SMOKE FAILED"; tail -20 $W/a5_fid_smoke.log; exit 1; }
  echo "fidelity smoke ok"
fi
if [ ! -f $WEI ] || ! grep -q DET_DONE $W/a5_det_train.log 2>/dev/null; then
  $PY $LOCK $REPO -- $PY $X/a5_det.py train /home/uservutl/data/visdrone_yolo $W/a5_det > $W/a5_det_train.log 2>&1 \
    && echo DET_DONE >> $W/a5_det_train.log || { echo "DET TRAIN FAILED"; tail -20 $W/a5_det_train.log; exit 1; }
  mkdir -p $A5/detector
  cp $W/a5_det/yolo11s_visdrone_960/results.csv $W/a5_det/yolo11s_visdrone_960/args.yaml $A5/detector/
  cp /home/uservutl/data/visdrone_yolo/split.txt $A5/detector/
  sha256sum $WEI > $A5/detector/best_pt.sha256
  echo "detector done $(date)"
fi
if ! grep -q PROFILE_A5_DONE $W/a5_profile_testdev.log 2>/dev/null; then
  bash $REPO/scripts/uav_semcom/profile_a5.sh testdev $TD/images 0 > $W/a5_profile_testdev.log 2>&1 \
    || { echo "TESTDEV PROFILE FAILED"; exit 1; }
  echo "testdev profile done $(date)"
fi
if [ ! -f $A5/fidelity/coco_val.json ]; then
  $PY $LOCK $REPO -- $PY $X/a5_fidelity.py $A5/fidelity --pool val --image-dir $VAL/images --ann-dir $VAL/annotations \
    --weights $WEI --gen 100 > $W/a5_fid_val.log 2>&1 || { echo "FIDELITY VAL FAILED"; exit 1; }
  echo "fidelity val done $(date)"
fi
if [ ! -f $A5/fidelity/coco_testdev.json ]; then
  $PY $LOCK $REPO -- $PY $X/a5_fidelity.py $A5/fidelity --pool testdev --image-dir $TD/images --ann-dir $TD/annotations \
    --weights $WEI > $W/a5_fid_testdev.log 2>&1 || { echo "FIDELITY TESTDEV FAILED"; exit 1; }
  echo "fidelity testdev done $(date)"
fi
echo A5_GPU_DONE
