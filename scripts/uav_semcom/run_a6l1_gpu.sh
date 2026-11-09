#!/bin/bash
# A6.0-L1 GPU stage (one GPU job at a time, shared lock): task fidelity of EVERY (image, mode) pair of the A5 table,
# generative modes included (A5 measured them on the first 100 val frames only), with every detection saved for
# policy-level AP. Same detector, codecs, scoring as A5.2 (a5_fidelity.py); doubles as a reproducibility check of the
# A5.2 per-image rows. Outputs: runs/uav_semcom/a6l1/fidelity/ (per-image rows, COCO summaries; small) and
# ~/work/a6l1/dets_<pool>.npz (detections; NOT in git, sha256 recorded). Resumable (a stage is skipped if done).
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
LOCK=$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py
OUT=$REPO/runs/uav_semcom/a6l1/fidelity
W=/home/uservutl/work/a6l1
VAL=$REPO/data/visdrone/VisDrone2019-DET-val
TD=/home/uservutl/data/visdrone_dl/VisDrone2019-DET-test-dev
WEI=/home/uservutl/work/a5_det/yolo11s_visdrone_960/weights/best.pt
X=$REPO/swarmbridge/experiments/uav_semcom
mkdir -p $OUT $W
cd $W
if [ ! -f $W/smoke/coco_val.json ]; then
  $PY $LOCK $REPO -- $PY $X/a5_fidelity.py $W/smoke --pool val --image-dir $VAL/images --ann-dir $VAL/annotations \
    --weights $WEI --n 2 --gen 2 --save-dets $W/smoke/dets.npz > $W/smoke.log 2>&1 || { echo "SMOKE FAILED"; tail -20 $W/smoke.log; exit 1; }
  echo "smoke ok $(date)"
fi
if [ ! -f $OUT/coco_val.json ]; then
  $PY $LOCK $REPO -- $PY $X/a5_fidelity.py $OUT --pool val --image-dir $VAL/images --ann-dir $VAL/annotations \
    --weights $WEI --gen 548 --save-dets $W/dets_val.npz > $W/fid_val.log 2>&1 || { echo "FID VAL FAILED"; tail -20 $W/fid_val.log; exit 1; }
  echo "val done $(date)"
fi
if [ ! -f $OUT/coco_testdev.json ]; then
  $PY $LOCK $REPO -- $PY $X/a5_fidelity.py $OUT --pool testdev --image-dir $TD/images --ann-dir $TD/annotations \
    --weights $WEI --gen 403 --save-dets $W/dets_testdev.npz > $W/fid_testdev.log 2>&1 || { echo "FID TESTDEV FAILED"; tail -20 $W/fid_testdev.log; exit 1; }
  echo "testdev done $(date)"
fi
sha256sum $WEI $W/dets_val.npz $W/dets_testdev.npz > $OUT/sha256.txt
echo A6L1_GPU_DONE
