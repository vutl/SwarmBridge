#!/bin/bash
# R1b-0a queue (protocol reports/swarmbridge-r1b0a-protocol.md): precompute G7 + AUX with the locked FRAME config
# (detections only, nothing scored), then tune every condition x R_ul level on TUNE (CPU). Resumable precompute.
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
W=/home/uservutl/work/r1b
E=$REPO/swarmbridge/experiments/uav_semcom
R=$REPO/runs/uav_semcom/r1b
Q=$($PY -c "import json;print(json.load(open('$R/lock_v1.json'))['frame_q'])")
mkdir -p $W && cd $W
for S in gate aux; do
  $PY $REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py $REPO -- $PY -I $E/r1b_precompute.py $R/split_v1.json \
    /home/uservutl/data/visdrone_vid $W/precomp --set $S --q $Q \
    --onboard /home/uservutl/work/r1a/yolo11n_visdrone_640/weights/best.pt \
    --server /home/uservutl/work/a5_det/yolo11s_visdrone_960/weights/best.pt > $W/precomp_$S.log 2>&1 || { echo "PRECOMP $S FAILED"; exit 1; }
done
cd $E
CUDA_VISIBLE_DEVICES= $PY r1b_run.py tune $R/split_v1.json $W/precomp $R/lock_v1.json $R/tune_v1.json --procs 6 > $W/tune.log 2>&1 \
  || { echo "TUNE FAILED"; tail -20 $W/tune.log; exit 1; }
echo R1B0A_TUNE_DONE
