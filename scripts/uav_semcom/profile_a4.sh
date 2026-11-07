#!/bin/bash
# A4 step 1: re-profile the mode table on NATIVE-resolution VisDrone-DET-val frames (same 100-image subset as A1:
# sorted by name, every 5th). Real bitstreams, PSNR/LPIPS, encode/decode latency (GPU; learned codecs also on CPU).
# Holds the shared GPU lock; run with the CPU otherwise idle (latency timing).
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
IMG=$REPO/data/visdrone/VisDrone2019-DET-val/images
W=/home/uservutl/work
LOCK=$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py
cd $REPO/swarmbridge/experiments/uav_semcom
MODES=avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25
$PY $LOCK $REPO -- $PY profile_a11.py $IMG $W/a4_native_a11 --size 0 --n 100 --cpu-learned 10 --modes $MODES || exit 1
$PY $LOCK $REPO -- $PY profile_a12.py $IMG $W/a4_native_a12 --size 0 --n 100 --thumbs 64,128 || exit 1
mkdir -p $REPO/runs/uav_semcom/a4_native/a11 $REPO/runs/uav_semcom/a4_native/a12
cp -r $W/a4_native_a11/{profile.jsonl,cpu_learned.jsonl,meta.json} $REPO/runs/uav_semcom/a4_native/a11/
cp -r $W/a4_native_a12/{profile.jsonl,meta.json} $REPO/runs/uav_semcom/a4_native/a12/
echo PROFILE_DONE
