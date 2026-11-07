#!/bin/bash
# A5.1: profile the A4 mode set on ALL images of an A5 pool (manifest split list), native resolution, same code,
# codecs, metrics and timing protocol as A4 (profile_a11 / profile_a12; learned codecs also timed on CPU).
# Holds the shared GPU lock; run with the CPU otherwise idle (latency timing).
# Usage: profile_a5.sh POOL(val|testdev) IMAGE_DIR [N_CPU_LEARNED]
set -e
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
W=/home/uservutl/work
LOCK=$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py
A5=$REPO/runs/uav_semcom/a5
POOL=$1; IMG=$2; NCPU=${3:-30}
cd $REPO/swarmbridge/experiments/uav_semcom
NAMES=$W/a5_names_$POOL.txt
if [ "$POOL" = "val" ]; then SEL='e["source"]=="VisDrone2019-DET-val"'; else SEL='e["split"]=="testdev"'; fi
$PY -c "import json; print('\n'.join(e['name'] for e in json.load(open('$A5/manifest.json'))['images'] if $SEL))" > $NAMES
echo "$POOL: $(wc -l < $NAMES) images"
MODES=avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25
$PY $LOCK $REPO -- $PY profile_a11.py $IMG $W/a5_a11_$POOL --size 0 --names $NAMES --cpu-learned $NCPU --modes $MODES
$PY $LOCK $REPO -- $PY profile_a12.py $IMG $W/a5_a12_$POOL --size 0 --names $NAMES --thumbs 64,128
mkdir -p $A5/profile/a11_$POOL $A5/profile/a12_$POOL
cp $W/a5_a11_$POOL/{profile.jsonl,meta.json} $A5/profile/a11_$POOL/
if [ -f $W/a5_a11_$POOL/cpu_learned.jsonl ]; then cp $W/a5_a11_$POOL/cpu_learned.jsonl $A5/profile/a11_$POOL/; fi
cp $W/a5_a12_$POOL/{profile.jsonl,meta.json} $A5/profile/a12_$POOL/
echo PROFILE_A5_DONE $POOL
