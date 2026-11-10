#!/bin/bash
# VisDrone2019-VID train + val from the official README Google Drive links (test-dev NOT downloaded).
# Accepts zips already placed in $D or in the Windows Downloads folder (browser download). Retries every 30 min
# (Drive download quota), up to 16 attempts; then sha256 + unzip + integrity audit.
D=/home/uservutl/data/visdrone_vid
WD=/mnt/c/Users/USER/Downloads
WD2=/mnt/c/Users/USER/Documents/SwarmBridge/data/visdrone   # browser download placed in the repo data dir (git-ignored)
mkdir -p $D && cd $D
LOG=$D/download.log
dl() {  # id name pattern
  [ -f $2.ok ] && return 0
  if [ ! -f $2 ]; then
    for f in $WD/$3 $WD2/$3; do [ -f "$f" ] && { echo "found browser download $f" >> $LOG; cp "$f" $2; break; }; done
  fi
  if [ -f $2 ] && unzip -tq $2 >> $LOG 2>&1; then touch $2.ok; echo "ok (existing file) $2 $(date)" >> $LOG; return 0; fi
  echo "try gdown $2 $(date)" >> $LOG
  rm -f $2.part
  /home/uservutl/.local/bin/uvx gdown "https://drive.google.com/uc?id=$1" -O $2.part >> $LOG 2>&1 || { echo "GDOWN FAILED $2" >> $LOG; rm -f $2.part; return 1; }
  unzip -tq $2.part >> $LOG 2>&1 || { echo "ZIP TEST FAILED $2" >> $LOG; rm -f $2.part; return 1; }
  mv $2.part $2; touch $2.ok; echo "ok $2 $(date)" >> $LOG
}
for a in $(seq 1 16); do
  dl 1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B VisDrone2019-VID-val.zip "*VID*val*.zip"
  dl 1NSNapZQHar22OYzQYuXCugA3QlMndzvw VisDrone2019-VID-train.zip "*VID*train*.zip"
  if [ -f VisDrone2019-VID-val.zip.ok ] && [ -f VisDrone2019-VID-train.zip.ok ]; then break; fi
  echo "attempt $a incomplete, sleep 30 min $(date)" >> $LOG; sleep 1800
done
if ! { [ -f VisDrone2019-VID-val.zip.ok ] && [ -f VisDrone2019-VID-train.zip.ok ]; }; then echo VID_DL_FAILED >> $LOG; exit 1; fi
sha256sum VisDrone2019-VID-val.zip VisDrone2019-VID-train.zip > sha256.txt
for z in VisDrone2019-VID-val.zip VisDrone2019-VID-train.zip; do unzip -qn $z -d $D; done
echo VID_DL_DONE >> $LOG
cd /mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom
/home/uservutl/envs/semcom-a/bin/python -I vid_audit.py $D /home/uservutl/work/vid_audit /home/uservutl/work/vid_audit/side \
  --det det_train=/home/uservutl/data/visdrone_dl/VisDrone2019-DET-train/images \
  --det det_val=/mnt/c/Users/USER/Documents/SwarmBridge/data/visdrone/VisDrone2019-DET-val/images \
  --det det_testdev=/home/uservutl/data/visdrone_dl/VisDrone2019-DET-test-dev/images --procs 6 > /home/uservutl/work/vid_audit.log 2>&1
echo AUDIT_EXIT $? >> $LOG
