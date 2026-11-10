#!/bin/bash
# A6.0-L1 method audit (reports/uav-semcom-a6-l1-audit.md section 0), CPU only, no PPO training. Resumable.
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
O=$REPO/runs/uav_semcom/a6l1/audit
L=/home/uservutl/work/a6l1/audit_logs
mkdir -p $O $L
cd $REPO/swarmbridge/experiments/uav_semcom
step() { name=$1; done_file=$2; shift 2
  if [ ! -f "$done_file" ]; then
    $PY a6l1_audit.py "$@" > $L/$name.log 2>&1 || { echo "FAILED $name"; tail -20 $L/$name.log; exit 1; }
    echo "done $name $(date)"
  fi; }
step tune $O/tune/phy_mcs_blind_xq4_noqmin.json tune
step cross_test $O/cross_test.json cross --pool test
step cross_testdev $O/cross_testdev.json cross --pool testdev
step val $O/val.json val
step final_test $O/final_test.json final --pool test
step final_testdev $O/final_testdev.json final --pool testdev
echo AUDIT_DONE
