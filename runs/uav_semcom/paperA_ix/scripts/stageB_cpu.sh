#!/bin/bash
# stage B, CPU part: $1 = uavdt (6 UAVDT blocks, 8,800 episodes) | x1 (2 X1 blocks, 1,400 episodes); then the analysis
source "$(dirname "$0")/common.sh"
L="$LOG/stageB_$1.log"
stamp "stage B eval $1 start" >> "$L"
$PY ix_eval.py "$1" --procs 6 >> "$L" 2>&1
$PY ix_analysis.py >> "$LOG/analysis.log" 2>&1
stamp "STAGE_B_${1}_DONE" >> "$L"
