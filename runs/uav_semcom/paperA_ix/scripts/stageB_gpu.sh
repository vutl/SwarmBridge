#!/bin/bash
# stage B, GPU part (run under fd_gpulock_win.py): $1 = uavdt (U_test profile + frozen-YOLO fidelity) | x1 (X1 profile)
source "$(dirname "$0")/common.sh"
stamp "stage B profile $1 start" >> "$LOG/stageB_$1.log"
$PY ix_profile.py run "$1" >> "$LOG/stageB_$1_profile.log" 2>&1
stamp "stage B profile $1 done" >> "$LOG/stageB_$1.log"
