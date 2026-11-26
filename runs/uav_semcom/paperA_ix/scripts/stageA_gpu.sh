#!/bin/bash
# stage A, GPU part (run under fd_gpulock_win.py): profile U_train + U_val only
source "$(dirname "$0")/common.sh"
stamp "stage A profile start" >> "$LOG/stageA.log"
$PY ix_profile.py run A >> "$LOG/stageA_profile.log" 2>&1
stamp "stage A profile done" >> "$LOG/stageA.log"
