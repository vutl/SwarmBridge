#!/bin/bash
# stage A, CPU part: CPU MBT2018 timing on U_train, fit retrained state, H* tuning, 9 RL runs, checkpoint selection.
# Stops BEFORE the L2 lock (mandatory review stop). Resumable: finished steps are skipped.
source "$(dirname "$0")/common.sh"
L="$LOG/stageA.log"
stamp "stage A cpu start" >> "$L"
[ -f "$IXD/cpu/cpu_learned_stratified.jsonl" ] || $PY ix_cpu.py measure >> "$L" 2>&1
[ -f "$IXD/cpu/cpu_learned_stratified_estimators.json" ] || $PY ix_cpu.py estimate >> "$L" 2>&1
[ -f "$IXD/state_retrained.json" ] || $PY ix_stage.py fit >> "$L" 2>&1
[ -f "$IXD/tune.json" ] || $PY ix_stage.py tune --procs 6 >> "$L" 2>&1
$PY ix_stage.py train >> "$L" 2>&1
[ -f "$IXD/selection.json" ] || $PY ix_stage.py select >> "$L" 2>&1
$PY ix_stage.py status >> "$L" 2>&1
stamp "STAGE_A_DONE (stop: review, then python ix_stage.py lock and commit L2)" >> "$L"
