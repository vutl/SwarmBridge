# paperA_ix launcher (Windows). Usage: run_stage.ps1 A | uavdt | x1
# A     : stage A (GPU profile of U_train/U_val under the GPU lock, then CPU timing / fitting / H* tuning / 9 RL runs /
#         checkpoint selection). Stops before the L2 lock.
# uavdt : stage B UAVDT (requires the committed, verified L2 lock): U_test profile + fidelity under the GPU lock, then
#         the 6 UAVDT blocks and the analysis
# x1    : stage B X1 (requires the L2 lock; independent of uavdt): X1 profile under the GPU lock, then the 2 X1 blocks
#         and the analysis
param([Parameter(Mandatory = $true)][ValidateSet("A", "uavdt", "x1")][string]$Stage)
$R = "C:\Users\USER\Documents\SwarmBridge"
$S = "/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_ix/scripts"
$L = "$R\runs\uav_semcom\paperA_ix\logs"
New-Item -ItemType Directory -Force $L | Out-Null
# fd_gpulock_win runs one WSL script while holding the GPU lock: stageA_gpu.sh or the stage-B wrapper per dataset
$gpu = if ($Stage -eq "A") { "$S/stageA_gpu.sh" } else { "$S/stageB_gpu_$Stage.sh" }
& "$R\.venv\Scripts\python.exe" "$R\swarmbridge\experiments\uav_semcom\fd_gpulock_win.py" $R $gpu *> "$L\run_$Stage.gpu.log"
if ($LASTEXITCODE -ne 0) {
    "GPU part failed or GPU lock busy (exit $LASTEXITCODE)" | Out-File -Append "$L\run_$Stage.gpu.log"
    exit 1
}
if ($Stage -eq "A") {
    wsl.exe -d Ubuntu -u uservutl -- bash "$S/stageA_cpu.sh" *> "$L\run_A.cpu.log"
} else {
    wsl.exe -d Ubuntu -u uservutl -- bash "$S/stageB_cpu.sh" $Stage *> "$L\run_$Stage.cpu.log"
}
