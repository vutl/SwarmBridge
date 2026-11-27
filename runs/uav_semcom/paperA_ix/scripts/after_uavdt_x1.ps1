# Waits for the stage-B UAVDT launcher (given PID) to exit, then runs stage B X1. Sequential on purpose: the MPC arms
# have a wall-clock budget, so X1 must not compete with the UAVDT evaluation for CPU.
param([Parameter(Mandatory = $true)][int]$WaitPid)
$R = "C:\Users\USER\Documents\SwarmBridge"
$L = "$R\runs\uav_semcom\paperA_ix\logs\after_uavdt_x1.log"
"$(Get-Date -Format s) waiting for PID $WaitPid" | Out-File -Append $L
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
"$(Get-Date -Format s) PID $WaitPid exited; launching stage x1" | Out-File -Append $L
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$R\runs\uav_semcom\paperA_ix\scripts\run_stage.ps1" -Stage x1
"$(Get-Date -Format s) run_stage x1 exit $LASTEXITCODE" | Out-File -Append $L
