param()

$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $workspacePath 'runs/sb_audit_ve_v1'
$pythonPath = Join-Path $workspacePath '.venv/Scripts/python.exe'
$statusPath = Join-Path $outputPath 'status.json'
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

if (Test-Path -LiteralPath $statusPath) {
    $auditStatus = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($auditStatus.state -eq 'completed') {
        Write-Output "Audit already completed: $outputPath"
        exit 0
    }
    $existingProcess = Get-Process -Id $auditStatus.pid -ErrorAction SilentlyContinue
    if ($existingProcess -and $existingProcess.ProcessName -match '^python') {
        Write-Output "Audit process already exists (PID $($auditStatus.pid)): $statusPath"
        exit 0
    }
}

$logStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutPath = Join-Path $outputPath "stdout-$logStamp.log"
$stderrPath = Join-Path $outputPath "stderr-$logStamp.log"
$auditProcess = Start-Process -FilePath $pythonPath `
    -ArgumentList @('-u', '-m', 'swarmbridge.experiments.sb_audit.run', '--suite', 'all') `
    -WorkingDirectory $workspacePath -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru

[ordered]@{
    pid = $auditProcess.Id
    launched_at = (Get-Date).ToString('o')
    stdout = $stdoutPath
    stderr = $stderrPath
    command = '.venv\Scripts\python.exe -u -m swarmbridge.experiments.sb_audit.run --suite all'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'launcher.json') -Encoding utf8

Write-Output "Started/resumed SB audit: PID $($auditProcess.Id)"
Write-Output "Status: $statusPath"
Write-Output "Log: $stdoutPath"
