# Overnight queue: wait for the seed-0 pilot process, then extend by priority.
# Finished jobs are skipped and interrupted ones resume, so re-running this script is safe.
param([int]$WaitPid = 0)
Set-Location (Split-Path $PSScriptRoot -Parent)
$out = "runs/sb_pilot_matched_v1"
if ($WaitPid -gt 0) { try { Wait-Process -Id $WaitPid -ErrorAction Stop } catch {} }
$stages = @(
    @("map_x", "sb_x_p50", "sb_x_p10"),   # pre-registered x_t-only pairs, seeds 1-2
    @("sb_x_p70"),                        # exploratory, added after seeing seed-0 results
    @("map_c", "sb_c_p10", "sb_c_p50")    # conditioned pairs, seeds 1-2
)
$i = 0
foreach ($models in $stages) {
    $i++
    "stage $i ($($models -join ' ')) start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$out/queue.log"
    & .venv\Scripts\python.exe -m swarmbridge.experiments.sb_pilot.run --seeds 0 1 2 --models @models > "$out/stage$i.log" 2> "$out/stage$i.err"
    "stage $i end $(Get-Date -Format s) exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 "$out/queue.log"
}
