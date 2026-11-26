# sourced by the paperA_ix stage scripts (WSL)
set -euo pipefail
cd /mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom
PY=/home/uservutl/envs/semcom-a/bin/python
IXD=/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_ix
LOG=$IXD/logs
mkdir -p "$LOG"
stamp() { echo "[$(date '+%F %T')] $*"; }
