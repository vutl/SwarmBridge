#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/envs/wireless-abc"
ENV="$HOME/envs/active-recovery"
if [[ ! -f "$ENV/pyvenv.cfg" ]]; then
  "$BASE/bin/python" -m venv "$ENV"
  printf '%s\n' "$BASE/lib/python3.12/site-packages" > "$ENV/lib/python3.12/site-packages/wireless_base.pth"
fi
"$ENV/bin/python" -m pip install --no-deps /mnt/c/Users/USER/Documents/SwarmBridge/pdf/GithubModels/ultralytics__ultralytics \
  opencv-python-headless polars requests pycocotools ultralytics-thop urllib3 certifi charset-normalizer idna
"$ENV/bin/python" -m pip freeze > /mnt/c/Users/USER/Documents/SwarmBridge/runs/active_recovery/environment.lock.txt
export YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR="$ENV/yolo-config"
"$ENV/bin/python" -c 'from ultralytics import YOLO; import torch; print("READY", torch.__version__,torch.cuda.is_available())'
