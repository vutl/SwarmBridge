#!/usr/bin/env bash
# Reproduce the WSL env for the wireless A/B/C pilot EXACTLY from the lock file (does not touch envs uav3d / sb).
# Usage: bash setup_env.sh [ENV_DIR]   (default ~/envs/wireless-abc)
# Locked: Python 3.12.16 (uv-managed), packages in wireless-abc.lock.txt; torch wheels from the PyTorch cu130 index,
# everything else from PyPI. First created 2026-11-05 with:
#   uv pip install torch==2.12.* --index-url https://download.pytorch.org/whl/cu130
#   uv pip install numpy==2.5.4 sionna-rt==2.2.0 sionna==2.2.0 gymnasium==1.4.0 stable-baselines3==2.9.2 scipy matplotlib pytest
set -euxo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV="${1:-$HOME/envs/wireless-abc}"
[ -d "$ENV" ] || uv venv --python 3.12.16 "$ENV"
source "$ENV/bin/activate"
uv pip sync "$HERE/wireless-abc.lock.txt" \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
python - <<'PY'
import sys, torch, numpy, gymnasium, stable_baselines3, importlib.metadata as md
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", numpy.__version__,
      "sionna-rt", md.version("sionna-rt"), "sionna", md.version("sionna"), "gymnasium", gymnasium.__version__,
      "sb3", stable_baselines3.__version__)
PY
