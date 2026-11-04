#!/usr/bin/env bash
# Create the WSL env for the wireless A/B/C pilot (does not touch envs uav3d / sb). Usage: bash setup_env.sh
set -euxo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd "$HOME/envs"
[ -d wireless-abc ] || uv venv --python 3.12 wireless-abc
source wireless-abc/bin/activate
uv pip install "torch==2.12.*" --index-url https://download.pytorch.org/whl/cu130
uv pip install "numpy==2.5.4" "sionna-rt==2.2.0" "sionna==2.2.0" "gymnasium==1.4.0" "stable-baselines3==2.9.2" scipy matplotlib pytest
python - <<'PY'
import torch, numpy, gymnasium, stable_baselines3, importlib.metadata as md
import sionna.rt
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", numpy.__version__,
      "sionna-rt", md.version("sionna-rt"), "sionna", md.version("sionna"), "gymnasium", gymnasium.__version__,
      "sb3", stable_baselines3.__version__)
PY
uv pip freeze > "$HOME/envs/wireless-abc.lock.txt"
du -sh "$HOME/envs/wireless-abc"
