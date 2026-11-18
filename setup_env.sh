#!/bin/bash
# Python environment for the student and the robot client, in .venv (Python 3.12). Needs internet once.
# usage: ./setup_env.sh        if PyPI is slow: UV_DEFAULT_INDEX=<mirror> (uv) or PIP_INDEX_URL=<mirror> (pip)
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
else
  PY=$(command -v python3.12 || true); [ -n "$PY" ] || { echo "need python3.12 (or install uv: https://docs.astral.sh/uv/)"; exit 1; }
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python -c "import torch, lerobot, zmq, msgpack, cv2; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available(), '| lerobot', lerobot.__version__)"
echo "SETUP_DONE  next: ./assemble_weights.sh  then  ./run_selftest.sh"
