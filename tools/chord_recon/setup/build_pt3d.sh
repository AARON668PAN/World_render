#!/bin/bash
source /data/repos/chord/local/recon_env.sh
cd /data/repos/chord/local/third_party
export FORCE_CUDA=1 MAX_JOBS=16
echo "start $(date)"; nvcc --version | tail -1
pip install -v --no-deps --no-build-isolation ./pytorch3d
echo "EXIT=$? $(date)"
