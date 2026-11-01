#!/bin/bash
source /data/repos/chord/local/recon_env.sh
cd /data/repos/chord/local/third_party
export FORCE_CUDA=1 MAX_JOBS=12 SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0
echo "== nvdiffrast $(date)"; pip install -v --no-deps --no-build-isolation ./nvdiffrast; echo "NVDIFFRAST_EXIT=$?"
echo "== groundingdino $(date)"; pip install -v --no-deps --no-build-isolation ./GroundingDINO; echo "GDINO_EXIT=$?"
echo "== sam2 $(date)"; pip install -v --no-deps --no-build-isolation ./sam2; echo "SAM2_EXIT=$?"
echo "done $(date)"
