#!/bin/bash
# GroundingDINO SwinT-OGC checkpoint (GitHub release) and AnyCalib + DINOv2 (GitHub release / dl.fbaipublicfiles via torch.hub), through the proxy
source /data/repos/chord/local/recon_env.sh
export HTTPS_PROXY=http://127.0.0.1:10900 HTTP_PROXY=http://127.0.0.1:10900 https_proxy=http://127.0.0.1:10900 http_proxy=http://127.0.0.1:10900
python -m v2d.grounding_dino.lib.download_weights --output_dir /data/models/grounding_dino/swint_ogc; echo GDINO_EXIT=$?
CUDA_VISIBLE_DEVICES="" python -m v2d.anycalib.lib.download_weights --output_dir /data/models/anycalib --model_id anycalib_gen; echo ANYCALIB_EXIT=$?
