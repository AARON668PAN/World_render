#!/bin/bash
# SAM2.1 large (official .pt for the sam2 repo), MoGe-2 ViT-L normal, bert-base-uncased (GroundingDINO text encoder) via hf-mirror
source /data/repos/chord/local/recon_env.sh
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy; export NO_PROXY='*' no_proxy='*'
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
for f in ["sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml"]:
    print(hf_hub_download("facebook/sam2.1-hiera-large", f, local_dir="/data/models/sam2/sam2.1-hiera-large"), flush=True)
print(snapshot_download("Ruicheng/moge-2-vitl-normal", local_dir="/data/models/moge/moge-2-vitl-normal"), flush=True)
print(snapshot_download("bert-base-uncased", cache_dir="/data/models/grounding_dino/hf_home/hub",
      allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"]), flush=True)
PY
echo EXIT=$?
