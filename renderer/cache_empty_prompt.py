"""Encode the empty prompt once with Wan's T5 (umt5-xxl) and save it as PROMPT_CACHE = {"": embedding}. train_src.py,
infer_g1i.py, cache_g1i.py and diag_gap.py load only the VAE and CLIP (inst_model.lite_pipe) and read the prompt from this
file, because T5 is too large to keep next to the DiT. The renderer always uses the empty prompt.
Trimmed from local/cache_prompts.py of the working copy (which also encoded four instruction files of the older
instance-channel work). Not part of the upstream repo.
usage: ROBORENDER_ROOT=<RoboRender clone> WAN_MODEL_DIR=<Wan2.1-Fun-V1.1-1.3B-Control> python cache_empty_prompt.py [out.pt]
(default out = $PROMPT_CACHE, else /data/models/roborender/prompt_cache.pt; needs models_t5_umt5-xxl-enc-bf16.pth and
google/umt5-xxl/ in the model dir)"""
import os
import sys

import torch

sys.path.insert(0, os.environ.get("ROBORENDER_ROOT", "/data/repos/roborender"))
from core.pipeline import RoboRenderPipeline  # noqa: E402

MODEL_DIR = os.environ.get("WAN_MODEL_DIR", "/data/models/roborender/Wan2.1-Fun-V1.1-1.3B-Control")
out = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PROMPT_CACHE", "/data/models/roborender/prompt_cache.pt")
pipe = RoboRenderPipeline(MODEL_DIR, None)
with torch.no_grad():
    cache = {"": pipe.encode_prompt("").detach().cpu()}
torch.save(cache, out)
print("cached", list(cache), "->", out)
