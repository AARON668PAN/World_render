"""Rebuild a full train_src.py checkpoint (mode lora_merged, what infer_g1i.py loads) from the small trainable-only file
(LoRA A/B + patch_embedding + ref_conv, taken from runs/<run>/resume.pt on Isambard by cluster/fetch_steps.sh) and the local base
Wan2.1-Fun-Control weights: W = W_base + B @ A (LoRA scale alpha / r = 1, as in train_src.py). 350 MB to download
instead of 3.1 GB. Not part of the upstream repo.
usage: merge_lora_g1i.py <lora_NNNNN.pt> <out ckpt_NNNNN.pt>"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get("ROBORENDER_ROOT", "/data/repos/roborender")  # upstream RoboRender (core/)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)  # inst_model next to this script
from core.dit import WanModel  # noqa: E402
from core.loader import load_model, convert_dit  # noqa: E402
from core.pipeline import DIT_CONFIG  # noqa: E402
import inst_model as IM  # noqa: E402

src, out = sys.argv[1:3]
z = torch.load(src, map_location="cpu"); keep = z["keep"]
sd = {k: v.float() for k, v in load_model(WanModel, f"{IM.MODEL_DIR}/diffusion_pytorch_model.safetensors", DIT_CONFIG, convert_dit, torch.bfloat16, "cpu").state_dict().items()}
n = 0
for k, v in keep.items():
    if k.endswith(".A.weight"):
        base = k[: -len(".A.weight")]
        sd[f"{base}.weight"] = sd[f"{base}.weight"] + keep[f"{base}.B.weight"].float() @ v.float(); n += 1
    elif not k.endswith(".B.weight"):
        sd[k] = v.float()  # patch_embedding (65 input channels), ref_conv
torch.save(dict(mode="lora_merged", layout="g1src", in_dim=sd["patch_embedding.weight"].shape[1], state={k: v.to(torch.bfloat16) for k, v in sd.items()},
                step=z["step"]), out)
print(f"merged {n} LoRA layers, step {z['step']}, in_dim {sd['patch_embedding.weight'].shape[1]} -> {out}")
