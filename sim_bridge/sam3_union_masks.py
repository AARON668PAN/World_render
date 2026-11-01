"""SAM3 text-prompted masks for every frame of a video, one union mask per prompt (all instances above a score
threshold), e.g. both human hands and forearms, or the table.

usage: /data/repos/sam3/.venv/bin/python sam3_union_masks.py <video.mp4> <out.npz> "<prompt>" ["<prompt>" ...]
       out.npz: one packed (T, H, W) mask per prompt (key = prompt with spaces as _), plus shape
"""

import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

video, out, prompts = sys.argv[1], sys.argv[2], sys.argv[3:]
THRESH = 0.5
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
proc = Sam3Processor(build_sam3_image_model(checkpoint_path=os.environ.get("SAM3_CKPT", "/data/models/sam3/sam3/sam3.pt"), load_from_HF=False))
masks = {p: [] for p in prompts}
for frame in imageio.get_reader(video):
    h, w = frame.shape[:2]
    state = proc.set_image(Image.fromarray(frame))
    for p in prompts:
        result = proc.set_text_prompt(state=state, prompt=p)
        union = np.zeros((h, w), bool)
        if len(result["scores"]):
            keep = result["scores"].float().cpu().numpy() >= THRESH
            if keep.any():
                union = (result["masks"].cpu().numpy().reshape(-1, h, w)[keep] > 0).any(0)
        masks[p].append(union)
np.savez_compressed(out, shape=np.array([len(masks[prompts[0]]), h, w]),
                    **{p.replace(" ", "_"): np.packbits(np.array(m), axis=-1) for p, m in masks.items()})
for p, m in masks.items():
    m = np.array(m)
    print(f"{p}: detected in {int(m.any((1, 2)).sum())}/{len(m)} frames, mean area {m.mean() * 100:.1f}%")
