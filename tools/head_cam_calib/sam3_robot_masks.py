"""SAM3 text-prompted robot masks ('robot arm' + 'robot hand') for the Unitree G1 head-camera calibration frames.

usage: /data/repos/sam3/.venv/bin/python sam3_robot_masks.py <frames.json> <out.npz>
"""

import json
import sys

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

frames_json, out = sys.argv[1], sys.argv[2]
rows = json.load(open(frames_json))
root = frames_json.rsplit("/", 2)[0]
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
proc = Sam3Processor(build_sam3_image_model(checkpoint_path="/data/models/sam3/sam3/sam3.pt", load_from_HF=False))
masks = []
for row in rows:
    img = Image.open(f"{root}/{row['image']}").convert("RGB")
    state = proc.set_image(img)
    union = np.zeros((img.height, img.width), bool)
    for prompt in ("robot arm", "robot hand"):
        result = proc.set_text_prompt(state=state, prompt=prompt)
        if len(result["scores"]):
            union |= result["masks"].cpu().numpy().reshape(-1, img.height, img.width).any(axis=0)
    masks.append(union)
    print(row["image"], f"robot pixels {union.mean()*100:.1f}%", flush=True)
np.savez_compressed(out, masks=np.array(masks), images=np.array([r["image"] for r in rows]))
