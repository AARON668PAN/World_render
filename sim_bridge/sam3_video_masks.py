"""SAM3 text-prompted masks for every frame of a video (image mode, one prompt).

usage: /data/repos/sam3/.venv/bin/python sam3_video_masks.py <video.mp4> "<prompt>" <out.npz>
"""

import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

video, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
proc = Sam3Processor(build_sam3_image_model(checkpoint_path=os.environ.get("SAM3_CKPT", "/data/models/sam3/sam3/sam3.pt"), load_from_HF=False))
masks, scores = [], []
for frame in imageio.get_reader(video):
    state = proc.set_image(Image.fromarray(frame))
    result = proc.set_text_prompt(state=state, prompt=prompt)
    h, w = frame.shape[:2]
    if len(result["scores"]):
        k = int(result["scores"].float().argmax())
        masks.append(result["masks"].cpu().numpy().reshape(-1, h, w)[k] > 0)
        scores.append(float(result["scores"][k]))
    else:
        masks.append(np.zeros((h, w), bool))
        scores.append(0.0)
np.savez_compressed(out, masks=np.packbits(np.array(masks), axis=-1), shape=np.array([len(masks), h, w]), scores=np.array(scores))
print(f"{len(masks)} frames, detected in {sum(s > 0 for s in scores)}, mean score {np.mean(scores):.2f}")
