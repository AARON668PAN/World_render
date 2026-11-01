"""SAM3 check of the extracted segments for the robot's own legs (the joint-state classifier is ~88 % precise).
Every STEP-th frame of a segment is checked with the prompt "leg" (rule in legs_in: the old bottom-edge rule, plus thick
masks reaching the bottom fifth near the middle, which catches legs seen from above, e.g. squatting at the washing
machine, that end at 0.86-0.95 H; people's legs, chair legs, table edges and floor posts are ignored).
Per frame: leg if any checked frame within +-STEP shows legs. A segment is rejected if its 49-frame window has legs;
the per-frame flags ("free") go to cache_g1i.py, which keeps source windows off leg frames.
usage: /data/repos/sam3/.venv/bin/python leg_check.py   -> leg_check.json {id: {"leg": bool, "free": [0/1 per frame]}}"""
import json, os
import cv2, imageio.v2 as imageio, numpy as np, torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

ROOT = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire")  # dataset work dir
STEP, WIN = 6, 49
segs = json.load(open(f"{ROOT}/segments.json"))
out_p = f"{ROOT}/leg_check.json"; out = json.load(open(out_p)) if os.path.exists(out_p) else {}
out = {k: v for k, v in out.items() if "free" in v}  # results of the old rule are redone
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
proc = Sam3Processor(build_sam3_image_model(checkpoint_path=os.environ.get("SAM3_CKPT", "/data/models/sam3/sam3/sam3.pt"), load_from_HF=False))


def legs_in(r, H, W):
    """robot's own legs: a "leg" mask touching the bottom 6 rows (score >= 0.4, > 0.3 % of the image), or a thick one
    (survives a 9x9 erosion; table edges and floor posts do not) with score >= 0.5, > 0.5 % of the image, reaching
    y >= 0.8 H with its centroid in the middle 60 % horizontally."""
    if not len(r["scores"]):
        return np.zeros((H, W), bool)
    m = np.zeros((H, W), bool)
    for sc, mk in zip(r["scores"].float().cpu().numpy(), r["masks"].cpu().numpy().reshape(-1, H, W) > 0):
        if sc >= 0.4 and mk[-6:].any() and mk.mean() > 0.003:
            m |= mk; continue
        if sc >= 0.5 and mk.mean() > 0.005:
            ys, xs = np.nonzero(mk)
            if ys.max() >= 0.8 * H and 0.2 * W <= xs.mean() <= 0.8 * W and cv2.erode(mk.astype(np.uint8), np.ones((9, 9), np.uint8)).any():
                m |= mk
    return m


def own_legs(img):
    H, W = img.shape[:2]
    return legs_in(proc.set_text_prompt(state=proc.set_image(Image.fromarray(img)), prompt="leg"), H, W).any()


for i, s in enumerate(segs):
    vp = f"{ROOT}/segments/{s['id']}.mp4"
    if s["id"] in out or not os.path.exists(vp):
        continue
    n = len(s["raw"]); w0 = s["win_start"]
    ks = sorted(set(range(0, n, STEP)) | {n - 1})
    rd = imageio.get_reader(vp); hit = {k: own_legs(rd.get_data(k)) for k in ks}; rd.close()
    leg = np.zeros(n, bool)
    for k, h in hit.items():
        if h:
            leg[max(0, k - STEP):k + STEP + 1] = True
    out[s["id"]] = dict(leg=bool(leg[w0:w0 + WIN].any()), free=(~leg).astype(int).tolist(), checked=[k for k in ks], hits=[k for k in ks if hit[k]])
    if i % 50 == 0:
        json.dump(out, open(out_p, "w")); print(f"{i + 1}/{len(segs)}: rejected so far {sum(v['leg'] for v in out.values())}/{len(out)}", flush=True)
json.dump(out, open(out_p, "w"))
print(f"LEG_CHECK_DONE: {sum(v['leg'] for v in out.values())} of {len(out)} segments show legs in the window")
