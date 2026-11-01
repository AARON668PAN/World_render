"""Run the leg_check.py SAM3 rule on the windows sampled by sample_check.py (check/motion.json): 5 frames per window
(start, 1/4, middle, 3/4, end); contact sheet of the frame with the largest leg mask (or the middle frame), mask in red,
labelled LEG / ok.  usage: /data/repos/sam3/.venv/bin/python check_legs_sample.py  -> check/legs_sample_v3.jpg, check/legs_sample_v3.json"""
import json, subprocess
import cv2, numpy as np, torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

ROOT = "/data/datasets/unitree_g1_inspire"
KEY = "videos/observation.images.head_stereo_left"
index = {(e["dataset"], e["episode"]): e for e in json.load(open(f"{ROOT}/index.json"))}
motion = json.load(open(f"{ROOT}/check/motion.json"))
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
proc = Sam3Processor(build_sam3_image_model(checkpoint_path="/data/models/sam3/sam3/sam3.pt", load_from_HF=False))


def frame(e, t):
    vp = f"{ROOT}/official/{e['dataset']}/{KEY}/chunk-{e['chunk']:03d}/file-{e['file']:03d}.mp4"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", vp, "-frames:v", "1", "-vf", "scale=448:336",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(336, 448, 3)


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


tiles, res = [], {}
for key in motion:
    short, ep, s = key.split("/"); ds = "G1_WBT_Inspire_" + short; e = index[(ds, int(ep[2:]))]; s = int(s[1:])
    best, best_a, leg = None, -1, False
    for k in (0, 12, 24, 36, 48):
        img = frame(e, e["t0"] + 2 * (s + k) / 30); st = proc.set_image(Image.fromarray(img)); m = np.zeros((336, 448), bool)
        m = legs_in(proc.set_text_prompt(state=st, prompt="leg"), 336, 448)  # same rule as leg_check.py
        if m.mean() > best_a or (k == 24 and best_a <= 0):
            best, best_a = (img, m), m.mean()
        leg |= bool(m.any())
    img, m = best; vis = img.copy(); vis[m] = (0.5 * vis[m] + [127, 0, 0]).astype(np.uint8)
    vis = cv2.resize(vis, (288, 216)); cv2.rectangle(vis, (0, 0), (288, 16), (0, 0, 0), -1)
    cv2.putText(vis, f"{'LEG' if leg else 'ok '} {short[:24]} {ep}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (255, 80, 80) if leg else (120, 255, 120), 1, cv2.LINE_AA)
    tiles.append(vis); res[key] = leg
    print(key, "LEG" if leg else "ok", flush=True)
COLS = 6; rows = (len(tiles) + COLS - 1) // COLS; sheet = np.zeros((rows * 216, COLS * 288, 3), np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, COLS); sheet[r * 216:(r + 1) * 216, c * 288:(c + 1) * 288] = t
cv2.imwrite(f"{ROOT}/check/legs_sample_v3.jpg", sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
json.dump(res, open(f"{ROOT}/check/legs_sample_v3.json", "w"), indent=1)
print(f"LEGS_SAMPLE_DONE {sum(res.values())}/{len(res)} windows flagged")
