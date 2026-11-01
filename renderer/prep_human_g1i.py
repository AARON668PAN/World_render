"""Renderer inputs for a human ego clip on G1 + Inspire (the user's framing, 10-29): the 16:9 human video is NOT cropped but
letterboxed into the model's 4:3 448x336 (448x252 in the middle, grey bands of 42 px top and bottom); the sim robot is
rendered for the whole 448x336 with the same focal (render_sim_depth.py with CAM_WHF=448,336,<f>), so the arm also shows
in the bands. 15 fps like the training segments (every 2nd 30 fps frame).
  control = grey + the sim object (box, render label 2) in one saturated colour + the sim robot render on top (as in
            training: handled objects coloured, the real robot cut out on top)
  source  = the human video at the same time (no shift) with holes: the bands, the human hand/forearm (SAM3, dilated),
            the sim robot (dilated)
  scene   = the same letterboxed frames with only the bands and the human hand/forearm holed (v5 info frames, infer_g1i.py --info)
Not part of the upstream repo.
usage: prep_human_g1i.py <human video> <hand masks npz (sam3_union_masks)> <sim dir (sim_color.mp4 + sim_depth.npz)> <out.npz> [R,G,B object colour] [--c1]
       [--hole_px N]  (margin of the hand / robot holes in px, default 5; v6 checkpoints: --hole_px 1)"""
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

video, masks, sim, out = sys.argv[1:5]
ENC = "c1" if "--c1" in sys.argv else "flat"
HPX = int(sys.argv[sys.argv.index("--hole_px") + 1]) if "--hole_px" in sys.argv else 5  # v6: 1 (thin holes keep the fingers)  # c1: the sim object drawn with its depth (objdepth_enc.py), as in a --obj_enc c1 cache
args = [x for i, x in enumerate(sys.argv[5:], 5) if not x.startswith("--") and sys.argv[i - 1] != "--hole_px"]
OBJ_COL = np.array([int(v) for v in (args[0] if args else "235,60,200").split(",")], np.uint8)
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import objdepth_enc as OD
W, H, GRAY = 448, 336, 128
K_ROB = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HPX + 1, 2 * HPX + 1))
lab = np.load(f"{sim}/sim_depth.npz")["label"]
assert lab.shape[1:] == (H, W), lab.shape
srgb = [f for f in imageio.get_reader(f"{sim}/sim_color.mp4")]
ht = np.load(masks); vw = int(ht["shape"][2])
human = (np.unpackbits(ht["hand"], axis=-1) | np.unpackbits(ht["forearm"], axis=-1))[..., :vw]
frames = [f for f in imageio.get_reader(video)]
T = min(len(frames), len(lab), len(srgb), len(human))
vh, vw = frames[0].shape[:2]; h = int(round(W * vh / vw)); y0 = (H - h) // 2
band = np.ones((H, W), bool); band[y0:y0 + h] = False
ctrl, src, hole, scene, scene_hole = [], [], [], [], []
for t in range(0, T, 2):
    rob = np.isin(lab[t], (3, 4))
    c = np.full((H, W, 3), GRAY, np.uint8); c[lab[t] == 2] = OBJ_COL; c[rob] = srgb[t][rob]
    s = np.full((H, W, 3), GRAY, np.uint8); s[y0:y0 + h] = cv2.resize(frames[t], (W, h), interpolation=cv2.INTER_AREA)
    hm = np.zeros((H, W), np.uint8); hm[y0:y0 + h] = cv2.resize(human[t], (W, h), interpolation=cv2.INTER_NEAREST)
    hl = band | (cv2.dilate(hm, K_ROB) > 0) | (cv2.dilate(rob.astype(np.uint8), K_ROB) > 0)
    sh = band | (cv2.dilate(hm, K_ROB) > 0); sc = s.copy(); sc[sh] = GRAY; scene.append(sc); scene_hole.append(sh)
    s[hl] = GRAY
    ctrl.append(c); src.append(s); hole.append(hl)
ctrl = np.stack(ctrl)
if ENC == "c1":  # per 49-frame chunk (the model's window), inverse sim depth, ranges from that chunk
    dz = np.load(f"{sim}/sim_depth.npz")["depth"][0:T:2].astype(np.float32); inv = 1.0 / np.maximum(dz, 1e-3)
    labs = lab[0:T:2]; ob, rb = labs == 2, np.isin(labs, (3, 4))
    for c0 in range(0, len(ctrl), 49):
        sl = slice(c0, c0 + 49); seg = ctrl[sl].copy(); seg[ob[sl]] = 128
        OD.paint(seg, inv[sl], ob[sl], rb[sl]); seg[rb[sl]] = ctrl[sl][rb[sl]]; ctrl[sl] = seg
np.savez_compressed(out, ctrl=ctrl, src=np.stack(src), hole=np.stack(hole), scene=np.stack(scene), scene_hole=np.stack(scene_hole), band=band, y0=y0, h=h, enc=ENC)
print(f"wrote {out}: {len(ctrl)} frames at 15 fps, image rows {y0}..{y0 + h}, robot in {np.mean([np.isin(lab[t], (3, 4)).mean() for t in range(0, T, 2)]) * 100:.0f}% of the frame")
