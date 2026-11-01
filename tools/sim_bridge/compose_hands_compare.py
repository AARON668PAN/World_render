"""Side by side: the human ego video and robot renders (render_sim_depth.py outputs) overlaid on it (robot pixels:
75 % render), with a 2x zoom on the hands underneath.
usage: python compose_hands_compare.py <human video> <out.mp4> <label>=<sim dir> [<label>=<sim dir> ...]
       e.g. "G1 + Dex3 (CHORD)"=ego/tissue_box/depth_compare/v5"""
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

video, out = sys.argv[1:3]
sims = [a.split("=", 1) for a in sys.argv[3:]]
lab = {k: np.load(f"{d}/sim_depth.npz")["label"] for k, d in sims}
T, H, W = lab[sims[-1][0]].shape
rd = {k: imageio.get_reader(f"{d}/sim_color.mp4") for k, d in sims}
wr = imageio.get_writer(out, fps=30, codec="libx264", quality=8, macro_block_size=8)


def tag(img, text):
    img = img.copy(); cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# zoom window: around the mean robot-hand pixel over the clip (label 3)
ys, xs = np.nonzero((lab[sims[-1][0]] == 3).any(0))
cy, cx = int(np.median(ys)), int(np.median(xs)); zh, zw = H // 2, W // 2
y0, x0 = int(np.clip(cy - zh // 2, 0, H - zh)), int(np.clip(cx - zw // 2, 0, W - zw))
for t, fr in enumerate(imageio.get_reader(video)):
    if t >= T:
        break
    hum = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
    panels = [tag(hum, "human (original)")]
    for k, _ in sims:
        sim = rd[k].get_data(t); m = np.isin(lab[k][t], (3, 4))[..., None]
        panels.append(tag(np.where(m, (0.75 * sim + 0.25 * hum).astype(np.uint8), hum), k))
    zoom = [cv2.resize(p[y0:y0 + zh, x0:x0 + zw], (W, H), interpolation=cv2.INTER_CUBIC) for p in panels]
    wr.append_data(np.concatenate([np.concatenate(panels, 1), np.concatenate(zoom, 1)], 0))
wr.close()
print("wrote", out, T, "frames")
