"""Human ego video vs the camera-consistent G1 + Dex3 result, image and depth side by side, with alignment numbers.

Panels (640x360 each, 3 x 2):
  original + fingertips (green: human MANO, red: robot)  |  G1 sim render, same camera   |  50% overlay + fingertips
  video depth (VDA, metric fit, x scene scale)            |  sim depth (robot, box, table) |  composite depth for the renderer
All depths are in the scaled sim world (real x scale) and share one colour map. The composite takes the sim depth
where the sim has geometry (table, box, robot) and the video depth elsewhere; human pixels the sim does not cover
are left empty (black) because they are the hole the renderer fills.

Numbers (printed, and per frame on the overlay):
  fingertip image error  robot thumb/index/middle tips vs MANO tips, px at the video resolution;
  hand depth gap         on pixels where the robot hand covers the human hand in the image: sim depth minus video
                         depth, reported in real-world cm (+ = robot farther), for video depth fits on box+table,
                         box only, table only;
  occlusion order        where the human hand is in front of the box in the video (hand pixels over the box), how
                         often the robot is in front of the box in the sim, and where the box hides the human
                         fingers (box pixels over the MANO hand silhouette), how often the box is in front of the robot.

usage: /data/repos/sam3/.venv/bin/python compare_depth.py <video> <sim dir> <recon_geom.npz> <ego_camera_refined.npz>
       <box_masks.npz> <human_table_masks.npz> <vda_rel dir> <depth_align root> <out.mp4>
"""

import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

video, sim_dir, recon_npz, cam_npz, box_npz, ht_npz, vda_dir, fit_root, out = sys.argv[1:10]
sim = np.load(f"{sim_dir}/sim_depth.npz")
d_sim, d_rob, d_box, lab = (sim[k].astype(np.float32) if k != "label" else sim[k] for k in ("depth", "depth_robot", "depth_box", "label"))
tips_uv, f_sim, scale = sim["tips_uv"], float(sim["focal"]), float(sim["scale"])
T, h, w = d_sim.shape
disp = np.load(f"{vda_dir}/depth.npz")["raw"]
Hf, Wf = disp.shape[1:]
s = Wf // w  # video / panel resolution
disp = disp[:, ::s, ::s].astype(np.float32)


def unpack(z, key):
    return np.unpackbits(z[key], axis=-1)[..., :Wf].astype(bool)[:, ::s, ::s]


box_vis = unpack(np.load(box_npz), "masks")
ht = np.load(ht_npz)
hand_vis, arm_vis = unpack(ht, "hand"), unpack(ht, "forearm")
human_vis = hand_vis | arm_vis


def load_fit(name):  # per-frame (a, b), gaps filled by linear interpolation in time
    fit = np.load(f"{fit_root}_{name}/video_depth_fit.npy") if name != "both" else np.load(f"{fit_root}/video_depth_fit.npy")
    ok = np.isfinite(fit[:, 0])
    return np.stack([np.interp(np.arange(T), np.nonzero(ok)[0], fit[ok, i]) for i in range(2)], 1)


fits = {n: load_fit(n) for n in ("both", "box", "table")}
vdepth = {n: scale / np.maximum(f[:, 0, None, None] * disp + f[:, 1, None, None], 1e-3) for n, f in fits.items()}  # sim units

# MANO: projected joints and an approximate amodal hand silhouette (bone capsules + palm polygon)
geom = np.load(recon_npz)
mano = geom["mano"]
c2w = np.load(cam_npz)["cam_to_world"]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
mano_uv = np.zeros((T, 42, 2))
hand_amodal = np.zeros((T, h, w), bool)
for t in range(T):
    w2c = np.linalg.inv(c2w[t])
    c = mano[t] @ w2c[:3, :3].T + w2c[:3, 3]
    mano_uv[t] = np.stack([f_sim * c[:, 0] / c[:, 2] + w / 2, f_sim * c[:, 1] / c[:, 2] + h / 2], 1)
    canvas = np.zeros((h, w), np.uint8)
    for o in (0, 21):
        uv = mano_uv[t, o:o + 21]
        cv2.fillConvexPoly(canvas, cv2.convexHull(uv[[0, 1, 5, 9, 13, 17]].astype(np.int32)), 1)
        for a, b in BONES:
            thick = max(2, int(round(2 * 0.009 * f_sim / c[o + a, 2])))
            cv2.line(canvas, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), 1, thick)
    hand_amodal[t] = canvas > 0

# 1. fingertip image error (video px)
TIP_MAP = [(0, 4), (1, 8), (2, 12), (4, 25), (5, 29), (6, 33)]  # robot tip index -> MANO joint (right 0-20, left 21-41)
tip_px = np.stack([np.linalg.norm(tips_uv[:, r] - mano_uv[:, m], axis=1) for r, m in TIP_MAP], 1) * s
# 2. hand depth gap on overlapping pixels
robot_hand = lab == 3
gap = {n: np.full(T, np.nan) for n in vdepth}
for t in range(T):
    ov = robot_hand[t] & hand_vis[t]
    if ov.sum() >= 30:
        for n in vdepth:
            gap[n][t] = np.median(d_sim[t][ov] - vdepth[n][t][ov]) / scale * 100
# 3. occlusion order
box_am = d_box > 0
rob_am = d_rob > 0
h_front = human_vis & box_am  # video: human in front of the box
b_front = box_vis & hand_amodal & ~human_vis  # video: box hides the human hand
both = rob_am & box_am
rob_front = both & (d_rob < d_box)
hf_cov, bf_cov = h_front & both, b_front & both
print(f"{T} frames, panel {w}x{h}, scene scale {scale}")
print(f"fingertip image error (px at {Wf}x{Hf}): median {np.median(tip_px):.0f}, 90% {np.percentile(tip_px, 90):.0f}; "
      "per tip (R thumb/index/middle, L thumb/index/middle): " + " ".join(f"{np.median(tip_px[:, i]):.0f}" for i in range(6)))
for n in ("both", "box", "table"):
    g = gap[n]
    print(f"hand depth gap, video depth fit on {n:5s}: median {np.nanmedian(g):+.1f} cm, |gap| median {np.nanmedian(np.abs(g)):.1f} cm, "
          f"90% {np.nanpercentile(np.abs(g), 90):.1f} cm ({np.isfinite(g).sum()} frames with overlap)")
print(f"occlusion order: human in front of box -> robot in front in sim on {rob_front[hf_cov].mean() * 100:.1f}% of "
      f"{hf_cov.sum()} px (sim robot+box cover {hf_cov.sum() / max(h_front.sum(), 1) * 100:.0f}% of those video pixels); "
      f"box hides human -> box in front in sim on {(~rob_front[bf_cov]).mean() * 100:.1f}% of {bf_cov.sum()} px "
      f"(cover {bf_cov.sum() / max(b_front.sum(), 1) * 100:.0f}%)")
agree = (hf_cov & rob_front) | (bf_cov & ~rob_front)
cov = hf_cov | bf_cov
order_t = np.where(cov.sum((1, 2)) > 50, agree.sum((1, 2)) / np.maximum(cov.sum((1, 2)), 1) * 100, np.nan)
np.savez(out.replace(".mp4", "_metrics.npz"), order=order_t, tip_px=tip_px, **{f"gap_{n}": g for n, g in gap.items()})

# video
font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", 20)
small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", 17)
LO, HI = 0.2, 1.1


def label(img, text, sub=None):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    b = dr.textbbox((8, 5), text, font=font)
    dr.rectangle((b[0] - 5, b[1] - 3, b[2] + 5, b[3] + 3), fill=(0, 0, 0))
    dr.text((8, 5), text, font=font, fill=(255, 255, 255))
    if sub:
        b = dr.textbbox((8, h - 28), sub, font=small)
        dr.rectangle((b[0] - 5, b[1] - 3, b[2] + 5, b[3] + 3), fill=(0, 0, 0))
        dr.text((8, h - 28), sub, font=small, fill=(255, 255, 0))
    return np.asarray(im)


def cmap(z):
    v = np.clip((HI - z) / (HI - LO), 0, 1)
    img = cv2.applyColorMap((255 * v).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    img[z <= 0] = 0
    return img


def dots(img, t):
    img = img.copy()
    for r, m in TIP_MAP:
        cv2.circle(img, tuple(mano_uv[t, m].astype(int)), 5, (0, 255, 0), -1)
        cv2.circle(img, tuple(tips_uv[t, r].astype(int)), 5, (255, 0, 0), -1)
        cv2.line(img, tuple(mano_uv[t, m].astype(int)), tuple(tips_uv[t, r].astype(int)), (255, 255, 255), 1)
    return img


writer = imageio.get_writer(out, fps=30, codec="libx264", quality=8, macro_block_size=8)
sim_rgb = imageio.get_reader(f"{sim_dir}/sim_color.mp4")
for t, (frame, rgb) in enumerate(zip(imageio.get_reader(video), sim_rgb)):
    v = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    vd = vdepth["both"][t]
    comp = np.where(np.isin(lab[t], (1, 2, 3, 4)), d_sim[t], np.where(human_vis[t], 0, vd))
    sub = f"指尖偏差 {np.median(tip_px[t]):.0f} px" + (f" | 手和盒子前后关系一致 {order_t[t]:.0f}%" if np.isfinite(order_t[t]) else "")
    top = [label(dots(v, t), "人:原片 + 指尖(绿=人,红=机器人)"), label(rgb, "G1 仿真渲染(同一相机)"),
           label(dots((0.5 * v + 0.5 * rgb).astype(np.uint8), t), "叠加 50%", sub)]
    bottom = [label(cmap(vd), f"视频深度 VDA ×{scale}(仿真尺度)"), label(cmap(d_sim[t]), "仿真深度:机器人+盒子+桌子"),
              label(cmap(comp), "合成深度(给渲染器,黑=人手洞)", f"色标 {LO}–{HI} m,红=近")]
    writer.append_data(np.vstack([np.hstack(top), np.hstack(bottom)]))
writer.close()
print("wrote", out)
