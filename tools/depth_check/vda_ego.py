"""Video Depth Anything on a monocular ego video, checked against the ego reconstruction.

Runs VDA (relative disparity, or --metric for metric depth) on every frame, then compares with the reconstruction:
box surface depth (mesh + pose + refined camera, z-buffered, only on SAM3 box-mask pixels) and MANO hand joint depths.
Relative output is aligned with one global affine fit in disparity (1/z = a*d + b) over all box pixels of the video;
metric output is compared raw and after one global scale. Writes <out>/depth.npz and <out>/rgb_depth.mp4.

usage: /data/repos/sam3/.venv/bin/python vda_ego.py <video.mp4> <recon_geom.npz from export_recon.py> <ego_camera_refined.npz> <box_masks.npz> <out dir> [--metric]
"""

import os
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

video, recon_npz, cam_npz, masks_npz, out = sys.argv[1:6]
METRIC = "--metric" in sys.argv
os.makedirs(out, exist_ok=True)

_rd = imageio.get_reader(video)
frames = np.zeros((_rd.count_frames(), *_rd.get_data(0).shape), np.uint8)
for _k, _f in enumerate(_rd):
    frames[_k] = _f
T, H, W = frames.shape[:3]
sys.path.insert(0, "/data/repos/video_depth_anything")
from video_depth_anything.dinov2_layers import attention as _att
from video_depth_anything.video_depth import VideoDepthAnything


def _sdpa(self, x, attn_bias=None):  # no xformers here
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    return self.proj_drop(self.proj(torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(B, N, C)))


_att.Attention.forward = _sdpa
_att.MemEffAttention.forward = _sdpa
ckpt = ("/data/models/video_depth_anything/metric_large/metric_video_depth_anything_vitl.pth" if METRIC
        else "/data/models/video_depth_anything/large/video_depth_anything_vitl.pth")
vda = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=METRIC)
vda.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
vda = vda.to("cuda").eval()
import time
t0 = time.time()
with torch.no_grad():
    d, _ = vda.infer_video_depth(frames, 30, input_size=518, device="cuda")
d = np.asarray(d, np.float32)
print(f"VDA {'metric' if METRIC else 'relative'}: {T} frames {W}x{H} in {time.time() - t0:.1f} s, output range {d.min():.3f}..{d.max():.3f}")
del vda
torch.cuda.empty_cache()

# Reconstruction depth: box surface z-buffer on box-mask pixels, MANO joint depths.
geom = np.load(recon_npz)
mano, obj_p, obj_q, surf = geom["mano"], geom["obj_p"], geom["obj_q"], geom["surf"]  # mano (T, 42, 3)
cams = np.load(cam_npz)
c2w, F = cams["cam_to_world"], float(cams["focal"])
m = np.load(masks_npz)
box_masks = np.unpackbits(m["masks"], axis=-1)[..., :W].astype(bool)


def to_cam(t, pts):
    w2c = np.linalg.inv(c2w[t])
    return pts @ w2c[:3, :3].T + w2c[:3, 3]


def project(c):
    return F * c[:, 0] / c[:, 2] + W / 2, F * c[:, 1] / c[:, 2] + H / 2


box_px, joint_px = [], []  # (t, v, u, z_recon)
for t in range(T):
    c = to_cam(t, R.from_quat(obj_q[t], scalar_first=True).apply(surf) + obj_p[t])
    u, v = project(c)
    ok = (c[:, 2] > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok].astype(int), v[ok].astype(int), c[ok, 2]
    zbuf = np.full((H, W), np.inf)
    np.minimum.at(zbuf, (v, u), z)
    sel = np.isfinite(zbuf) & box_masks[t]
    vv, uu = np.nonzero(sel)
    if len(vv) > 2000:
        k = np.random.default_rng(t).choice(len(vv), 2000, replace=False)
        vv, uu = vv[k], uu[k]
    box_px += [(t, a, b, zbuf[a, b]) for a, b in zip(vv, uu)]
    cj = to_cam(t, mano[t])
    u, v = project(cj)
    for k in range(len(cj)):
        if cj[k, 2] > 0.05 and 0 <= u[k] < W and 0 <= v[k] < H:
            joint_px.append((t, int(v[k]), int(u[k]), cj[k, 2]))
box_px, joint_px = np.array(box_px), np.array(joint_px)


def sample(px):
    return d[px[:, 0].astype(int), px[:, 1].astype(int), px[:, 2].astype(int)]


zb, zj = box_px[:, 3], joint_px[:, 3]
db, dj = sample(box_px), sample(joint_px)
if METRIC:
    s = np.median(zb / db)
    pred_b, pred_j, pred_all = db, dj, d
    note = f"raw (no fit); global scale that would match the box: x{s:.3f}"
    pred_b_s, pred_j_s = s * db, s * dj
else:
    A = np.stack([db, np.ones_like(db)], 1)
    coef = np.linalg.lstsq(A, 1 / zb, rcond=None)[0]
    for _ in range(3):  # trim outliers (mask edges) and refit
        r = np.abs(A @ coef - 1 / zb)
        keep = r < 3 * np.median(r)
        coef = np.linalg.lstsq(A[keep], 1 / zb[keep], rcond=None)[0]
    inv = lambda x: 1 / np.maximum(coef[0] * x + coef[1], 1e-3)
    pred_b, pred_j, pred_all = inv(db), inv(dj), inv(d)
    note = f"one global affine fit in disparity over all box pixels (a={coef[0]:.4g}, b={coef[1]:.4g})"
    pred_b_s, pred_j_s = pred_b, pred_j


def report(name, pred, ref, tt):
    e = np.abs(pred - ref) * 100
    per_frame = np.array([np.median(pred[tt == t] - ref[tt == t]) for t in np.unique(tt)]) * 100
    print(f"  {name}: |err| median {np.median(e):.1f} cm, 90% {np.percentile(e, 90):.1f} cm; per-frame bias "
          f"spread (std over frames) {per_frame.std():.1f} cm; reference depth {ref.min():.2f}-{ref.max():.2f} m")


print(note)
report("box surface", pred_b, zb, box_px[:, 0])
report("hand joints", pred_j, zj, joint_px[:, 0])
if METRIC:
    print("after that one global scale:")
    report("box surface", pred_b_s, zb, box_px[:, 0])
    report("hand joints", pred_j_s, zj, joint_px[:, 0])
# Per-frame fit on the box only, tested on the hand joints of the same frame: separates the per-frame shape
# accuracy from temporal scale drift.
tb, tj = box_px[:, 0].astype(int), joint_px[:, 0].astype(int)
fit_err_b, test_err_j, frame_fit = [], [], np.full((T, 2), np.nan)
for t in np.unique(tb):
    sb, sj = tb == t, tj == t
    if sb.sum() < 200 or sj.sum() < 5:
        continue
    if METRIC:
        k = np.median(zb[sb] / db[sb])
        fb, fj = k * db[sb], k * dj[sj]
    else:
        c = np.linalg.lstsq(np.stack([db[sb], np.ones(sb.sum())], 1), 1 / zb[sb], rcond=None)[0]
        fb, fj = 1 / np.maximum(c[0] * db[sb] + c[1], 1e-3), 1 / np.maximum(c[0] * dj[sj] + c[1], 1e-3)
        frame_fit[t] = c
    fit_err_b.append(np.median(np.abs(fb - zb[sb])))
    test_err_j.append(np.median(np.abs(fj - zj[sj])))
print(f"per-frame {'scale' if METRIC else 'affine'} fit on the box ({len(fit_err_b)} frames): box residual median "
      f"{np.median(fit_err_b)*100:.1f} cm; hand joints of the same frame (not used in the fit) median {np.median(test_err_j)*100:.1f} cm, "
      f"90% {np.percentile(test_err_j, 90)*100:.1f} cm")
np.save(f"{out}/frame_fit.npy", frame_fit)  # relative: per-frame (a, b) with 1/z = a*d + b
np.savez_compressed(f"{out}/depth.npz", depth_m=pred_all.astype(np.float16), raw=d.astype(np.float16), metric=METRIC)

lo, hi = np.percentile(pred_all, 1), np.percentile(pred_all, 99)
writer = imageio.get_writer(f"{out}/rgb_depth.mp4", fps=30, codec="libx264", quality=8, macro_block_size=8)
for t in range(T):
    z = np.clip((pred_all[t] - lo) / (hi - lo), 0, 1)
    vis = cv2.applyColorMap((255 * (1 - z)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]  # red = near
    left = cv2.resize(frames[t], (640, 360))
    right = cv2.resize(vis, (640, 360))
    cv2.putText(right, f"depth {lo:.2f}-{hi:.2f} m (red=near)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    writer.append_data(np.hstack([left, right]))
writer.close()
print("wrote", f"{out}/rgb_depth.mp4")
