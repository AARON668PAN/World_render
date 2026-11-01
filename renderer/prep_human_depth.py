"""Control B (whole-frame depth) for a human clip at inference (v6b, 10-31), added to a prep_human_g1i.py npz as "ctrlB".
  1. VDA (relative inverse depth) of the human video, letterboxed like the npz (448x336, image rows y0..y0+h)
  2. the human hands/forearms (SAM3) and the human-video object (SAM3 box_masks) removed; a STATIC camera clip gets them
     back from the per-pixel temporal median over the frames where neither covers the pixel (--moving: spatial fill)
  3. the sim object + robot inverse depth (render_sim_depth.py, same camera) fitted onto the VDA scale on table pixels in
     front of the object that both see (robust affine), and laid on top where the sim label says object / robot is front
  4. letterbox bands: below the image the table plane (inverse depth is linear in image coordinates for a plane), above
     the top row smoothed; then fulldepth_enc.encode_window per 49-frame chunk (as training)
Not part of the upstream repo.
usage: /data/repos/sam3/.venv/bin/python prep_human_depth.py <human video> <hand masks npz> <object masks npz> <sim dir> <npz to extend> [--moving]"""
import os
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # fulldepth_enc next to this script
import fulldepth_enc as FE  # noqa: E402
sys.path.insert(0, os.environ.get("VDA_ROOT", "/data/repos/video_depth_anything"))  # Video-Depth-Anything clone
from video_depth_anything.dinov2_layers import attention as _att  # noqa: E402
from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402

video, hands, objm, sim, npz = sys.argv[1:6]; MOVING = "--moving" in sys.argv
H, W, WIN = 336, 448, 49


def _sdpa(self, x, attn_bias=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    return self.proj_drop(self.proj(torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(B, N, C)))


_att.Attention.forward = _sdpa; _att.MemEffAttention.forward = _sdpa
z = dict(np.load(npz)); y0, h = int(z["y0"]), int(z["h"]); T = len(z["ctrl"])
vid = [f for i, f in enumerate(imageio.get_reader(video)) if i % 2 == 0][:T]
vda = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=False)
vda.load_state_dict(torch.load(os.environ.get("VDA_CKPT", "/data/models/video_depth_anything/large/video_depth_anything_vitl.pth"), map_location="cpu"), strict=True)
vda = vda.to("cuda").eval()
with torch.no_grad():
    d = np.asarray(vda.infer_video_depth(np.stack(vid), 15, input_size=518, device="cuda")[0], np.float32)
del vda; torch.cuda.empty_cache()
v = np.zeros((T, H, W), np.float32)
for t in range(T):
    v[t, y0:y0 + h] = cv2.resize(d[t], (W, h), interpolation=cv2.INTER_AREA)


def unpack(path, keys):
    m = np.load(path); vw = int(m["shape"][2]); out = 0
    for k in keys:
        out = out | np.unpackbits(m[k], axis=-1)[..., :vw]
    return out


hum30 = unpack(hands, ("hand", "forearm")); ob30 = unpack(objm, ("masks",))
cover = np.zeros((T, H, W), bool); K5 = np.ones((5, 5), np.uint8)
for t in range(T):
    m = (hum30[2 * t] | ob30[2 * t]).astype(np.uint8)
    cover[t, y0:y0 + h] = cv2.dilate(cv2.resize(m, (W, h), interpolation=cv2.INTER_NEAREST), K5) > 0
inside = np.zeros((H, W), bool); inside[y0:y0 + h] = True
if MOVING:  # nothing to borrow from other frames: nearest uncovered pixel of the same frame
    comp = v.copy()
    for t in range(T):
        ok = ~cover[t] & inside
        _, (iy, ix) = distance_transform_edt(~ok, return_indices=True); comp[t][cover[t]] = v[t][iy[cover[t]], ix[cover[t]]]
    bgm = np.median(comp, 0)
else:
    bgm = np.nanmedian(np.where(cover, np.nan, v), 0); bgm = np.where(np.isnan(bgm), np.nanmedian(v, 0), bgm)
    comp = np.where(cover, bgm[None], v)
zs = np.load(f"{sim}/sim_depth.npz"); lab, dz = zs["label"][0:2 * T:2], zs["depth"][0:2 * T:2].astype(np.float32)
box, rob, table = lab == 2, np.isin(lab, (3, 4)), lab == 1
inv_sim = 1.0 / np.maximum(dz, 1e-3)
ylow = int(np.percentile(np.nonzero(box.any(2))[1], 99)) if box.any() else H // 2  # table rows in front of the object
rows_ok = np.zeros((H, W), bool); rows_ok[ylow + 5:y0 + h] = True
fit = table & ~cover & rows_ok[None]
xs, ys = inv_sim[fit][::20], comp[fit][::20]
a, b = np.polyfit(xs, ys, 1); r = np.abs(ys - (a * xs + b)); keep = r < np.percentile(r, 80); a, b = np.polyfit(xs[keep], ys[keep], 1)
lay = box | rob
comp = np.where(lay, a * inv_sim + b, comp)
yy, xx = np.mgrid[0:H, 0:W]; tb = rows_ok & ~np.isnan(bgm)
P = np.linalg.lstsq(np.stack([xx[tb], yy[tb], np.ones(tb.sum())], 1), bgm[tb], rcond=None)[0]
band_bg = P[0] * xx + P[1] * yy + P[2]; band_bg[:y0] = cv2.GaussianBlur(bgm[y0:y0 + 1].astype(np.float32), (0, 0), 6)
comp = np.where(~inside[None] & ~lay, band_bg[None], comp)
ctrlB = np.zeros(z["ctrl"].shape, np.uint8)
for c0 in range(0, T, WIN):
    sl = slice(c0, min(T, c0 + WIN)); ctrlB[sl] = FE.encode_window(comp[sl])
rng_v = np.percentile(comp[:, y0:y0 + h], 98) - np.percentile(comp[:, y0:y0 + h], 2)
z["ctrlB"] = ctrlB; np.savez_compressed(npz, **z)
print(f"ctrlB added to {npz}: {T} frames, sim->VDA fit residual {np.std(ys[keep] - (a * xs[keep] + b)) / rng_v * 100:.1f} % of the range")
