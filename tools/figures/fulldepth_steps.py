"""Preview of the v6 whole-frame depth control (control B), 10-31. Training (real G1 segments): full_depth/ VDA of the whole
frame, the handled object replaced by the refined object depth of obj_depth/ (crop pass for small objects, rim filled;
affine-fitted onto the frame's depth), contact fill near the robot, normalised once per 49-frame window over the whole
frame. Inference (tissue clip): VDA of the human video with the human hands and the human-video box replaced by the clean
background (static camera: per-pixel temporal median over frames where neither covers it), the sim table/box/robot depth
fitted onto the VDA scale (affine on table pixels both see) and laid on top where the sim box or robot is, letterbox
bands filled from the nearest image row. Four encodings side by side: grey depth, grey depth x shading, turbo depth x
shading, two channels (R,B = depth, G = shading). -> check/fulldepth_preview.jpg"""
import json, os, sys
import cv2, imageio.v2 as imageio, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "renderer"))  # objdepth_enc / ghost_holes of this repo
import objdepth_enc as OD

R = "/data/datasets/unitree_g1_inspire"; E = "/data/repos/chord/local/ego/tissue_box"; H, W = 336, 448


def load_full(c):
    z = np.load(f"{R}/full_depth/{c}.npz")
    return z["d"].astype(np.float32) / 255 * (float(z["hi"]) - float(z["lo"])) + float(z["lo"])


def norm_window(inv):
    lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
    return np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)


def shading(n01, slope=250.0):
    z = cv2.GaussianBlur(n01.astype(np.float32), (0, 0), 1.5) * slope
    gy, gx = np.gradient(z)
    nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    L = np.array([-0.5, -0.6, 0.62]); L /= np.linalg.norm(L)
    return np.clip(nrm @ L, 0, 1)


def encodings(n01):
    s = shading(n01)
    g = (n01 * 255).astype(np.uint8)
    e1 = np.repeat(g[..., None], 3, -1)
    e2 = np.repeat((np.clip(n01 * (0.35 + 0.8 * s), 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
    t = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32)
    e3 = np.clip(t * (0.35 + 0.8 * s[..., None]), 0, 255).astype(np.uint8)
    e4 = np.stack([g, (s * 255).astype(np.uint8), g], -1)
    return [e1, e2, e3, e4]


def tag(im, s, col=(255, 255, 255)):
    im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(im, s, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA); return im



# inference composite, step by step

z = np.load(f"{E}/depth_compare/v7_letterbox/sim_depth.npz"); lab, dz = z["label"][::2], z["depth"][::2].astype(np.float32)
vid = [f for i, f in enumerate(imageio.get_reader("/data/repos/chord/reconstruction/assets/tissue_box.mp4")) if i % 2 == 0]
T = min(len(lab), len(vid)); lab, dz, vid = lab[:T], dz[:T], vid[:T]
sys.path.insert(0, "/data/repos/video_depth_anything")
from video_depth_anything.dinov2_layers import attention as _att
from video_depth_anything.video_depth import VideoDepthAnything
def _sdpa(self, x, attn_bias=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    return self.proj_drop(self.proj(torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(B, N, C)))
_att.Attention.forward = _sdpa; _att.MemEffAttention.forward = _sdpa
vda = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=False)
vda.load_state_dict(torch.load("/data/models/video_depth_anything/large/video_depth_anything_vitl.pth", map_location="cpu"), strict=True)
vda = vda.to("cuda").eval()
with torch.no_grad():
    d = np.asarray(vda.infer_video_depth(np.stack(vid), 15, input_size=518, device="cuda")[0], np.float32)
y0, h = 42, 252
v = np.zeros((T, H, W), np.float32); img = np.full((T, H, W, 3), 128, np.uint8)
for t in range(T):
    v[t, y0:y0 + h] = cv2.resize(d[t], (W, h), interpolation=cv2.INTER_AREA); img[t, y0:y0 + h] = cv2.resize(vid[t], (W, h), interpolation=cv2.INTER_AREA)
ht = np.load(f"{E}/align/human_table_masks.npz"); vw = int(ht["shape"][2])
hum30 = (np.unpackbits(ht["hand"], axis=-1) | np.unpackbits(ht["forearm"], axis=-1))[..., :vw]
hum = np.zeros((T, H, W), bool)
for t in range(T):
    hum[t, y0:y0 + h] = cv2.dilate(cv2.resize(hum30[2 * t], (W, h), interpolation=cv2.INTER_NEAREST), np.ones((5, 5), np.uint8)) > 0
box = lab == 2; rob = np.isin(lab, (3, 4)); table = lab == 1
inside = np.zeros((H, W), bool); inside[y0:y0 + h] = True
cover = hum | cv2.dilate(box.reshape(-1, W).astype(np.uint8), np.ones((5, 5), np.uint8)).reshape(T, H, W).astype(bool)
bg = np.where(cover, np.nan, v); bgm = np.nanmedian(bg, 0)  # clean background (static camera)
bgm = np.where(np.isnan(bgm), np.nanmedian(v, 0), bgm)
comp = np.where(cover, bgm[None], v)
inv_sim = 1.0 / np.maximum(dz, 1e-3)
# table pixels both see: the sim "table" label also covers the back wall the sim does not have, so only the table rows
# in front of the box (below its lowest point) count; robust fit (two rounds, drop the worst 20 %)
ys = np.nonzero(box.any((0, 2)))[0]; ylow = int(np.percentile(np.nonzero(box.any(2))[1], 99)) if box.any() else H // 2
rows_ok = np.zeros((H, W), bool); rows_ok[ylow + 5:y0 + h] = True
fit = table & ~cover & rows_ok[None]
xs_, ys_ = inv_sim[fit][::20], comp[fit][::20]
a, b = np.polyfit(xs_, ys_, 1); r = np.abs(ys_ - (a * xs_ + b)); keep = r < np.percentile(r, 80); a, b = np.polyfit(xs_[keep], ys_[keep], 1)
rng_v = np.percentile(comp[:, y0:y0 + h], 98) - np.percentile(comp[:, y0:y0 + h], 2)
print(f"table fit residual {np.std(ys_[keep] - (a * xs_[keep] + b)) / rng_v * 100:.1f} % of the VDA range")
lay = box | rob
comp_bg = comp.copy(); comp = np.where(lay, a * inv_sim + b, comp)
# bands: below the image the table continues - inverse depth of a plane is linear in the image coordinates, so fit
# it on the clean table background of the lower image rows and extend it; above, the top row smoothed and repeated
yy, xx = np.mgrid[0:H, 0:W]; tb = rows_ok & ~np.isnan(bgm)
P = np.linalg.lstsq(np.stack([xx[tb], yy[tb], np.ones(tb.sum())], 1), bgm[tb], rcond=None)[0]
plane = P[0] * xx + P[1] * yy + P[2]
top = cv2.GaussianBlur(bgm[y0:y0 + 1].astype(np.float32), (0, 0), 6)
band_bg = plane.copy(); band_bg[:y0] = top
for t in range(T):
    comp[t] = np.where(~inside & ~lay[t], band_bg, comp[t])

k = 60
lo, hi = np.percentile(comp[49:98], 2), np.percentile(comp[49:98], 98)
N = lambda x: np.clip((x - lo) / (hi - lo), 0, 1)
def turbo(n01, sh=True):
    g = (np.nan_to_num(n01) * 255).astype(np.uint8); t = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32)
    return np.clip(t * ((0.35 + 0.8 * shading(n01)[..., None]) if sh else 1), 0, 255).astype(np.uint8)
s1 = turbo(N(v[k]), False); s1[~inside] = 0
s2 = s1.copy(); s2[cover[k]] = 128
s3 = turbo(N(np.where(inside, comp_bg[k], np.nan)), False); s3[~inside] = 0
simv = a * inv_sim[k] + b; s4 = np.full((H, W, 3), 0, np.uint8); s4[lay[k]] = turbo(N(simv), False)[lay[k]]
s5 = turbo(N(comp[k]), False)
s6 = turbo(N(comp[k]), True)
tiles = [tag(img[k], "0 human video f60"), tag(s1, "1 VDA of the human video"), tag(s2, "2 remove hands + box (grey)"),
         tag(s3, "3 fill from other frames"), tag(s4, "4 sim robot + box depth"), tag(s5, "5 layer 4 on 3, fill bands"), tag(s6, "6 + shading (final)")]
top = np.hstack(tiles[:4]); bot = np.hstack(tiles[4:] + [np.zeros_like(tiles[0])])
cv2.imwrite(f"{R}/check/fulldepth_inference_steps.jpg", np.vstack([top, bot])[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
cv2.imwrite(sys.argv[1] + "/inf_ctrlB.jpg", s6[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86]) if len(sys.argv) > 1 else None
print("ok")
