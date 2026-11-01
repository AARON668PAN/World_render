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


NAMES = ["grey depth", "grey depth x shading", "turbo depth x shading", "R,B depth  G shading"]
segs = {s["id"]: s for s in json.load(open(f"{R}/segments.json"))}
rows = []
for c in ["Empty_Folder_In_Cabinet_ep00038_s0135", "Put_Sink_Towel_ep00009_s0094", "Put_Vegetables_Into_Basket_ep00012_s0394", "Pick_Up_Drinks_v0_ep00046_s0466"]:
    fr = np.stack([f for f in imageio.get_reader(f"{R}/segments/{c}.mp4")]); T = len(fr)
    s0 = segs[c]["win_start"]; win = slice(s0, min(T, s0 + 49))
    inv = load_full(c)[:T]
    rob = np.unpackbits(np.load(f"{R}/robot/{c}.npz")["mask"], axis=-1)[..., :W].astype(bool)[:T]
    obj = (np.load(f"{R}/obj1/{c}.npz")["lab"][:T] > 0) & ~rob
    od = np.load(f"{R}/obj_depth/{c}.npz")["inv"].astype(np.float32)[:T]
    m = obj[win]
    if m.sum() > 200:  # refined object depth, affine-fitted onto the whole-frame depth over the window
        a, b = np.polyfit(od[win][m], inv[win][m], 1); inv[win][m] = a * od[win][m] + b
    w_inv = OD.contact_fill(inv[win], obj[win], rob[win])
    n = norm_window(w_inv); k = 30
    rows.append(np.hstack([tag(fr[win][k], f"real  {c[:28]}")] + [tag(e, nm) for e, nm in zip(encodings(n[k]), NAMES)]))

# inference: tissue clip composite
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
comp = np.where(lay, a * inv_sim + b, comp)
# bands: below the image the table continues - inverse depth of a plane is linear in the image coordinates, so fit
# it on the clean table background of the lower image rows and extend it; above, the top row smoothed and repeated
yy, xx = np.mgrid[0:H, 0:W]; tb = rows_ok & ~np.isnan(bgm)
P = np.linalg.lstsq(np.stack([xx[tb], yy[tb], np.ones(tb.sum())], 1), bgm[tb], rcond=None)[0]
plane = P[0] * xx + P[1] * yy + P[2]
top = cv2.GaussianBlur(bgm[y0:y0 + 1].astype(np.float32), (0, 0), 6)
band_bg = plane.copy(); band_bg[:y0] = top
for t in range(T):
    comp[t] = np.where(~inside & ~lay[t], band_bg, comp[t])
k0 = 49; n = norm_window(comp[k0:k0 + 49]); k = 11  # frame 60
rows.append(np.hstack([tag(img[k0 + k], "INFERENCE human video f60", (120, 255, 120))] + [tag(e, nm, (120, 255, 120)) for e, nm in zip(encodings(n[k]), NAMES)]))
nv = norm_window(np.where(inside[None], v[k0:k0 + 49], np.nan)[:, y0:y0 + h])
raw = np.zeros((H, W)); raw[y0:y0 + h] = nv[k]
rows.append(np.hstack([tag(img[k0 + k], "for reference: raw VDA of the human video")] + [tag(e, nm) for e, nm in zip(encodings(raw), NAMES)]))
cv2.imwrite(f"{R}/check/fulldepth_preview.jpg", np.vstack(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
print("ok")
