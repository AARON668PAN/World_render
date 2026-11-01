"""Images for the v6 pipeline page (10-31): one training window and the tissue clip at inference, each renderer input as
its own picture, plus the depth composite. usage: python page_images.py OUT_DIR"""
import json, os, sys
import cv2, imageio.v2 as imageio, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "renderer"))  # objdepth_enc / ghost_holes of this repo
import objdepth_enc as OD, ghost_holes as GH
OUT = sys.argv[1]; R = "/data/datasets/unitree_g1_inspire"; E = "/data/repos/chord/local/ego/tissue_box"; H, W, G = 336, 448, 128
K3 = np.ones((3, 3), np.uint8); dil = lambda m: cv2.dilate(m.astype(np.uint8), K3) > 0
save = lambda n, im: cv2.imwrite(f"{OUT}/{n}.jpg", im[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86])


def shading(n01, slope=250.0):
    z = cv2.GaussianBlur(n01.astype(np.float32), (0, 0), 1.5) * slope
    gy, gx = np.gradient(z); nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    L = np.array([-0.5, -0.6, 0.62]); L /= np.linalg.norm(L); return np.clip(nrm @ L, 0, 1)


def turbo_shade(n01):
    t = cv2.applyColorMap((n01 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32)
    return np.clip(t * (0.35 + 0.8 * shading(n01)[..., None]), 0, 255).astype(np.uint8)


# ---- training window
c = "Put_Vegetables_Into_Basket_ep00012_s0394"
seg = [s for s in json.load(open(f"{R}/segments.json")) if s["id"] == c][0]
fr = np.stack([f for f in imageio.get_reader(f"{R}/segments/{c}.mp4")]); T = len(fr); s0 = seg["win_start"]; win = slice(s0, s0 + 49); k = 30; t = s0 + k
rob = np.unpackbits(np.load(f"{R}/robot/{c}.npz")["mask"], axis=-1)[..., :W].astype(bool)[:T]
obj = (np.load(f"{R}/obj1/{c}.npz")["lab"][:T] > 0) & ~rob
save("tr_target", fr[t])
ca = np.full((H, W, 3), G, np.uint8); ca[obj[t]] = (235, 60, 200); ca[rob[t]] = fr[t][rob[t]]; save("tr_ctrlA", ca)
z = np.load(f"{R}/full_depth/{c}.npz"); inv = z["d"].astype(np.float32) / 255 * (float(z["hi"]) - float(z["lo"])) + float(z["lo"])
od = np.load(f"{R}/obj_depth/{c}.npz")["inv"].astype(np.float32)
m = obj[win]; a, b = np.polyfit(od[win][m], inv[win][m], 1); inv[win][m] = a * od[win][m] + b
w_inv = OD.contact_fill(inv[win], obj[win], rob[win]); lo, hi = np.percentile(w_inv, 2), np.percentile(w_inv, 98)
save("tr_ctrlB", turbo_shade(np.clip((w_inv[k] - lo) / (hi - lo), 0, 1)))
rng = np.random.default_rng(7); gh = GH.ghost(rob[win], rng)
hole = dil(rob[t]) | gh[k]; save("tr_source", np.where(hole[..., None], G, fr[t]).astype(np.uint8))
save("tr_holemask", np.repeat((hole * 255).astype(np.uint8)[..., None], 3, -1))
save("tr_ref", fr[s0 - 1])
for i, ti in enumerate([2, 18, 92]):  # frames outside the window (window 31..79)
    save(f"tr_info{i + 1}", np.where(dil(rob[ti])[..., None], G, fr[ti]).astype(np.uint8))
# ---- tissue clip, frame 60 (chunk 1)
zz = np.load(f"{E}/human_g1i_letterbox_obj_realcol.npz"); band = zz["band"]
lab = np.load(f"{E}/depth_compare/v7_letterbox/sim_depth.npz")["label"][::2]
ht = np.load(f"{E}/align/human_table_masks.npz"); vw = int(ht["shape"][2])
hum30 = (np.unpackbits(ht["hand"], axis=-1) | np.unpackbits(ht["forearm"], axis=-1))[..., :vw]
vid = [f for i, f in enumerate(imageio.get_reader("/data/repos/chord/reconstruction/assets/tissue_box.mp4")) if i % 2 == 0]
def human(k):
    img = np.full((H, W, 3), G, np.uint8); img[42:294] = cv2.resize(vid[k], (W, 252), interpolation=cv2.INTER_AREA)
    hm = np.zeros((H, W), bool); hm[42:294] = cv2.resize(hum30[2 * k], (W, 252), interpolation=cv2.INTER_NEAREST) > 0
    return img, hm
k = 60; img, hm = human(k); rb = np.isin(lab[k], (3, 4))
save("inf_human", img); save("inf_ctrlA", zz["ctrl"][k])
hole = band | dil(hm) | dil(rb); save("inf_source", np.where(hole[..., None], G, img).astype(np.uint8))
save("inf_holemask", np.repeat((hole * 255).astype(np.uint8)[..., None], 3, -1))
for i, ti in enumerate([0, 24, 108]):
    im2, h2 = human(ti); save(f"inf_info{i + 1}", np.where((band | dil(h2))[..., None], G, im2).astype(np.uint8))
print("ok")
