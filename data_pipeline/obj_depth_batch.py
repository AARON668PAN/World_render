"""Object (+ robot) inverse depth for every segment with an object map (obj1/), for the C1 object-depth control (10-30):
VDA relative on the whole segment at the original 640x480, object = obj1 mask (rim 4 px filled from the inside), small
objects (median box side < 120 px) get a second VDA pass on a crop fixed over the window (3x the union box, affine-fitted
back onto the full-frame depth). Saved at the cache resolution 448x336, float16, zero outside object | robot (dilated):
obj_depth/<clip>.npz {inv, small}. Normalisation / shading happen in cache_g1i.py (--obj_enc c1). Skips done clips.
usage: /data/repos/sam3/.venv/bin/python obj_depth_batch.py [--rank R --world N]"""
import json, os, subprocess, sys, time
import cv2, numpy as np, torch
from scipy.ndimage import distance_transform_edt

R = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire"); OUT = f"{R}/obj_depth"; os.makedirs(OUT, exist_ok=True)
W0, H0, W, H = 640, 480, 448, 336
rank = int(sys.argv[sys.argv.index("--rank") + 1]) if "--rank" in sys.argv else 0
world = int(sys.argv[sys.argv.index("--world") + 1]) if "--world" in sys.argv else 1
sys.path.insert(0, os.environ.get("VDA_ROOT", "/data/repos/video_depth_anything"))  # Video-Depth-Anything clone
from video_depth_anything.dinov2_layers import attention as _att
from video_depth_anything.video_depth import VideoDepthAnything


def _sdpa(self, x, attn_bias=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    return self.proj_drop(self.proj(torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(B, N, C)))


_att.Attention.forward = _sdpa; _att.MemEffAttention.forward = _sdpa
vda = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=False)
vda.load_state_dict(torch.load(os.environ.get("VDA_CKPT", "/data/models/video_depth_anything/large/video_depth_anything_vitl.pth"), map_location="cpu"), strict=True)
vda = vda.to("cuda").eval()
depth = lambda fr: np.asarray(vda.infer_video_depth(fr, 15, input_size=518, device="cuda")[0], np.float32)
segs = {s["id"]: s for s in json.load(open(f"{R}/segments.json"))}
index = {(e["dataset"], e["episode"]): e for e in json.load(open(f"{R}/index.json"))}


def raw_frames(c):
    s = segs[c]; e = index[(s["dataset"], s["episode"])]; r0, n = s["raw"][0], len(s["raw"])
    vp = f"{R}/official/{s['dataset']}/videos/observation.images.head_stereo_left/chunk-{e['chunk']:03d}/file-{e['file']:03d}.mp4"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{r0 / 30:.4f}", "-i", vp, "-frames:v", str(2 * n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H0, W0, 3)[::2][:n]


def up(m):
    return np.stack([cv2.resize(x.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0 for x in m])


def fill_rim(d, m, px=4):
    inner = cv2.erode(m.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0
    if inner.sum() < 20:
        return d
    _, (iy, ix) = distance_transform_edt(~inner, return_indices=True)
    out = d.copy(); rim = m & ~inner; out[rim] = d[iy[rim], ix[rim]]; return out


clips = sorted(f[:-4] for f in os.listdir(f"{R}/obj1") if f.endswith(".npz"))[rank::world]
t0 = time.time(); n_done = 0
for i, c in enumerate(clips):
    dst = f"{OUT}/{c}.npz"
    if os.path.exists(dst):
        continue
    try:
        fr = raw_frames(c); s0 = segs[c]["win_start"]
        lab448 = np.load(f"{R}/obj1/{c}.npz")["lab"] > 0
        rob448 = np.unpackbits(np.load(f"{R}/robot/{c}.npz")["mask"], axis=-1)[..., :W].astype(bool)
        T = min(len(fr), len(lab448), len(rob448)); fr, lab448, rob448 = fr[:T], lab448[:T], rob448[:T]
        win = slice(s0, min(T, s0 + 49))
        rob, obj = up(rob448), up(lab448) & ~up(rob448)
        with torch.no_grad():
            d = depth(fr)
        d_obj = d.copy(); small = False
        boxes = [max(np.ptp(np.nonzero(m)[0]), np.ptp(np.nonzero(m)[1])) for m in obj[win] if m.sum() > 20]
        if boxes and np.median(boxes) < 120:
            small = True; ys, xs = np.nonzero(obj[win].any(0))
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            w = min(max(3 * (xs.max() - xs.min()), 4 / 3 * 3 * (ys.max() - ys.min()), 160), W0); h = min(w * 3 / 4, H0)
            x0, y0 = int(np.clip(cx - w / 2, 0, W0 - w)), int(np.clip(cy - h / 2, 0, H0 - h)); x1, y1 = min(W0, int(x0 + w)), min(H0, int(y0 + h))
            crop = np.stack([cv2.resize(f[y0:y1, x0:x1], (W0, H0), interpolation=cv2.INTER_CUBIC) for f in fr])
            with torch.no_grad():
                dc = depth(crop)
            dc = np.stack([cv2.resize(x, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA) for x in dc])
            a, b = np.polyfit(dc[win].ravel()[::7], d[win][:, y0:y1, x0:x1].ravel()[::7], 1)
            d_obj[:, y0:y1, x0:x1] = a * dc + b
        d_obj = np.stack([fill_rim(d_obj[t], obj[t]) for t in range(T)])
        comb = np.where(obj, d_obj, d)  # object: refined depth; robot: full-frame depth (for the joint normalisation)
        small_res = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA) for x in comb])
        keep = np.stack([cv2.dilate((a_ | b_).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 for a_, b_ in zip(lab448, rob448)])
        np.savez_compressed(dst, inv=np.where(keep, small_res, 0).astype(np.float16), small=small)
        n_done += 1
    except Exception as e:  # keep going; a missing file just means flat colour for that clip
        print(f"{c}: FAILED {type(e).__name__}: {e}", flush=True); torch.cuda.empty_cache(); continue
    if n_done % 10 == 0:
        print(f"{i + 1}/{len(clips)} done {n_done} this run, {(time.time() - t0) / n_done:.1f} s/clip", flush=True)
print(f"OBJ_DEPTH_DONE {n_done} new, {len(os.listdir(OUT))} files")
