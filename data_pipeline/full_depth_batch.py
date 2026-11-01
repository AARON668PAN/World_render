"""Whole-frame depth for the v6 renderer control (10-31, user: full-scene depth instead of the object mask): VDA (relative
inverse depth, near = large) on every training segment (segments/<id>.mp4, 448x336, 15 fps; the 1043 leg-free ones with a
robot mask), the whole frame kept (obj_depth_batch.py kept only the object + robot). Stored as uint8 per segment:
inverse depth mapped linearly from its 0.5th..99.5th percentile over the segment to 0..255 (lo / hi kept in the file, so
the cache can re-normalise per window). full_depth/<id>.npz {d uint8 (T,336,448), lo, hi}. Skips done segments.
usage: /data/repos/sam3/.venv/bin/python full_depth_batch.py"""
import json, os, sys, time
import imageio.v2 as imageio, numpy as np, torch

R = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire"); OUT = f"{R}/full_depth"; os.makedirs(OUT, exist_ok=True)
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
legchk = json.load(open(f"{R}/leg_check.json"))
clips = [s["id"] for s in json.load(open(f"{R}/segments.json")) if not legchk.get(s["id"], {}).get("leg", False)
         and os.path.exists(f"{R}/robot/{s['id']}.npz") and os.path.exists(f"{R}/segments/{s['id']}.mp4")]
t0 = time.time(); n = 0
for i, c in enumerate(clips):
    dst = f"{OUT}/{c}.npz"
    if os.path.exists(dst):
        continue
    try:
        fr = np.stack([f for f in imageio.get_reader(f"{R}/segments/{c}.mp4")])
        with torch.no_grad():
            d = np.asarray(vda.infer_video_depth(fr, 15, input_size=518, device="cuda")[0], np.float32)
        lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.5)
        q = np.clip((d - lo) / max(hi - lo, 1e-6) * 255 + 0.5, 0, 255).astype(np.uint8)
        np.savez_compressed(dst + ".tmp.npz", d=q, lo=lo, hi=hi); os.replace(dst + ".tmp.npz", dst); n += 1
    except Exception as e:  # keep going; the cache skips segments without a depth file
        print(f"{c}: FAILED {type(e).__name__}: {e}", flush=True); torch.cuda.empty_cache(); continue
    if n % 25 == 0:
        print(f"{i + 1}/{len(clips)} done, {n} this run, {(time.time() - t0) / n:.1f} s/clip", flush=True)
print(f"FULL_DEPTH_DONE {n} new, {len(os.listdir(OUT))} files")
