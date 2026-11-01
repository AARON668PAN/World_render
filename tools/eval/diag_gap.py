"""Where does the rendered val output differ from the real frames? For each val segment of an infer_g1i.py output dir:
  VAE ceiling   PSNR of the real window after a Wan VAE encode/decode (the best any latent model can reach)
  output        PSNR / per-region (robot, object, background) of the generated window
  source copy   PSNR of the source frames (shift 15) against the target, outside the holes: how much the
                scene moved in 1 s (head camera motion), i.e. how much the model has to re-place, not copy
  per frame     PSNR over the 49 frames (first frames have no reference)
and an error-map sheet (|output - real|, amplified) per segment. usage: diag_gap.py <out dir>"""
import glob, json, os, sys
import cv2, imageio.v2 as imageio, numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get("ROBORENDER_ROOT", "/data/repos/roborender")  # upstream RoboRender (core/)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "..", "..", "renderer"))  # inst_model
import inst_model as IM  # noqa: E402
from core.video import tensor_to_video  # noqa: E402

O = sys.argv[1]; R = "/data/datasets/unitree_g1_inspire"
pipe = IM.lite_pipe("cuda")
psnr = lambda a, b, m=None: float(10 * np.log10(255 ** 2 / max(1e-6, (((a.astype(np.float32) - b.astype(np.float32)) ** 2)[m] if m is not None else (a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())))
segs = {s["id"]: s for s in json.load(open(f"{R}/segments.json"))}
rows, sheet = [], []
for f in sorted(glob.glob(f"{O}/val_*.mp4")):
    c = os.path.basename(f)[4:-4]; v = [x for x in imageio.get_reader(f)]
    ctrl, src, out, real = [np.stack([x[:, i * 448:(i + 1) * 448] for x in v]) for i in range(4)]
    with torch.no_grad():
        lat = pipe.encode_video([Image.fromarray(x) for x in real])[0][None].to("cuda", torch.bfloat16)
        rec = np.stack([np.array(x) for x in tensor_to_video(pipe.vae.decode(lat, device="cuda"))])
    T = len(real); s0 = segs[c]["win_start"]
    rob = np.unpackbits(np.load(f"{R}/robot/{c}.npz")["mask"], axis=-1)[..., :448].astype(bool)[s0:s0 + T]
    op = f"{R}/obj1/{c}.npz"; obj = (np.load(op)["lab"][s0:s0 + T] > 0) & ~rob if os.path.exists(op) else np.zeros_like(rob)
    hole = (src == 128).all(-1); bg = ~rob & ~obj
    per = [psnr(out[k], real[k]) for k in range(T)]
    r = dict(seg=c[:34], vae=psnr(rec, real), out=psnr(out, real), robot=psnr(out, real, rob), obj=psnr(out, real, obj) if obj.any() else None,
             bg=psnr(out, real, bg), src_copy=psnr(src, real, ~hole), first8=float(np.mean(per[:8])), last8=float(np.mean(per[-8:])))
    rows.append(r)
    k = T // 2; err = np.clip(np.abs(out[k].astype(np.int16) - real[k]).mean(-1) * 4, 0, 255).astype(np.uint8)
    sheet.append(np.concatenate([real[k], out[k], cv2.applyColorMap(err, cv2.COLORMAP_INFERNO)[..., ::-1], src[k]], 1))
print(f"{'segment':34s} {'VAE':>6s} {'out':>6s} {'robot':>6s} {'obj':>6s} {'bg':>6s} {'srcCopy':>8s} {'f0-7':>6s} {'f41-48':>7s}")
for r in rows:
    print(f"{r['seg']:34s} {r['vae']:6.2f} {r['out']:6.2f} {r['robot']:6.2f} {(r['obj'] or 0):6.2f} {r['bg']:6.2f} {r['src_copy']:8.2f} {r['first8']:6.2f} {r['last8']:7.2f}")
cv2.imwrite(f"{O}/diag_errmap.jpg", np.concatenate(sheet, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(rows, open(f"{O}/diag_gap.json", "w"), indent=1)
