"""Automatic checks of the object maps (obj1/<clip>.npz from box_sam3.py) before they go into the cache:
  seed_fit   IoU of the mask's bounding box with Claude's box in the seed frame (low: SAM3 took something else)
  win_vis    share of the 49 window frames where the object is visible (> 30 px)
  jump       largest frame-to-frame area ratio inside the window (tracking jumped to / from another object)
  on_robot   share of the window's object pixels that lie on the SAM3 robot mask (tracking the hand instead)
  touch      frames of the whole segment where the object touches the dilated robot mask (0: never handled)
  far        largest centroid jump between consecutive frames in the window, px (switched to another object)
Writes check/obj_qc.json (per clip) and prints how many clips each flag hits.
usage: python obj_qc.py"""
import glob, json, os
import cv2, numpy as np

R = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire")  # dataset work dir
segs = {s["id"]: s for s in json.load(open(f"{R}/segments.json"))}
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
out = {}
for f in sorted(glob.glob(f"{R}/obj1/*.npz")):
    c = os.path.basename(f)[:-4]; z = np.load(f); m = z["lab"] > 0; T = len(m)
    rz = np.load(f"{R}/robot/{c}.npz"); rob = np.unpackbits(rz["mask"], axis=-1)[..., :448].astype(bool)[:T]
    s0 = segs[c]["win_start"]; w = slice(s0, min(T, s0 + 49))
    k0 = int(z["frame"]); x0, y0, x1, y1 = [float(v) for v in z["box"]]
    ys, xs = np.nonzero(m[k0])
    if len(xs):
        bx = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        ix = max(0, min(x1, bx[2]) - max(x0, bx[0])) * max(0, min(y1, bx[3]) - max(y0, bx[1]))
        seed_fit = ix / ((x1 - x0) * (y1 - y0) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - ix)
    else:
        seed_fit = 0.0
    area = m[w].reshape(m[w].shape[0], -1).sum(1).astype(float)
    vis = area > 30
    a = area[vis]; jump = float(np.max(np.maximum(a[1:], a[:-1]) / np.maximum(np.minimum(a[1:], a[:-1]), 1))) if len(a) > 1 else 1.0
    cen = [np.argwhere(x).mean(0) if x.sum() > 30 else None for x in m[w]]
    far = max([float(np.linalg.norm(cen[i] - cen[i - 1])) for i in range(1, len(cen)) if cen[i] is not None and cen[i - 1] is not None] or [0.0])
    on_robot = float((m[w] & rob[w]).sum() / max(1, m[w].sum()))
    rd = np.stack([cv2.dilate(x.astype(np.uint8), K) > 0 for x in rob])
    touch = int((m & rd).reshape(T, -1).any(1).sum())
    out[c] = dict(seed_fit=round(float(seed_fit), 3), win_vis=round(float(vis.mean()), 3), jump=round(jump, 2), far=round(far, 1),
                  on_robot=round(on_robot, 3), touch=touch, name=str(z["name"]))
json.dump(out, open(f"{R}/check/obj_qc.json", "w"), indent=0)
flags = {"seed_fit<0.5": lambda d: d["seed_fit"] < 0.5, "win_vis<0.3": lambda d: d["win_vis"] < 0.3, "jump>4": lambda d: d["jump"] > 4,
         "far>60px": lambda d: d["far"] > 60, "on_robot>0.3": lambda d: d["on_robot"] > 0.3, "touch==0": lambda d: d["touch"] == 0}
for k, fn in flags.items():
    print(f"{k:14s} {sum(fn(d) for d in out.values()):4d}")
anyf = [c for c, d in out.items() if any(fn(d) for fn in flags.values())]
print(f"any flag: {len(anyf)} of {len(out)}")
json.dump(anyf, open(f"{R}/check/obj_qc_flagged.json", "w"))
