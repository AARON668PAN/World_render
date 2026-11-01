"""Training cache for the source-video renderer on Unitree G1 + Inspire (official WBT Inspire, leg-free segments). Not part of
the upstream repo. Adapted from cache_v2.py (Unitree Dex3 pilot, not shipped): per segment (segments.json + segments/<id>.mp4 from
data_pipeline/extract_windows.py, 15 fps, 448x336, the 49-frame window plus up to 2 s on each side) and its
SAM3 robot mask (robot/<id>.npz: "robotic arm", edge-entering tracks):
  (v3, 10-30: with p --aligned the source is the SAME window (shift 0, robot holed), so the model learns to reproduce known
  scene pixels exactly, as needed for the human clip; object holes p --obj_hole 0.15)
  z0 target window, y control = grey + the ONE handled object (SAM3 instance, random colour, see manipulated()) + the REAL
  robot cut out on top (the first G1i run on 10-29 had no objects by my mistake), ref + clip = frame
  before the window, src = the same segment shifted by +-DMIN..DMAX frames (only where the joint-state classifier says
  leg-free) with grey holes (robot at source and target time, a human hand/arm shape with p .7, with p .4 grey bands at the
  top and bottom as for a letterboxed 16:9 human video (inference: no crop, 42 px bands); the wide-lens ring hole of human_footprint is off by default), colour/texture aug on
  everything but the robot (same for target, source, ref). Segments the SAM3 spot check (leg_check.json) flags are skipped.
  (v5, 10-31: --info K scene frames of the same segment at other moments, robot holed, picked to show what the source
  hides over the window; --obj_stick: hand-sized holes glued to the handled object for the whole window, so the only place
  to see what is under them is the info frames - the human thumbs on the tissue box left grey patches in v3)
  (v6, 10-31, user: holes thin enough to keep the fingers, no random hand stamps: --hole_px 1 --hand_stamps 0; --ghost p:
  a second finger-detailed copy of the robot mask per arm, posed a little differently (ghost_holes.py, calibrated on the
  tissue clip), as a human hand at roughly the robot's place leaves; --mask_pool mean: the hole channel keeps the share of
  hole pixels per 8x8 cell; --depth: control B = whole-frame depth (fulldepth_enc.py) stored as "yd", it goes into the
  base model's unused first-frame slot; --mask_drop p: with depth, no object colour in control A)
Usage: cache_g1i.py OUT [--rank R --world N]"""
import argparse
import colorsys
import glob
import os
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ROBORENDER_ROOT", "/data/repos/roborender")  # upstream RoboRender clone (core/)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)  # inst_model + sibling modules live next to this script
import inst_model as IM  # noqa: E402
import objdepth_enc as OD  # noqa: E402
import fulldepth_enc as FE  # noqa: E402
import ghost_holes as GH  # noqa: E402
from core.video import video_to_tensor, image_to_tensor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--root", default="/data/datasets/unitree_g1_inspire")
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--world", type=int, default=1)
ap.add_argument("--win", type=int, default=49)
ap.add_argument("--dmin", type=int, default=8)
ap.add_argument("--dmax", type=int, default=30)
ap.add_argument("--hands", default="/data/repos/chord/local/ego/tissue_box/align/human_table_masks.npz",
                help="human hand/forearm masks of the tissue clip (sim_bridge/sam3_union_masks.py), the shapes of the random hand holes; read only when --hand_stamps > 0 or --ring > 0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ring", type=float, default=0.0, help="probability of the wide-lens ring hole (human video seen through the G1 lens); off: the user found the ring too large")
ap.add_argument("--obj_dir", default="obj1", help="object maps under ROOT: obj1 = the ONE Claude-boxed object tracked by SAM3 (data_pipeline/box_sam3.py), obj = SAM3 per-noun instances")
ap.add_argument("--obj_enc", default="flat", choices=["flat", "c1"], help="object in the control: flat random colour, or c1 = depth (hue distance, shading shape) from ROOT/obj_depth")
ap.add_argument("--aligned", type=float, default=0.5, help="probability of an aligned source (shift 0, same time as the target, robot holed); the rest shifted by +-dmin..dmax")
ap.add_argument("--obj_hole", type=float, default=0.15, help="probability of greying the handled object in the source (was 0.3 in g1i_obj1_2g: it learned to invent objects)")
ap.add_argument("--info", type=int, default=0, help="v5: number of scene (info) frames from other moments of the segment, robot holed; 0 = none (v3/v4)")
ap.add_argument("--obj_stick", type=float, default=0.0, help="v5: probability of holes that stay on the handled object for the whole window (a thumb on a box)")
ap.add_argument("--hole_px", type=int, default=5, help="v6: margin of the robot hole in px (5 = the old 11x11 ellipse; 1 = keeps the fingers)")
ap.add_argument("--hand_stamps", type=float, default=0.7, help="probability of random human-hand-shaped holes anywhere (v6: 0, the user found them wrong)")
ap.add_argument("--ghost", type=float, default=0.0, help="v6: probability of a ghost hole (the robot mask posed a little differently, ghost_holes.py)")
ap.add_argument("--mask_pool", default="max", choices=["max", "mean"], help="hole channel per 8x8 cell: max (any hole pixel) or mean (share of hole pixels, v6)")
ap.add_argument("--depth", action="store_true", help="v6b: control B, whole-frame depth from ROOT/full_depth (+ obj_depth refinement)")
ap.add_argument("--mask_drop", type=float, default=0.0, help="v6b: probability of no object colour in control A (depth only)")
ap.add_argument("--band", type=float, default=0.4, help="probability of grey bands at the top and bottom (16:9 human video letterboxed into 4:3, the user's choice for inference)")
a = ap.parse_args()
os.makedirs(f"{a.out}/qa", exist_ok=True)
rng = np.random.default_rng(a.seed + a.rank)
W, H, GRAY = 448, 336, 128
K_ROB = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
K_OBJ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
K_HOLE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.hole_px + 1, 2 * a.hole_px + 1))
pipe = IM.lite_pipe("cuda")
hum = None  # human hand/forearm shapes for hand_holes(), loaded on first use (v6 runs with --hand_stamps 0 --ring 0 never need them)


def load_hands():
    global hum
    if hum is None:
        ht = np.load(a.hands)
        hm = (np.unpackbits(ht["hand"], axis=-1) | np.unpackbits(ht["forearm"], axis=-1))[..., :int(ht["shape"][2])][::2].astype(np.uint8)
        hum = np.stack([cv2.resize(m, (W, int(W * m.shape[0] / m.shape[1])), interpolation=cv2.INTER_NEAREST) for m in hm])
    return hum


def hand_holes(n, Wo=W, Ho=H):
    load_hands()
    s0 = rng.integers(0, max(1, len(hum) - n))
    seg = hum[s0:s0 + n]
    if len(seg) < n:
        seg = np.concatenate([seg, np.repeat(seg[-1:], n - len(seg), 0)])
    sc, flip = rng.uniform(0.7, 1.1) * Wo / W, rng.random() < 0.5
    dx, dy = int(rng.integers(-80, 81) * Wo / W), int(rng.integers(-20, 90) * Ho / H)
    out = np.zeros((n, Ho, Wo), bool)
    for k, m in enumerate(seg):
        m = cv2.resize(m, (int(W * sc), int(m.shape[0] * sc)), interpolation=cv2.INTER_NEAREST)
        if flip:
            m = m[:, ::-1]
        h, w = m.shape
        y0, x0 = Ho - h + dy, (Wo - w) // 2 + dx
        ys, xs, ye, xe = max(0, y0), max(0, x0), min(Ho, y0 + h), min(Wo, x0 + w)
        if ye > ys and xe > xs:
            out[k, ys:ye, xs:xe] = m[ys - y0:ye - y0, xs - x0:xe - x0] > 0
    return out


# Human video at inference: re-projected into the G1 head lens (wide, ~125 deg diagonal), where it covers only the middle
def stick_holes(objm):
    """1-2 hand-sized ellipses fixed in the object's box coordinates (mostly at its left / right side, where hands hold
    things), so they move with the object over the window and hide the same part of it in every frame"""
    params = []
    for _ in range(rng.integers(1, 3)):
        u = rng.uniform(-0.1, 0.35) if rng.random() < 0.35 else rng.uniform(0.65, 1.1) if rng.random() < 0.55 else rng.uniform(0, 1)
        params.append((u, rng.uniform(0.1, 0.9), rng.uniform(0.12, 0.35), rng.uniform(0.15, 0.45), rng.uniform(0, 180)))
    out = np.zeros(objm.shape, bool)
    for k, m in enumerate(objm):
        if m.sum() < 30:
            continue
        ys, xs = np.nonzero(m); x0, y0 = xs.min(), ys.min(); bw, bh = xs.max() - x0 + 1, ys.max() - y0 + 1
        img = np.zeros((H, W), np.uint8)
        for u, v, rx, ry, ang in params:
            cv2.ellipse(img, (int(x0 + u * bw), int(y0 + v * bh)), (max(4, int(rx * bw)), max(4, int(ry * bh))), ang, 0, 360, 1, -1)
        out[k] = img > 0
    return out


def pick_info(T, ok, s, rob_d, hole, k):
    """k frames of the segment outside the window (>= 6 frames off it and off the reference frame), greedy: each next one
    shows the most of what the source hides over the window (a cell hidden in many window frames counts more); with p .3
    the earliest usable frame goes first (a look at the scene before the work, as the first frame of a human clip)"""
    cand = [t for t in range(T) if ok[t] and (t < s - 7 or t >= s + a.win + 6)]
    if len(cand) < k:
        cand = [t for t in range(T) if ok[t] and (t < s - 1 or t >= s + a.win)]
    if not cand:
        return []
    cell = lambda m: m.reshape(-1, H // 8, 8, W // 8, 8).mean((2, 4))
    need = cell(hole).mean(0)
    vis = {t: 1 - cell(rob_d[t:t + 1])[0] for t in cand}
    pick = [cand[0]] if rng.random() < 0.3 else []
    for t in pick:
        need = need * (1 - vis[t])
    while len(pick) < k:
        rest = [t for t in cand if t not in pick] or cand
        t = max(rest, key=lambda t: (need * vis[t]).sum() + 1e-3 * rng.random())
        pick.append(t); need = need * (1 - vis[t])
    return sorted(pick)


# (~36 % for the 68 deg tissue clip); the ring outside is a hole the model has to fill. Training: with p RING the source
# gets that ring hole for a random narrower pinhole camera (HFOV 55-90 deg, 16:9 or 4:3, tilted a little), and the
# hand holes are drawn in that camera's frame and mapped into the lens, as a human video would be.
FISH_F = float(os.environ.get("G1_FISH_F", 256.7))  # equidistant fisheye focal at 448x336 (125 deg diagonal at 640x480)
_px, _py = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
_dx, _dy = (_px - W / 2) / FISH_F, (_py - H / 2) / FISH_F
_th = np.hypot(_dx, _dy); _sc = np.sin(_th) / np.maximum(_th, 1e-9)
RAYS = np.stack([_dx * _sc, _dy * _sc, np.cos(_th)], -1)  # unit rays of the lens, OpenCV camera axes


def _rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]); Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def human_footprint():
    """(inside mask HxW, remap x, remap y, Wv, Hv) of a random narrower pinhole camera seen through the G1 lens."""
    hfov = np.radians(rng.uniform(55, 90)); Wv = W; Hv = int(round(W / (16 / 9 if rng.random() < 0.6 else 4 / 3)))
    f = Wv / 2 / np.tan(hfov / 2)
    r = RAYS @ _rot(*np.radians([rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(-5, 5)]))  # lens rays in the pinhole frame
    z = np.maximum(r[..., 2], 1e-6)
    mx, my = (f * r[..., 0] / z + Wv / 2).astype(np.float32), (f * r[..., 1] / z + Hv / 2).astype(np.float32)
    inside = (r[..., 2] > 0) & (mx >= 0) & (mx <= Wv - 1) & (my >= 0) & (my <= Hv - 1)
    return inside, mx, my, Wv, Hv


def track_dip(area, k=3):
    """A lost SAM3 track: the robot area drops below half of what it is both just before and just after (within k
    frames). Arms entering or leaving the view change the area gradually and are kept."""
    for t in range(1, len(area) - 1):
        before, after = area[max(0, t - k):t].max(), area[t + 1:t + 1 + k].max()
        if min(before, after) > 0.01 and area[t] < 0.5 * min(before, after):
            return True
    return False


def cells(m, how):
    x = m.astype(np.float32).reshape(len(m), H // 8, 8, W // 8, 8)
    x = x.max((2, 4)) if how == "max" else x.mean((2, 4))
    rest = x[1:].reshape((len(m) - 1) // 4, 4, H // 8, W // 8)
    return np.concatenate([x[:1], rest.max(1) if how == "max" else rest.mean(1)], 0)


def palette(n):
    """n clearly different saturated colours (hues spread around the circle, random start), one per object instance"""
    h0 = rng.random()
    hs = (h0 + np.arange(n) / max(n, 1) + rng.uniform(-0.2, 0.2, n) / max(n, 1)) % 1
    rng.shuffle(hs)
    return [np.array([int(255 * c) for c in colorsys.hsv_to_rgb(h, rng.uniform(0.75, 1), rng.uniform(0.85, 1))], np.uint8) for h in hs]


def load_instances(c, T, rob):
    """(T,H,W) uint8 instance ids: obj_inst/ (per-instance tracking) if present, else the union mask as one instance"""
    p = f"{a.root}/{a.obj_dir}/{c}.npz"  # obj1: data_pipeline/box_sam3.py (one object, id 1); obj: SAM3 track_prompt.py INSTANCES=1
    if not os.path.exists(p):
        return np.zeros((T, H, W), np.uint8)
    z = np.load(p)
    lab = z["lab"][:T] if "lab" in z.files else load_mask(p, T).astype(np.uint8)
    if len(lab) < T:
        lab = np.concatenate([lab, np.zeros((T - len(lab), H, W), np.uint8)])
    for i in [int(i) for i in np.unique(lab) if i > 0]:  # a "camera" track that is really the robot hand: drop it whole
        m = lab == i
        if (m & rob).sum() > 0.5 * m.sum():
            lab[m] = 0
    lab[rob] = 0
    return lab


def manipulated(lab, rob_d):
    """the ONE object that gets a control colour in this window (user, 10-29: only the manipulated object, one mask):
    the instance touching the (dilated) robot in the most frames, at least 5; everything else stays scene, as at
    inference where only the simulated object is drawn. (Image motion is no test here: the G1 head moves.)"""
    best, n_best = None, 4
    for i in [int(i) for i in np.unique(lab) if i > 0]:
        m = lab == i
        n = int((m & rob_d).reshape(len(m), -1).any(1).sum())
        if n > n_best:
            best, n_best = i, n
    return [] if best is None else [best]


def make_aug():
    """one random look per window: (hue shift, sat scale, val scale, texture or None)"""
    if rng.random() >= 0.8:
        return None
    tex = None
    if rng.random() < 0.5:  # luminance pattern (blotches + stripes, like wood / stone / fabric) with a mild colour tint
        g = cv2.resize(rng.uniform(0.7, 1.3, (rng.integers(3, 9), rng.integers(4, 12))).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        th, fr = rng.uniform(0, np.pi), rng.uniform(0.02, 0.15)
        stripes = 1 + rng.uniform(0, 0.2) * np.sin(fr * (xx * np.cos(th) + yy * np.sin(th)) + rng.uniform(0, 6.3))
        tint = rng.uniform(0.6, 1.4, 3).astype(np.float32)
        tint /= tint.mean()  # coloured surface (white table -> wood / paint colours), one pattern for all channels
        tex = np.clip((g * stripes)[..., None] * tint, 0.5, 1.6)
    return dict(hue=float(rng.uniform(-30, 30)), sat=float(rng.uniform(0.5, 1.5)), val=float(rng.uniform(0.7, 1.3)), tex=tex)


def apply_aug(frames, robot, aug):
    """same look on every frame, only off the (dilated, feathered) robot"""
    if aug is None:
        return frames
    out = np.empty_like(frames)
    for k, (f, r) in enumerate(zip(frames, robot)):
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + aug["hue"]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * aug["sat"], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * aug["val"], 0, 255)
        g = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        if aug["tex"] is not None:
            g = np.clip(g * aug["tex"], 0, 255)
        keep = cv2.GaussianBlur((cv2.dilate(r.astype(np.uint8), K_ROB) > 0).astype(np.float32), (9, 9), 3)[..., None]
        out[k] = (keep * f + (1 - keep) * g).astype(np.uint8)
    return out


def enc(frames):
    with torch.no_grad():
        return pipe.encode_video([Image.fromarray(f) for f in frames])[0].cpu().half()


def load_mask(path, T):
    if not os.path.exists(path):
        return np.zeros((T, H, W), bool)
    z = np.load(path)
    m = np.unpackbits(z["mask"], axis=-1)[..., :W].astype(bool)
    return m[:T] if len(m) >= T else np.concatenate([m, np.zeros((T - len(m), H, W), bool)])


import json  # noqa: E402
segs = json.load(open(f"{a.root}/segments.json"))[a.rank::a.world]
legchk = json.load(open(f"{a.root}/leg_check.json")) if os.path.exists(f"{a.root}/leg_check.json") else {}
for sg in segs:
    c = sg["id"]
    dst = f"{a.out}/{c}.pt"
    if os.path.exists(dst) or legchk.get(c, {}).get("leg", False):
        continue
    if not os.path.exists(f"{a.root}/robot/{c}.npz") or not os.path.exists(f"{a.root}/segments/{c}.mp4"):
        continue
    rgb = np.stack([fr for fr in imageio.get_reader(f"{a.root}/segments/{c}.mp4")])
    T = len(rgb)
    rob = load_mask(f"{a.root}/robot/{c}.npz", T)
    ok = np.array(sg["leg_free"][:T], bool)
    if "free" in legchk.get(c, {}):  # SAM3 per-frame leg flags (leg_check.py): source windows stay off leg frames too
        ok &= np.array(legchk[c]["free"][:T], bool)
    # segment starts at the episode start: the video file can be +-1 frame off, so frame 0 may be the previous episode's
    # last frame; only when there really is a cut between frame 0 and 1 is frame 0 kept out (and the window moved by one
    # so its reference frame is not frame 0). (Marking frame 0 bad unconditionally dropped all 393 episode-start windows.)
    cut0 = sg["a15"] == 0 and np.abs(rgb[1].astype(np.int16) - rgb[0]).mean() > 25
    if cut0:
        ok[0] = False
    rob_d = np.stack([cv2.dilate(m.astype(np.uint8), K_ROB) > 0 for m in rob])
    rob_h = rob_d if a.hole_px == 5 else np.stack([cv2.dilate(m.astype(np.uint8), K_HOLE) > 0 for m in rob])  # the hole around the robot
    area = rob.reshape(T, -1).mean(1)
    if np.median(area) < 0.015:  # the wide G1 head lens shows the arms small (median 6 %)
        print(f"{c}: robot mask median area {np.median(area) * 100:.1f}%, skipped", flush=True); continue
    s = sg["win_start"] + int(cut0 and sg["win_start"] == 1); seg = slice(s, s + a.win)
    if s < 1 or s + a.win > T or not ok[s - 1:s + a.win].all() or track_dip(area[s - 1:s + a.win]):
        print(f"{c}: window not usable", flush=True); continue
    aug = make_aug()
    lab = load_instances(c, T, rob)
    if a.obj_dir == "obj1":  # Claude already picked the handled object: take it wherever it is visible in >= 3 window frames
        ids = [1] if (lab[seg] == 1).reshape(a.win, -1).any(1).sum() >= 3 else []
    else:
        ids = manipulated(lab[seg], rob_d[seg])
    cols = palette(len(ids))
    ctrl = np.full((a.win, H, W, 3), GRAY, np.uint8)
    drop_obj = bool(a.depth and ids and a.mask_drop and rng.random() < a.mask_drop)  # v6b: control A without the object, depth alone has to say where it is
    for i, col in zip(ids, cols):
        if not drop_obj:
            ctrl[lab[seg] == i] = col
    obj_enc = "flat"
    dp = f"{a.root}/obj_depth/{c}.npz"
    if a.obj_enc == "c1" and ids and os.path.exists(dp):  # object drawn with its depth: hue = distance, shading = shape
        inv = np.load(dp)["inv"][:T].astype(np.float32)
        if len(inv) >= s + a.win:
            objw = np.isin(lab[seg], ids) & ~rob[seg]
            OD.paint(ctrl, inv[seg], objw, rob[seg]); obj_enc = "c1"
    ctrl = np.where(rob[seg][..., None], rgb[seg], ctrl).astype(np.uint8)  # robot (real cut-out) on top
    tgt = apply_aug(rgb[seg], rob[seg], aug)
    ref_img = apply_aug(rgb[s - 1:s], rob[s - 1:s], aug)[0]
    shifts = [d for d in range(-a.dmax, a.dmax + 1) if abs(d) >= a.dmin and 0 <= s + d and s + d + a.win <= T and ok[s + d:s + d + a.win].all()]
    if not shifts:
        shifts = [d for d in range(-a.dmax, a.dmax + 1) if abs(d) >= 3 and 0 <= s + d and s + d + a.win <= T and ok[s + d:s + d + a.win].all()]
    if not shifts:
        print(f"{c}: no leg-free source window", flush=True); continue
    d = int(rng.choice(shifts))
    if rng.random() < a.aligned:  # 10-30: the source at the SAME time (as the human clip at inference), robot holed: copy what is known
        d = 0
    ss = slice(s + d, s + d + a.win)
    hole = rob_h[ss] | rob_h[seg]
    flags = ["aligned"] if d == 0 else []
    if drop_obj:
        flags.append("nomask")
    gh = None
    if a.ghost and rng.random() < a.ghost:  # v6: where a human hand at roughly the robot's place would also have been holed
        gh = GH.ghost(rob[seg], rng); hole |= gh; flags.append("ghost")
    if rng.random() < a.ring:
        inside, mx, my, Wv, Hv = human_footprint(); hole |= ~inside[None]; flags.append("ring")
        if rng.random() < 0.7:
            hv = hand_holes(a.win, Wv, Hv).astype(np.uint8)
            hole |= np.stack([cv2.remap(m, mx, my, cv2.INTER_NEAREST, borderValue=0) > 0 for m in hv]) & inside[None]; flags.append("hand")
    elif a.hand_stamps and rng.random() < a.hand_stamps:
        hole |= hand_holes(a.win); flags.append("hand")
    if ids and rng.random() < a.obj_hole:  # the handled object greyed at the source time: its place must come from the control
        objm = np.isin(lab[ss], ids)
        hole |= np.stack([cv2.dilate(m.astype(np.uint8), K_OBJ) > 0 for m in objm]); flags.append("obj")
    if ids and a.obj_stick and rng.random() < a.obj_stick:  # v5: glued to the object for the whole window, like the human thumbs on the tissue box
        hole |= stick_holes(np.isin(lab[ss], ids)); flags.append("stick")
    b = 0
    if rng.random() < a.band:  # letterbox: 16:9 uncropped in 4:3 = 42 px (12.5 %) top and bottom (60 %), else 3-13 %
        b = 42 if rng.random() < 0.6 else int(round(rng.uniform(0.03, 0.13) * H)); hole[:, :b] = True; hole[:, H - b:] = True; flags.append("band")
    src = np.where(hole[..., None], GRAY, apply_aug(rgb[ss], rob[ss], aug)).astype(np.uint8)
    with torch.no_grad():
        ref = pipe.vae.encode(video_to_tensor([Image.fromarray(ref_img)], pipe.dtype, pipe.device), device=pipe.device)[0].cpu().half()
        clip = pipe.image_encoder.encode_image([image_to_tensor(Image.fromarray(ref_img), pipe.dtype, pipe.device)])[0].cpu().half()
    srcs = [dict(lat=enc(src), hole=torch.from_numpy(cells(hole, a.mask_pool)).half()[None], shift=d, flags=flags)]
    yd, dep = None, None
    if a.depth:  # control B: whole-frame depth of the target window
        fp, dp2 = f"{a.root}/full_depth/{c}.npz", f"{a.root}/obj_depth/{c}.npz"
        if os.path.exists(fp):
            finv = FE.load_full(fp, T)
            od = np.load(dp2)["inv"][:T].astype(np.float32) if (ids and os.path.exists(dp2)) else None
            if len(finv) >= s + a.win and (od is None or len(od) >= s + a.win):
                objw = np.isin(lab[seg], ids) & ~rob[seg] if ids else np.zeros((a.win, H, W), bool)
                dep = FE.training_window(finv[seg], None if od is None else od[seg], objw, rob[seg], rng, gh)
                yd = enc(dep)
    info, info_t, info_img = None, pick_info(T, ok, s, rob_h, hole, a.info) if a.info else [], []
    for t in info_t:  # scene frames: same look as the target, robot holed (a human clip's info frames have the hands holed), same bands as the source
        m = rob_h[t].copy()
        if b:
            m[:b] = True; m[H - b:] = True
        if a.hand_stamps and rng.random() < 0.5:
            m |= hand_holes(1)[0]
        info_img.append(np.where(m[..., None], GRAY, apply_aug(rgb[t:t + 1], rob[t:t + 1], aug)[0]).astype(np.uint8))
    if info_img:
        with torch.no_grad():
            info = torch.stack([pipe.vae.encode(video_to_tensor([Image.fromarray(im)], pipe.dtype, pipe.device), device=pipe.device)[0][:, 0].cpu().half() for im in info_img])
    occ = torch.from_numpy(np.stack([cells(lab[seg] == i, "mean") for i in ids]) if ids else np.zeros((0, (a.win - 1) // 4 + 1, H // 8, W // 8), np.float32)).half()
    wins = [dict(z0=enc(tgt), y=enc(ctrl), ref=ref, clip=clip, srcs=srcs, occ=occ, start=s, n_obj=len(ids), colours=[c_.tolist() for c_ in cols], obj_enc=obj_enc, info=info, info_t=info_t, yd=yd,
                 aug=None if aug is None else {k: v for k, v in aug.items() if k != "tex"} | {"tex": aug["tex"] is not None})]
    k = a.win // 2
    qa = np.hstack([ctrl[k], src[k], tgt[k]] + ([dep[k]] if dep is not None else []))
    if info_img:  # second row: the info frames
        row = np.hstack(info_img)
        qa = np.vstack([qa, cv2.resize(row, (qa.shape[1], int(row.shape[0] * qa.shape[1] / row.shape[1])))])
    imageio.imwrite(f"{a.out}/qa/{c}.jpg", qa)
    torch.save(dict(clip=c, dataset=sg["dataset"], episode=sg["episode"], windows=wins), dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    print(f"{c}: cached (shift {d})", flush=True)
print("CACHE_DONE")
