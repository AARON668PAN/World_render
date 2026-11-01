"""Bring the reconstructed human hands to the depth the video shows, without moving them in the image.

1. Metric video depth: Video Depth Anything relative disparity, one affine fit per frame (1/z = a*d + b) on two
   references with known depth: box surface pixels (mesh + pose + camera, z-buffered, SAM3 box mask minus hands) and
   table pixels (SAM3 table mask minus box and hands; depth from the ray hitting the table plane, the plane through
   the box's lowest point at rest). The table spans 0.4-1.5 m, so the fit also holds away from the box.
2. Hand depth from the video: per frame and hand, the MANO joints that project inside the SAM3 hand/forearm mask,
   video depth there (median of a 5x5 patch inside the mask) plus the joint's distance below the skin.
3. Correction: per frame and hand a factor k = median(video depth / MANO depth); MANO joints are scaled about the
   camera centre by k, which keeps every joint on its pixel (projection unchanged) and only changes depth. k is
   smoothed in time (median filter over K_WIN frames, default 31).

usage: /data/repos/sam3/.venv/bin/python align_hand_depth.py <recon_geom.npz> <ego_camera_refined.npz> <box_masks.npz>
       <human_table_masks.npz> <vda_rel dir> <out dir>
writes <out>/video_depth_fit.npy (T, 2), <out>/hand_depth_k.npz (k_raw, k (T, 2) right/left, corrected mano (T, 42, 3))
"""

import os
import sys

import numpy as np
from scipy.ndimage import binary_erosion, median_filter
from scipy.spatial.transform import Rotation as R

recon_npz, cam_npz, box_npz, ht_npz, vda_dir, out = sys.argv[1:7]
os.makedirs(out, exist_ok=True)
K_WIN = int(os.environ.get("K_WIN", 31))
FIT_ON = os.environ.get("FIT_ON", "both")  # which references set the video depth scale: both | table | box
geom = np.load(recon_npz)
mano, obj_p, obj_q, surf = geom["mano"], geom["obj_p"], geom["obj_q"], geom["surf"]
cams = np.load(cam_npz)
c2w, F = cams["cam_to_world"], float(cams["focal"])
disp = np.load(f"{vda_dir}/depth.npz")["raw"]
T, H, W = disp.shape


def unpack(z, key, w):
    return np.unpackbits(z[key], axis=-1)[..., :w].astype(bool)


box_m = unpack(np.load(box_npz), "masks", W)
ht = np.load(ht_npz)
human = unpack(ht, "hand", W) | unpack(ht, "forearm", W)
table_m = unpack(ht, "table", W)
rest = int(np.argmin(obj_p[:, 2]))
table_z = (R.from_quat(obj_q[rest], scalar_first=True).apply(surf) + obj_p[rest])[:, 2].min()
print(f"table plane z = {table_z:.3f} m (box lowest point at rest, frame {rest})")
# joint depth below the visible skin: wrist, knuckles, other finger joints
R_SKIN = np.full(21, 0.008)
R_SKIN[0] = 0.02
R_SKIN[[1, 5, 9, 13, 17]] = 0.012
R_SKIN = np.concatenate([R_SKIN, R_SKIN])
rng = np.random.default_rng(0)
vv, uu = np.mgrid[0:H, 0:W]

fit = np.full((T, 2), np.nan)
res_box, res_table, res_hand, k_raw = [], [], [], np.full((T, 2), np.nan)
for t in range(T):
    w2c = np.linalg.inv(c2w[t])
    # box surface depth (z-buffer) on box-mask pixels that are not hand
    c = (R.from_quat(obj_q[t], scalar_first=True).apply(surf) + obj_p[t]) @ w2c[:3, :3].T + w2c[:3, 3]
    u, v = F * c[:, 0] / c[:, 2] + W / 2, F * c[:, 1] / c[:, 2] + H / 2
    ok = (c[:, 2] > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zbuf = np.full((H, W), np.inf)
    np.minimum.at(zbuf, (v[ok].astype(int), u[ok].astype(int)), c[ok, 2])
    sel_b = np.isfinite(zbuf) & binary_erosion(box_m[t] & ~human[t], iterations=3)
    bv, bu = np.nonzero(sel_b)
    if len(bv) > 3000:
        k = rng.choice(len(bv), 3000, replace=False)
        bv, bu = bv[k], bu[k]
    zb, db = zbuf[bv, bu], disp[t, bv, bu].astype(np.float64)
    # table depth: ray / plane intersection on table pixels that are neither box nor hand
    sel_t = binary_erosion(table_m[t] & ~box_m[t] & ~human[t], iterations=5)
    tv, tu = np.nonzero(sel_t)
    if len(tv) > 3000:
        k = rng.choice(len(tv), 3000, replace=False)
        tv, tu = tv[k], tu[k]
    rays = np.stack([(tu - W / 2) / F, (tv - H / 2) / F, np.ones(len(tu))], 1) @ c2w[t, :3, :3].T
    s = (table_z - c2w[t, 2, 3]) / rays[:, 2]
    good = (s > 0.1) & (s < 3.0)  # s is the camera-frame z because the camera-frame ray has z = 1
    zt, dt = s[good], disp[t, tv[good], tu[good]].astype(np.float64)
    if len(zb) < 100 or len(zt) < 100:
        continue
    A = np.concatenate([np.stack([db, np.ones_like(db)], 1), np.stack([dt, np.ones_like(dt)], 1)])
    y = np.concatenate([1 / zb, 1 / zt])
    wgt = np.concatenate([np.full(len(zb), 0.0 if FIT_ON == "table" else 1 / len(zb)), np.full(len(zt), 0.0 if FIT_ON == "box" else 1 / len(zt))])
    keep = np.ones(len(y), bool)
    for _ in range(3):
        sw = np.sqrt(wgt[keep])
        coef = np.linalg.lstsq(A[keep] * sw[:, None], y[keep] * sw, rcond=None)[0]
        r = np.abs(A @ coef - y)
        keep = (r < 3 * np.median(r[wgt > 0])) & (wgt > 0)
    fit[t] = coef
    inv = lambda d: 1 / np.maximum(coef[0] * d + coef[1], 1e-3)
    res_box.append(np.median(np.abs(inv(db) - zb)))
    res_table.append(np.median(np.abs(inv(dt) - zt)))
    # hands
    cj = mano[t] @ w2c[:3, :3].T + w2c[:3, 3]
    ju, jv = (F * cj[:, 0] / cj[:, 2] + W / 2).round().astype(int), (F * cj[:, 1] / cj[:, 2] + H / 2).round().astype(int)
    for h, sl in enumerate((slice(0, 21), slice(21, 42))):
        ratios = []
        for j in range(sl.start, sl.stop):
            if not (2 <= ju[j] < W - 2 and 2 <= jv[j] < H - 2):
                continue
            patch_m = human[t, jv[j] - 2:jv[j] + 3, ju[j] - 2:ju[j] + 3]
            if patch_m.sum() < 13:  # joint not clearly on the hand in the image (occluded or off the silhouette)
                continue
            zv = inv(np.median(disp[t, jv[j] - 2:jv[j] + 3, ju[j] - 2:ju[j] + 3][patch_m].astype(np.float64)))
            ratios.append((zv + R_SKIN[j]) / cj[j, 2])
            res_hand.append(zv + R_SKIN[j] - cj[j, 2])
        if len(ratios) >= 5:
            k_raw[t, h] = np.median(ratios)

res_hand = np.array(res_hand)
print(f"video depth fit on {FIT_ON}, {np.isfinite(fit[:, 0]).sum()}/{T} frames: box residual median "
      f"{np.median(res_box)*100:.1f} cm, table residual median {np.median(res_table)*100:.1f} cm")
print(f"MANO joints vs video depth (visible joints, before correction): signed median {np.median(res_hand)*100:+.1f} cm "
      f"(+ = video farther), |err| median {np.median(np.abs(res_hand))*100:.1f} cm, 90% {np.percentile(np.abs(res_hand), 90)*100:.1f} cm")
k = np.empty_like(k_raw)
for h, name in enumerate(("right", "left")):
    kr = k_raw[:, h]
    ok = np.isfinite(kr)
    filled = np.interp(np.arange(T), np.nonzero(ok)[0], kr[ok])
    k[:, h] = median_filter(filled, size=K_WIN, mode="nearest")
    print(f"{name} hand: k from {ok.sum()} frames, median {np.median(kr[ok]):.3f}, 10-90% {np.percentile(kr[ok], 10):.3f}-"
          f"{np.percentile(kr[ok], 90):.3f}; smoothed k range {k[:, h].min():.3f}-{k[:, h].max():.3f}")
corrected = mano.copy()
for t in range(T):
    C = c2w[t, :3, 3]
    corrected[t, :21] = C + k[t, 0] * (mano[t, :21] - C)
    corrected[t, 21:] = C + k[t, 1] * (mano[t, 21:] - C)
np.save(f"{out}/video_depth_fit.npy", fit)
np.savez(f"{out}/hand_depth_k.npz", k_raw=k_raw, k=k, mano=corrected, table_z=table_z)
print("wrote", out)
