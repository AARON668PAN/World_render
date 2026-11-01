"""Per-frame camera pose of the monocular ego video in the reconstruction world, from PnP between the
reconstructed MANO joints (world frame, ego_recon parquet) and MediaPipe 2D landmarks (same 21-joint order).
The focal length is searched first (principal point at the image centre), then every frame is solved and the
trajectory smoothed. Writes camera-to-world poses (OpenCV camera axes) and an overlay video for checking.

usage: estimate_ego_camera.py <ego parquet dir> <hands2d.json> <video.mp4> <out dir>
"""

import glob
import json
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

parquet_dir, hands_json, video, out_dir = sys.argv[1:5]
table = pq.read_table(glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True)[0])
col = lambda n: table.column(n)[0].as_py()
mano = {s: np.asarray(col(f"mano_{s}_joints")) for s in ("right", "left")}  # (T, 21, 3) world
obj_p, obj_q = np.asarray(col("object_body_position"))[:, 0], np.asarray(col("object_body_wxyz"))[:, 0]
det = json.load(open(hands_json))
W, H, frames = det["width"], det["height"], det["frames"]
T = len(frames)
box = trimesh.load(col("object_mesh_paths")[0].replace("/workspace/video_to_data", os.environ.get("CHORD", "/data/repos/chord")), force="mesh")
box_pts = np.asarray(box.sample(400, return_index=False))


def K(f):
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def correspondences(t):
    """Both left/right assignments of the detected hands to the MANO hands."""
    found = frames[t]
    if not found:
        return []
    options = []
    sides = ("right", "left")
    if len(found) == 2:
        for a, b in ((0, 1), (1, 0)):
            options.append((np.concatenate([mano[sides[0]][t], mano[sides[1]][t]]),
                            np.concatenate([found[a]["uv"], found[b]["uv"]])))
    else:
        for s in sides:
            options.append((mano[s][t], np.asarray(found[0]["uv"])))
    return options


def solve(t, f, guess=None):
    best = None
    for obj, img in correspondences(t):
        obj, img = np.asarray(obj, np.float64), np.asarray(img, np.float64)
        if guess is not None:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K(f), None, rvec=guess[0].copy(), tvec=guess[1].copy(),
                                          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(obj, img, K(f), None, reprojectionError=20.0, iterationsCount=200)
            if ok:
                rvec, tvec = cv2.solvePnPRefineLM(obj, img, K(f), None, rvec, tvec)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K(f), None)
        err = np.median(np.linalg.norm(proj[:, 0] - img, axis=1))
        if best is None or err < best[0]:
            best = (err, rvec, tvec)
    return best


# 1. Focal length: FOCAL env fixes it (hands alone barely constrain it: depth and focal trade off);
#    otherwise search on a subset of frames.
import os
sample = [t for t in range(0, T, 8) if len(frames[t]) == 2]
def median_err(f):
    errs = [s[0] for t in sample if (s := solve(t, f)) is not None]
    return np.median(errs) if errs else 1e9
if os.environ.get("FOCAL"):
    f_best = float(os.environ["FOCAL"])
    print(f"focal fixed at {f_best} px (median reprojection {median_err(f_best):.1f} px)")
else:
    grid = np.arange(500, 1600, 50)
    coarse = [median_err(f) for f in grid]
    f0 = grid[int(np.argmin(coarse))]
    fine = np.arange(f0 - 50, f0 + 51, 10)
    f_best = fine[int(np.argmin([median_err(f) for f in fine]))]
    print(f"focal search: coarse best {f0}, fine best {f_best} px (median reprojection {median_err(f_best):.1f} px)")
    print("  coarse curve:", {int(f): round(float(e), 1) for f, e in zip(grid, coarse)})

# 2. Every frame, warm-started from the previous solution.
poses, errs, prev = [], [], None
for t in range(T):
    s = solve(t, f_best, prev) if prev is not None else solve(t, f_best)
    if s is None or s[0] > 25:
        s2 = solve(t, f_best)
        s = s2 if s2 is not None and (s is None or s2[0] < s[0]) else s
    if s is None:
        poses.append(None); errs.append(np.nan)
        continue
    err, rvec, tvec = s
    prev = (rvec, tvec)
    rot = R.from_rotvec(rvec[:, 0]).as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = rot.T
    c2w[:3, 3] = -rot.T @ tvec[:, 0]
    poses.append(c2w); errs.append(err)
errs = np.asarray(errs)
valid = np.array([p is not None for p in poses])
print(f"solved {valid.sum()}/{T} frames; reprojection median {np.nanmedian(errs):.1f} px, 90% {np.nanpercentile(errs, 90):.1f} px")
raw_pos = np.array([p[:3, 3] for p in poses if p is not None])
jumps = np.linalg.norm(np.diff(raw_pos, axis=0), axis=1)
print(f"raw camera position: z {raw_pos[:, 2].min():.2f}-{raw_pos[:, 2].max():.2f} m, frame-to-frame jump median {np.median(jumps)*100:.1f} cm, max {jumps.max()*100:.1f} cm")

# 3. Smooth: positions with a moving average, rotations with a windowed slerp mean.
idx = np.where(valid)[0]
pos = np.array([poses[t][:3, 3] for t in idx])
rots = R.from_matrix(np.array([poses[t][:3, :3] for t in idx]))
pos_all = np.stack([np.interp(np.arange(T), idx, pos[:, k]) for k in range(3)], 1)
rot_all = Slerp(idx, rots)(np.clip(np.arange(T), idx[0], idx[-1]))
win = 7
kernel = np.ones(win) / win
pos_s = np.stack([np.convolve(np.pad(pos_all[:, k], win // 2, mode="edge"), kernel, mode="valid") for k in range(3)], 1)
quats = rot_all.as_quat()
quats[np.sum(quats * quats[0], axis=1) < 0] *= -1
quats_s = np.stack([np.convolve(np.pad(quats[:, k], win // 2, mode="edge"), kernel, mode="valid") for k in range(4)], 1)
quats_s /= np.linalg.norm(quats_s, axis=1, keepdims=True)
c2w_s = np.tile(np.eye(4), (T, 1, 1))
c2w_s[:, :3, :3] = R.from_quat(quats_s).as_matrix()
c2w_s[:, :3, 3] = pos_s
np.savez(f"{out_dir}/ego_camera.npz", cam_to_world=c2w_s, focal=f_best, width=W, height=H, reproj_err=errs)
print("camera height above world z=0: %.2f-%.2f m; distance to box mean %.2f m" % (
    pos_s[:, 2].min(), pos_s[:, 2].max(), np.linalg.norm(pos_s - obj_p, axis=1).mean()))

# 4. Overlay video: MANO joints (green) and box surface samples (red) projected with the smoothed cameras.
cap = cv2.VideoCapture(video)
writer = cv2.VideoWriter(f"{out_dir}/ego_camera_overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for t in range(T):
    ok, img = cap.read()
    if not ok:
        break
    w2c = np.linalg.inv(c2w_s[t])
    rvec, tvec = cv2.Rodrigues(w2c[:3, :3])[0], w2c[:3, 3]
    box_w = R.from_quat(obj_q[t], scalar_first=True).apply(box_pts) + obj_p[t]
    for pts, color in ((box_w, (0, 0, 255)), (mano["right"][t], (0, 255, 0)), (mano["left"][t], (0, 255, 0))):
        proj, _ = cv2.projectPoints(np.asarray(pts, np.float64), rvec, tvec, K(f_best), None)
        for u, v in proj[:, 0]:
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(img, (int(u), int(v)), 2, color, -1)
    writer.write(img)
writer.release()
print("overlay:", f"{out_dir}/ego_camera_overlay.mp4")
