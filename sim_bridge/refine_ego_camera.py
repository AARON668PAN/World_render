"""Refine the ego camera trajectory with hands AND the object: per frame, minimise
  hand keypoint reprojection (MediaPipe 2D vs reconstructed MANO joints, Huber) +
  object silhouette chamfer (projected mesh samples vs SAM3 mask, both directions) +
  a small pull towards the previous frame's pose,
tracking forward and backward from the most reliable frame. Focal length fixed (FOCAL env, default 950 px).

usage: refine_ego_camera.py <ego parquet dir> <hands2d.json> <box_masks.npz> <video.mp4> <out dir>
"""

import glob
import json
import os
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation as R

parquet_dir, hands_json, masks_npz, video, out_dir = sys.argv[1:6]
F = float(os.environ.get("FOCAL", 950))
table = pq.read_table(glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True)[0])
col = lambda n: table.column(n)[0].as_py()
mano = {s: np.asarray(col(f"mano_{s}_joints")) for s in ("right", "left")}
obj_p, obj_q = np.asarray(col("object_body_position"))[:, 0], np.asarray(col("object_body_wxyz"))[:, 0]
det = json.load(open(hands_json))
W, H, frames = det["width"], det["height"], det["frames"]
T = len(frames)
m = np.load(masks_npz)
n, h, w = m["shape"]
box_masks = np.unpackbits(m["masks"], axis=-1)[..., :w].astype(bool)
S = 2  # work at half resolution for the silhouette term
dmaps = [ndimage.distance_transform_edt(~mk[::S, ::S]) for mk in box_masks]
mask_px = [np.argwhere(mk[::S, ::S])[:: max(1, int(mk[::S, ::S].sum()) // 800)] for mk in box_masks]
mesh = trimesh.load(col("object_mesh_paths")[0].replace("/workspace/video_to_data", os.environ.get("CHORD", "/data/repos/chord")), force="mesh")
box_local = np.asarray(mesh.sample(1500, return_index=False))
box_world = [R.from_quat(obj_q[t], scalar_first=True).apply(box_local) + obj_p[t] for t in range(T)]


def project(params, pts):
    rot = R.from_rotvec(params[:3]).as_matrix()
    cam = pts @ rot.T + params[3:]
    z = np.maximum(cam[:, 2], 1e-3)
    return np.stack([F * cam[:, 0] / z + W / 2, F * cam[:, 1] / z + H / 2], 1), cam[:, 2] > 0.05


def hand_terms(t, params):
    found = frames[t]
    if not found:
        return 0.0
    sides = ("right", "left")
    options = []
    if len(found) == 2:
        for a, b in ((0, 1), (1, 0)):
            options.append((np.concatenate([mano[sides[0]][t], mano[sides[1]][t]]),
                            np.concatenate([found[a]["uv"], found[b]["uv"]])))
    else:
        options += [(mano[s][t], np.asarray(found[0]["uv"])) for s in sides]
    best = np.inf
    for obj, img in options:
        uv, ok = project(params, obj)
        r = np.linalg.norm(uv - img, axis=1)
        huber = np.where(r < 10, 0.5 * r ** 2 / 10, r - 5)
        best = min(best, huber.mean())
    return best


def box_term(t, params):
    uv, ok = project(params, box_world[t])
    uv = uv[ok] / S
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w / S) & (uv[:, 1] >= 0) & (uv[:, 1] < h / S)
    if inside.sum() < 50:
        return 100.0
    ui, vi = uv[inside, 0].astype(int), uv[inside, 1].astype(int)
    d1 = dmaps[t][vi, ui].mean()
    proj = np.zeros((h // S, w // S), bool)
    proj[vi, ui] = True
    proj = ndimage.binary_dilation(proj, iterations=2)
    d2 = ndimage.distance_transform_edt(~proj)[mask_px[t][:, 0], mask_px[t][:, 1]].mean()
    return (d1 + d2) * S


def solve_frame(t, init, prior):
    def loss(p):
        return hand_terms(t, p) + box_term(t, p) + 200.0 * np.sum((p - prior) ** 2)
    res = optimize.minimize(loss, init, method="Powell", options={"maxiter": 3000, "xtol": 1e-4, "ftol": 1e-3})
    return res.x, res.fun


init = np.load(f"{out_dir}/ego_camera.npz")
c2w0 = init["cam_to_world"]
def to_params(c2w):
    w2c = np.linalg.inv(c2w)
    return np.concatenate([R.from_matrix(w2c[:3, :3]).as_rotvec(), w2c[:3, 3]])

start = int(np.nanargmin(np.where([len(f) == 2 for f in frames], init["reproj_err"], np.nan)))
params = np.zeros((T, 6))
losses = np.zeros(T)
p0, _ = solve_frame(start, to_params(c2w0[start]), to_params(c2w0[start]))
p0, losses[start] = solve_frame(start, p0, p0)
params[start] = p0
for direction in (1, -1):
    prev = p0
    t = start + direction
    while 0 <= t < T:
        params[t], losses[t] = solve_frame(t, prev, prev)
        prev = params[t]
        if t % 50 == 0:
            print(f"frame {t}: loss {losses[t]:.1f}", flush=True)
        t += direction

hand_err, box_err = [], []
for t in range(T):
    box_err.append(box_term(t, params[t]))
    found = frames[t]
    hand_err.append(np.sqrt(2 * 10 * hand_terms(t, params[t])) if found else np.nan)
c2w = np.tile(np.eye(4), (T, 1, 1))
for t in range(T):
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = R.from_rotvec(params[t, :3]).as_matrix(), params[t, 3:]
    c2w[t] = np.linalg.inv(w2c)
pos = c2w[:, :3, 3]
jumps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
print(f"start frame {start}; box silhouette chamfer median {np.median(box_err):.1f} px, hand error (approx) median {np.nanmedian(hand_err):.1f} px")
print(f"camera z {pos[:, 2].min():.2f}-{pos[:, 2].max():.2f} m, jump max {jumps.max()*100:.1f} cm, distance to box mean {np.linalg.norm(pos - obj_p, axis=1).mean():.2f} m")
np.savez(f"{out_dir}/ego_camera_refined.npz", cam_to_world=c2w, focal=F, width=W, height=H, box_err=np.array(box_err))

cap = cv2.VideoCapture(video)
writer = cv2.VideoWriter(f"{out_dir}/ego_camera_refined_overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
edges = mesh.edges_unique
for t in range(T):
    ok, img = cap.read()
    if not ok:
        break
    verts = R.from_quat(obj_q[t], scalar_first=True).apply(mesh.vertices) + obj_p[t]
    uv, _ = project(params[t], verts)
    hull = cv2.convexHull(uv.astype(np.float32))
    cv2.polylines(img, [hull.astype(np.int32)], True, (0, 0, 255), 2)
    for s in ("right", "left"):
        uvh, _ = project(params[t], mano[s][t])
        for u, v in uvh:
            cv2.circle(img, (int(u), int(v)), 3, (0, 255, 0), -1)
    writer.write(img)
writer.release()
