"""The ego video's camera does not move (the wall strip shifts < 0.5 px over the whole clip), but the per-frame camera
fit (refine_ego_camera.py) wanders 7 cm / 8 deg because it absorbs the hand / box reconstruction errors. This makes
the camera static and moves those errors into the hands and the box instead:

  * one camera pose c2w_s for all frames (mean of the fitted poses over the first resting segment of the object);
  * every frame's hands and object are expressed in that frame's fitted camera and re-placed in front of c2w_s
    (M_t = c2w_s * inv(c2w_t)), so they project exactly as before (image alignment unchanged) and keep their mutual
    pose (contacts unchanged);
  * the object is held still while it rests on the table (one pose per resting segment), the object and hands are
    smoothed in time (Gaussian, BOX_SIGMA / HAND_SIGMA frames).

usage: make_static_camera.py <ego parquet dir> <ego_camera_refined.npz> <out dir>
writes <out>/staged/ (same layout, world-space columns transformed) and <out>/ego_camera_static.npz
"""

import glob
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

parquet_dir, cam_npz, out = sys.argv[1:4]
BOX_SIGMA = float(os.environ.get("BOX_SIGMA", 2.0))
HAND_SIGMA = float(os.environ.get("HAND_SIGMA", 1.5))
REST_CM = float(os.environ.get("REST_CM", 2.0))
pfile = glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True)[0]
table = pq.read_table(pfile)
col = lambda n: table.column(n)[0].as_py()
cams = np.load(cam_npz)
c2w = cams["cam_to_world"]
T = len(c2w)

# resting segments of the object: within REST_CM of its lowest height and slow
obj_p = np.asarray(col("object_body_position"))[:, 0]
obj_q = np.asarray(col("object_body_wxyz"))[:, 0]
height = obj_p[:, 2] - obj_p[:, 2].min()
speed = np.r_[0, np.linalg.norm(np.diff(obj_p, axis=0), axis=1)] * 30  # m/s
rest = (height < REST_CM / 100) & (gaussian_filter1d(speed, 2) < 0.05)
segments, t = [], 0
while t < T:
    if rest[t]:
        s = t
        while t < T and rest[t]:
            t += 1
        if t - s >= 10:
            segments.append((s, t))
    t += 1
print("object resting segments:", segments)
s0, e0 = segments[0] if segments else (0, min(T, 30))
c2w_s = np.eye(4)
c2w_s[:3, :3] = R.from_matrix(c2w[s0:e0, :3, :3]).mean().as_matrix()
c2w_s[:3, 3] = c2w[s0:e0, :3, 3].mean(0)
M = np.einsum("ij,tjk->tik", c2w_s, np.linalg.inv(c2w))  # (T, 4, 4) old world -> new world, per frame
Mr = R.from_matrix(M[:, :3, :3])


def tf_points(x):  # (T, ..., 3)
    x = np.asarray(x, float)
    return np.einsum("tij,t...j->t...i", M[:, :3, :3], x) + M[:, None, :3, 3].reshape((T,) + (1,) * (x.ndim - 2) + (3,))


def tf_dirs(x):
    return np.einsum("tij,t...j->t...i", M[:, :3, :3], np.asarray(x, float))


def tf_wxyz(x):  # (T, K, 4) wxyz
    x = np.asarray(x, float)
    sh = x.shape
    q = R.from_quat(x.reshape(-1, 4), scalar_first=True)
    rep = np.repeat(np.arange(T), int(np.prod(sh[1:-1])))
    return (Mr[rep] * q).as_quat(scalar_first=True).reshape(sh)


def smooth(x, sigma):
    return gaussian_filter1d(np.asarray(x, float), sigma, axis=0, mode="nearest") if sigma > 0 else x


def smooth_quat(q, sigma):  # (T, ..., 4) wxyz, sign-aligned then averaged and renormalised
    q = np.asarray(q, float).copy()
    for t in range(1, len(q)):
        flip = np.sum(q[t] * q[t - 1], axis=-1) < 0
        q[t][flip] *= -1
    q = smooth(q, sigma)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


old_wrist = {s: np.asarray(col(f"mano_{s}_joints"))[:, 0] for s in ("right", "left")}
new = {}
for side in ("right", "left"):
    j = tf_points(col(f"mano_{side}_joints"))
    new[f"mano_{side}_joints"] = smooth(j, HAND_SIGMA)
    new[f"mano_{side}_joints_wxyz"] = smooth_quat(tf_wxyz(col(f"mano_{side}_joints_wxyz")), HAND_SIGMA)
    for kind in ("link", "object"):
        new[f"mano_{side}_{kind}_contact_positions"] = smooth(tf_points(col(f"mano_{side}_{kind}_contact_positions")), HAND_SIGMA)
        new[f"mano_{side}_{kind}_contact_normals"] = tf_dirs(col(f"mano_{side}_{kind}_contact_normals"))
    # global orientation follows the frame rotation; trans is left as the joint-0 shift (only used for frame count)
    go = R.from_rotvec(np.asarray(col(f"mano_{side}_global_orient")))
    new[f"mano_{side}_global_orient"] = (Mr * go).as_rotvec()
    j0_old = np.asarray(col(f"mano_{side}_joints"))[:, 0]
    new[f"mano_{side}_trans"] = np.asarray(col(f"mano_{side}_trans")) + (new[f"mano_{side}_joints"][:, 0] - j0_old)

bp = smooth(tf_points(obj_p[:, None])[:, 0], BOX_SIGMA)
bq = smooth_quat(tf_wxyz(obj_q[:, None])[:, 0], BOX_SIGMA)
for s, e in segments:  # still while resting, blended over 5 frames at the edges
    p_rest = np.median(bp[s:e], axis=0)
    q_rest = R.from_quat(bq[s:e], scalar_first=True).mean()
    for t in range(max(0, s - 5), min(T, e + 5)):
        w = 1.0 if s <= t < e else 1.0 - min(abs(t - s), abs(t - (e - 1))) / 5.0
        bp[t] = w * p_rest + (1 - w) * bp[t]
        bq[t] = (R.from_quat(bq[t], scalar_first=True) * R.from_rotvec(
            w * (R.from_quat(bq[t], scalar_first=True).inv() * q_rest).as_rotvec())).as_quat(scalar_first=True)
new["object_body_position"] = bp[:, None]
new["object_body_wxyz"] = bq[:, None]
new["object_root_position"] = bp
new["object_root_axis_angle"] = R.from_quat(bq, scalar_first=True).as_rotvec()

for name, value in new.items():
    idx = table.column_names.index(name)
    table = table.set_column(idx, name, pa.array([np.asarray(value).tolist()], type=table.schema.field(name).type))
dst = f"{out}/staged/" + os.path.relpath(os.path.dirname(pfile), parquet_dir)
os.makedirs(dst, exist_ok=True)
pq.write_table(table, f"{dst}/{os.path.basename(pfile)}")
np.savez(f"{out}/ego_camera_static.npz", cam_to_world=np.repeat(c2w_s[None], T, 0), focal=cams["focal"],
         width=cams["width"], height=cams["height"], box_err=np.zeros(T))

# report: how much the hands / object now move compared with before (old world), and the jitter left
hf = lambda x: x - gaussian_filter1d(x, 2.0, axis=0, mode="nearest")
for name, old, nw in (("object", obj_p, bp), ("right wrist", old_wrist["right"], new["mano_right_joints"][:, 0]),
                      ("left wrist", old_wrist["left"], new["mano_left_joints"][:, 0])):
    print(f"{name:12s}: jitter RMS {np.sqrt((hf(old) ** 2).sum(1).mean()) * 1000:.1f} -> {np.sqrt((hf(nw) ** 2).sum(1).mean()) * 1000:.1f} mm; "
          f"moved by the static-camera transfer: median {np.median(np.linalg.norm(nw - old, axis=1)) * 100:.1f} cm, max {np.linalg.norm(nw - old, axis=1).max() * 100:.1f} cm")
print("wrote", dst, "and", f"{out}/ego_camera_static.npz")
