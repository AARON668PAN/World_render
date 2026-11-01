"""How close do the robot fingers get to the box? For a camtrack dir (ik_camera_track.py output): per finger (thumb ..
little / Dex3 thumb, index, middle), the smallest signed distance from the finger's mesh surface to the scaled box
surface (negative = inside), over the frames where the box is held (lifted > 1 cm above its resting height); the human
MANO tips are measured the same way (joint centres, minus a 0.75 * 8 mm finger radius) from the hand parquet.
usage: .venv-retarget/bin/python hand_box_gap.py <camtrack dir> <urdf> <hand parquet dir> [every n-th frame=3]"""
import glob
import re
import sys

import numpy as np
import pinocchio as pin
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

d, urdf, hand_dir = sys.argv[1:4]
step = int(sys.argv[4]) if len(sys.argv) > 4 else 3
tab = pq.read_table(f"{d}/motion.parquet"); col = lambda n: tab.column(n)[0].as_py()
names, Q = col("robot_joint_names"), np.asarray(col("robot_joint_positions"))
rp, rq = np.asarray(col("robot_root_position")), np.asarray(col("robot_root_wxyz"))
op, oq = np.asarray(col("object_body_position"))[:, 0], np.asarray(col("object_body_wxyz"))[:, 0]
box = trimesh.load(f"{d}/object_scaled.stl", force="mesh")
T = len(Q)
low = np.array([(R.from_quat(oq[t], scalar_first=True).apply(box.vertices) + op[t])[:, 2].min() for t in range(T)])
held = np.flatnonzero(low > low.min() + 0.01)[::step]

model = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
geom = pin.buildGeomFromUrdf(model, urdf, pin.GeometryType.VISUAL, package_dirs=[urdf.rsplit("/", 1)[0]])
data, gdata = model.createData(), None
gdata = pin.GeometryData(geom)
FING = ("thumb", "index", "middle", "ring", "little") if "inspire" in urdf else ("thumb", "index", "middle")
parts = {}  # (side, finger) -> list of (geom index, vertices)
for gi, g in enumerate(geom.geometryObjects):
    for side in ("left", "right"):
        for f in FING:
            if re.match(rf"{side}_(hand_)?{f}_", g.name):
                v = trimesh.load(g.meshPath, force="mesh").vertices * g.meshScale
                parts.setdefault((side, f), []).append((gi, v[:: max(1, len(v) // 400)]))
idx = [model.joints[model.getJointId(n)].idx_q for n in names]
gap = {k: [] for k in parts}
for t in held:
    q = pin.neutral(model); q[:3] = rp[t]; w, x, y, z = rq[t]; q[3:7] = [x, y, z, w]
    q[idx] = Q[t]
    pin.forwardKinematics(model, data, q); pin.updateGeometryPlacements(model, data, geom, gdata)
    Rb, pb = R.from_quat(oq[t], scalar_first=True), op[t]
    for k, lst in parts.items():
        pts = np.concatenate([v @ gdata.oMg[gi].rotation.T + gdata.oMg[gi].translation for gi, v in lst])
        local = Rb.inv().apply(pts - pb)
        gap[k].append(-trimesh.proximity.signed_distance(box, local).max())  # trimesh: positive inside
print(f"{d.rsplit('/', 1)[-1]}: {len(held)} held frames checked")
for side in ("left", "right"):
    print(f"  robot {side}: " + "  ".join(f"{f} {np.median(gap[(side, f)]) * 100:+.1f}" for f in FING) + " cm (median closest finger surface to box; - = inside)")
h = pq.read_table(glob.glob(f"{hand_dir}/**/*.parquet", recursive=True)[0])
for side in ("left", "right"):
    J = np.asarray(h.column(f"mano_{side}_joints")[0].as_py())
    g = []
    for t in held:
        local = R.from_quat(oq[t], scalar_first=True).inv().apply(J[t, [4, 8, 12, 16, 20]] - op[t])
        g.append(-trimesh.proximity.signed_distance(box, local) - 0.006)
    print(f"  human {side} tips: " + "  ".join(f"{f} {v * 100:+.1f}" for f, v in zip(("thumb", "index", "middle", "ring", "little"), np.median(g, 0))) + " cm")
