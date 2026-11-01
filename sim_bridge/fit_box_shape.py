"""Fit the reconstructed object mesh's size to the video: per-axis scale and offset along the mesh's principal axes
(object frame, so the per-frame poses stay as they are), so that its projection matches the SAM3 object mask.
Pixels covered by the human hands / forearms count as allowed (the object may be hidden behind them). The lowest
point at the resting frame is kept on the support plane.

Loss per frame: mean distance from projected surface samples to the allowed region (object mask or human mask)
+ mean distance from object-mask pixels to the projected silhouette (symmetric chamfer, px at quarter resolution).

usage: fit_box_shape.py <ego parquet dir> <ego_camera_refined.npz> <box_masks.npz> <human_table_masks.npz> <out dir>
writes <out>/<name>_fit.obj + <name>_fit_visual.stl, <out>/staged/ (parquet copy pointing at the fitted mesh)
"""

import glob
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation as R

parquet_dir, cam_npz, box_npz, ht_npz, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
pfile = glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True)[0]
table = pq.read_table(pfile)
col = lambda n: table.column(n)[0].as_py()
obj_p, obj_q = np.asarray(col("object_body_position"))[:, 0], np.asarray(col("object_body_wxyz"))[:, 0]
mesh_path = col("object_mesh_paths")[0].replace("/workspace/video_to_data", os.environ.get("CHORD", "/data/repos/chord"))
mesh = trimesh.load(mesh_path, force="mesh")
cams = np.load(cam_npz)
c2w, F0, W0, H0 = cams["cam_to_world"], float(cams["focal"]), int(cams["width"]), int(cams["height"])
DS = 4
W, H, F = W0 // DS, H0 // DS, F0 / DS
T = len(c2w)


def unpack(z, key):
    return np.unpackbits(z[key], axis=-1)[..., :W0].astype(bool)


box_m = unpack(np.load(box_npz), "masks")
ht = np.load(ht_npz)
human = unpack(ht, "hand") | unpack(ht, "forearm")
frames = list(range(0, T, 6))
allowed_d = {t: ndimage.distance_transform_edt(~(box_m[t] | human[t])[::DS, ::DS]) for t in frames}
box_px = {t: np.argwhere(box_m[t][::DS, ::DS])[::3] for t in frames}

# principal frame of the mesh (object frame)
verts = np.asarray(mesh.vertices)
centre = verts.mean(0)
_, _, vt = np.linalg.svd(verts - centre, full_matrices=False)
axes = vt.T  # columns = principal axes
proj = (verts - centre) @ axes
ext0 = np.percentile(proj, 99.5, 0) - np.percentile(proj, 0.5, 0)
samples = np.asarray(mesh.sample(6000, return_index=False))
rest = int(np.argmin(obj_p[:, 2]))
rot = {t: R.from_quat(obj_q[t], scalar_first=True) for t in set(frames) | {rest}}
table_z = (rot[rest].apply(verts) + obj_p[rest])[:, 2].min()


def deform(p, x):
    s, o = np.exp(x[:3]), x[3:]
    return centre + (((p - centre) @ axes) * s + o) @ axes.T


def frame_loss(t, pts_obj):
    w2c = np.linalg.inv(c2w[t])
    c = (rot[t].apply(pts_obj) + obj_p[t]) @ w2c[:3, :3].T + w2c[:3, 3]
    u, v = F * c[:, 0] / c[:, 2] + W / 2, F * c[:, 1] / c[:, 2] + H / 2
    ins = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (c[:, 2] > 0.05)
    ui, vi = u[ins].astype(int), v[ins].astype(int)
    d_out = allowed_d[t][vi, ui].mean() if ins.any() else 50.0
    occ = np.zeros((H, W), bool)
    occ[vi, ui] = True
    occ = ndimage.binary_dilation(occ, iterations=1)
    d_in = ndimage.distance_transform_edt(~occ)[box_px[t][:, 0], box_px[t][:, 1]].mean() if len(box_px[t]) else 0.0
    return d_out + d_in


def loss(x):
    pts = deform(samples, x)
    sil = np.mean([frame_loss(t, pts) for t in frames])
    low = (rot[rest].apply(deform(verts[::20], x)) + obj_p[rest])[:, 2].min()
    return sil + ((low - table_z) / 0.005) ** 2


x0 = np.zeros(6)
l0 = loss(x0)
res = optimize.minimize(loss, x0, method="Powell", options={"maxiter": 4000, "xtol": 1e-3, "ftol": 1e-4})
x = res.x
print(f"silhouette loss {l0:.2f} -> {res.fun:.2f} px (quarter res); extents {np.round(ext0 * 100, 1)} cm -> "
      f"{np.round(ext0 * np.exp(x[:3]) * 100, 1)} cm; offset {np.round(x[3:] * 100, 1)} cm (principal axes)")


def iou_all(pts_obj):
    out_iou = []
    for t in range(0, T, 3):
        w2c = np.linalg.inv(c2w[t])
        c = (R.from_quat(obj_q[t], scalar_first=True).apply(pts_obj) + obj_p[t]) @ w2c[:3, :3].T + w2c[:3, 3]
        u, v = (F * 2) * c[:, 0] / c[:, 2] + W0 / 4, (F * 2) * c[:, 1] / c[:, 2] + H0 / 4
        ins = (u >= 0) & (u < W0 // 2) & (v >= 0) & (v < H0 // 2)
        occ = np.zeros((H0 // 2, W0 // 2), bool)
        occ[v[ins].astype(int), u[ins].astype(int)] = True
        occ = ndimage.binary_closing(ndimage.binary_dilation(occ, iterations=1), iterations=3)
        bm, hm = box_m[t][::2, ::2], human[t][::2, ::2]
        # IoU with the object mask, ignoring pixels where the human hides it
        valid = ~hm | bm
        out_iou.append((occ & bm).sum() / max(((occ | bm) & valid).sum(), 1))
    return np.median(out_iou)


dense = np.asarray(mesh.sample(60000, return_index=False))
print(f"object mask IoU (human pixels ignored): {iou_all(dense):.3f} -> {iou_all(deform(dense, x)):.3f}")
name = os.path.splitext(os.path.basename(mesh_path))[0]
fitted = trimesh.Trimesh(deform(verts, x), mesh.faces, process=False)
fitted.export(f"{out}/{name}_fit.obj")
fitted.export(f"{out}/{name}_fit_visual.stl")
np.savez(f"{out}/box_fit.npz", x=x, centre=centre, axes=axes, extents_before=ext0, extents_after=ext0 * np.exp(x[:3]))
# staged parquet copy pointing at the fitted mesh
dst = f"{out}/staged/" + os.path.relpath(os.path.dirname(pfile), parquet_dir)
os.makedirs(dst, exist_ok=True)
idx = table.column_names.index("object_mesh_paths")
table = table.set_column(idx, "object_mesh_paths", pa.array([[f"{os.path.abspath(out)}/{name}_fit.obj"]], type=table.schema.field("object_mesh_paths").type))
pq.write_table(table, f"{dst}/{os.path.basename(pfile)}")
print("wrote", f"{out}/{name}_fit.obj", "and", dst)
