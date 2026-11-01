"""Render the camera-consistent G1 + Dex3 result from the tracked human camera with Newton's ray-traced camera:
exact OpenCV intrinsics and full camera rotation (roll included, no image-space warp), giving per-pixel depth
(camera z, in the scaled world), a label map and a shaded colour image.

Scene: G1 (head-less URDF, the camera sits at the head), the scaled box, a table slab at the scaled table height in
front of the robot, the ground plane.
Labels: 0 nothing, 1 table, 2 box, 3 robot hand (Dex3 links), 4 rest of the robot, 5 ground.
Also renders the robot alone and the box alone (amodal depth, for occlusion-order checks) and the Dex3 fingertip /
palm positions (projected pixel and camera z; thumb, index, middle tips, palm; right then left).

usage: render_sim_depth.py <camtrack dir> <out dir> [scale of the output resolution, default 0.5]
writes <out>/sim_depth.npz (depth, depth_robot, depth_box float16 (T, h, w), label uint8, tips_uv, tips_z, focal,
       scale) and <out>/sim_color.mp4
"""

import os
import re
import sys

import imageio.v2 as imageio
import newton
import numpy as np
import pyarrow.parquet as pq
import warp as wp
from newton.sensors import SensorTiledCamera

G1_URDF = os.environ.get("G1_URDF", "/data/repos/chord/local/assets/g1/main_with_hand_nohead.urdf")  # or assets/g1_inspire/g1_inspire_ftp_nohead.urdf
INSPIRE = "inspire" in G1_URDF
d, out = sys.argv[1], sys.argv[2]
res = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
os.makedirs(out, exist_ok=True)
tab = pq.read_table(f"{d}/motion.parquet")
col = lambda n: tab.column(n)[0].as_py()
names, Q = col("robot_joint_names"), np.asarray(col("robot_joint_positions"))
root_p, root_q = np.asarray(col("robot_root_position")), np.asarray(col("robot_root_wxyz"))
obj_p, obj_q = np.asarray(col("object_body_position"))[:, 0], np.asarray(col("object_body_wxyz"))[:, 0]
cam = np.load(f"{d}/camera_scaled.npz")
c2w = cam["cam_to_world"]
W, H = int(round(int(cam["width"]) * res)), int(round(int(cam["height"]) * res))
F = float(cam["focal"]) * res
T = len(Q)

wp.init()
builder = newton.ModelBuilder()
builder.add_urdf(source=G1_URDF, floating=True, collapse_fixed_joints=False, enable_self_collisions=False)
builder.add_urdf(source=col("object_urdf_paths")[0], floating=True, collapse_fixed_joints=False)
# table slab in front of the robot: near edge 0.15 m ahead of the pelvis, 1.2 m deep, 1.6 m wide
fwd = c2w[:, :3, 2].mean(0)
yaw = np.arctan2(fwd[1], fwd[0])
heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
top = float(cam["table_top"])
centre = root_p.mean(0) + heading * (0.15 + 0.6)
builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(centre[0], centre[1], top / 2), wp.quat_rpy(0.0, 0.0, float(yaw))),
                      hx=0.6, hy=0.8, hz=top / 2)
builder.add_ground_plane()
model = builder.finalize()
state = model.state()

labels = list(model.joint_label)
q_start = model.joint_q_start.numpy()
q = model.joint_q.numpy().copy()
free = [j for j, n in enumerate(labels) if n.endswith("/floating_base")]
robot_free, object_free = free[0], free[1]
slots = [q_start[next(j for j, n in enumerate(labels) if n.endswith("/" + name))] for name in names]

# shape -> label
body_label = list(model.body_label)
shape_body = model.shape_body.numpy()
obj_body = next(b for b, n in enumerate(body_label) if n.startswith("object_scaled"))
shape_lab = np.ones(model.shape_count, np.uint8)
shape_lab[list(model.shape_type.numpy()).index(int(newton.GeoType.PLANE))] = 5
for s, b in enumerate(shape_body):
    if b < 0:
        continue
    if b == obj_body:
        shape_lab[s] = 2
    elif "_hand_" in body_label[b].rsplit("/", 1)[-1] or (INSPIRE and re.search(r"_(base_link|palm_force_sensor|thumb_|index_|middle_|ring_|little_)", body_label[b].rsplit("/", 1)[-1])):
        shape_lab[s] = 3
    else:
        shape_lab[s] = 4
shape_lab = np.concatenate([shape_lab, [0]])  # index -1 (no hit) -> 0
shape_body_ext = np.concatenate([shape_body, [-1]]).astype(np.int16)  # per hit shape: its body (-1: static / none)


def set_free(joint, pos, wxyz):
    s = q_start[joint]
    q[s:s + 3] = pos
    q[s + 3:s + 7] = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]


sensor = SensorTiledCamera(model, config=SensorTiledCamera.RenderConfig(enable_shadows=False))
sensor.utils.create_default_light(enable_shadows=False)
# FISHEYE="W,H,f": render through an equidistant fisheye (G1 head camera model, centred) at the same camera pose instead
# of the human pinhole camera; depth is then the camera z of the hit point as before
FISH = os.environ.get("FISHEYE")
# CAM_WHF="W,H,f": pinhole with this size and focal, centred on the human camera's optical axis (e.g. 448,336,332.5 = the
# 16:9 human video letterboxed into 4:3, the bands rendered too)
if os.environ.get("CAM_WHF"):
    W, H, F = int(os.environ["CAM_WHF"].split(",")[0]), int(os.environ["CAM_WHF"].split(",")[1]), float(os.environ["CAM_WHF"].split(",")[2])
if FISH:
    W, H, F = int(FISH.split(",")[0]), int(FISH.split(",")[1]), float(FISH.split(",")[2])
px, py = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
if FISH:
    dx, dy = (px - W / 2) / F, (py - H / 2) / F
    th = np.hypot(dx, dy); sc = np.where(th > 1e-9, np.sin(th) / np.maximum(th, 1e-9), 1.0)
    dirs = np.stack([dx * sc, -dy * sc, -np.cos(th)], -1)  # unit rays, OpenGL camera axes
    cos_z = np.cos(th)
else:
    dirs = np.stack([(px - W / 2) / F, -(py - H / 2) / F, -np.ones_like(px)], -1)  # OpenGL camera axes
    cos_z = 1.0 / np.linalg.norm(dirs, axis=-1)  # camera z per unit ray length
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
rays_np = np.zeros((1, H, W, 2, 3), np.float32)
rays_np[0, :, :, 1] = dirs
rays = wp.array(rays_np.reshape(1, H, W, 2, 3), dtype=wp.vec3f)
color = sensor.utils.create_color_image_output(W, H)
depth = sensor.utils.create_depth_image_output(W, H)
index = sensor.utils.create_shape_index_image_output(W, H)
normal = sensor.utils.create_normal_image_output(W, H)  # world-frame surface normals (checked on the table top)
HAND_BASE = {b for b, n in enumerate(body_label) if re.search(r"(left|right)_base_link$", n)}  # Inspire hand housing
GL_FROM_CV = np.diag([1.0, -1.0, -1.0])

depth_all = np.zeros((T, H, W), np.float16)
depth_robot = np.zeros((T, H, W), np.float16)
depth_box = np.zeros((T, H, W), np.float16)
label_all = np.zeros((T, H, W), np.uint8)
body_all = np.full((T, H, W), -1, np.int16)
back_all = np.zeros((T, H, W), bool)  # Inspire hand back-cover pixels
nl_all = np.zeros((T, H, W, 3), np.float16)  # hand-housing surface normals in the base-link frame
BACK_AXIS, BACK_SIGN = int(os.environ.get("BACK_AXIS", 1)), float(os.environ.get("BACK_SIGN", -1))  # back of the hand in the base-link frame  # per pixel: index into body_names (for re-colouring the robot by part)
TIPS = []  # (body index, offset in body frame)
for side, sgn in (("right", 1), ("left", -1)):
    # Inspire FTP fingertips: mesh vertex farthest from the distal joint (mano_to_inspire.py); palm -> hand base link
    links = (((f"{side}_thumb_4", (sgn * 0.0331, 0.0123, -0.0103)), (f"{side}_index_2", (-0.0052, 0.0533, -sgn * 0.01)),
              (f"{side}_middle_2", (-0.0063, 0.0569, -sgn * 0.01)), (f"{side}_base_link", (0, 0, 0))) if INSPIRE else
             ((f"{side}_hand_thumb_2_link", (0, sgn * 0.035, 0)), (f"{side}_hand_index_1_link", (0.035, 0, 0)),
              (f"{side}_hand_middle_1_link", (0.035, 0, 0)), (f"{side}_hand_palm_link", (0, 0, 0))))
    for link, off in links:
        TIPS.append((next(b for b, n in enumerate(body_label) if n.endswith("/" + link)), np.array(off, float)))
tips_uv = np.zeros((T, len(TIPS), 2))
tips_z = np.zeros((T, len(TIPS)))
FAR = np.array([0.0, 0.0, -50.0])


def quat_rotate(xyzw, v):
    u, w = np.asarray(xyzw[:3]), xyzw[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def render(t, robot_at, box_at, keep):  # depth of the pixels whose label is in keep, 0 elsewhere
    set_free(robot_free, robot_at, root_q[t])
    q[slots] = Q[t]
    set_free(object_free, box_at, obj_p_q[t])
    model.joint_q.assign(q)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    newton.geometry.refit_bvh_shape(model, state)
    sensor.update(state, tf, rays, color_image=color, depth_image=depth, shape_index_image=index)
    dist = depth.numpy()[0, 0]
    idx = index.numpy()[0, 0].astype(np.int64)
    idx[idx >= model.shape_count] = -1
    return np.where((dist > 0) & np.isin(shape_lab[idx], keep), dist * cos_z, 0).astype(np.float16)


obj_p_q = obj_q
writer = imageio.get_writer(f"{out}/sim_color.mp4", fps=30, codec="libx264", quality=8, macro_block_size=8)
built = False
for t in range(T):
    set_free(robot_free, root_p[t], root_q[t])
    q[slots] = Q[t]
    set_free(object_free, obj_p[t], obj_q[t])
    model.joint_q.assign(q)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    if not built:
        newton.geometry.build_bvh_shape(model, state)
        built = True
    else:
        newton.geometry.refit_bvh_shape(model, state)
    w2c = np.linalg.inv(c2w[t])
    body_q = state.body_q.numpy()
    for i, (b, off) in enumerate(TIPS):
        p = w2c[:3, :3] @ (body_q[b, :3] + quat_rotate(body_q[b, 3:], off)) + w2c[:3, 3]
        if FISH:
            th_ = np.arctan2(np.hypot(p[0], p[1]), p[2]); r_ = np.hypot(p[0], p[1]) + 1e-12
            tips_uv[t, i] = [F * th_ * p[0] / r_ + W / 2, F * th_ * p[1] / r_ + H / 2]
        else:
            tips_uv[t, i] = [F * p[0] / p[2] + W / 2, F * p[1] / p[2] + H / 2]
        tips_z[t, i] = p[2]
    rot = c2w[t, :3, :3] @ GL_FROM_CV
    xyzw = wp.quat_from_matrix(wp.mat33(*rot.flatten().tolist()))
    tf = wp.array([[wp.transformf(wp.vec3f(*c2w[t, :3, 3].tolist()), xyzw)]], dtype=wp.transformf)
    sensor.update(state, tf, rays, color_image=color, depth_image=depth, shape_index_image=index, normal_image=normal)
    dist = depth.numpy()[0, 0]
    idx = index.numpy()[0, 0].astype(np.int64)
    idx[idx >= model.shape_count] = -1
    depth_all[t] = np.where(dist > 0, dist * cos_z, 0).astype(np.float16)
    # hand back plate: housing pixels whose surface faces the back of the hand (-y in the base link frame; the palm pad
    # sits at +y), for re-colouring (the real Inspire FTP has a white back cover, the rest of the hand is black)
    nw = normal.numpy()[0, 0]; bod = shape_body_ext[idx]
    if t == 0:
        tab = shape_lab[idx] == 1
        print("table-top normal (should be ~0,0,1 if world frame):", np.round(nw[tab].mean(0), 2) if tab.any() else "no table", flush=True)
    for b in HAND_BASE:
        m = bod == b
        if m.any():
            qx, qy, qz, qw = body_q[b, 3:7]
            Rb = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                           [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                           [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            nl = nw[m] @ Rb  # world -> body frame
            nl_all[t][m] = nl.astype(np.float16)
            back_all[t][m] = nl[:, BACK_AXIS] * BACK_SIGN > 0.55
    label_all[t] = shape_lab[idx]
    body_all[t] = shape_body_ext[idx]
    rgba = color.numpy()[0, 0].view(np.uint8).reshape(H, W, 4)
    writer.append_data(rgba[..., :3].copy())
    depth_robot[t] = render(t, root_p[t], obj_p[t] + FAR, (3, 4))  # box moved away: robot alone
    depth_box[t] = render(t, root_p[t] + FAR, obj_p[t], (2,))  # robot moved away: box alone
writer.close()
np.savez_compressed(f"{out}/sim_depth.npz", depth=depth_all, depth_robot=depth_robot, depth_box=depth_box, label=label_all,
                    body=body_all, body_names=np.array([n.rsplit("/", 1)[-1] for n in body_label]), hand_back=back_all, hand_normal=nl_all,
                    tips_uv=tips_uv, tips_z=tips_z, focal=F, scale=float(cam["scale"]))
hand = label_all == 3
print(f"{T} frames {W}x{H}: robot hand visible in {int(hand.any((1, 2)).sum())} frames, "
      f"box in {int((label_all == 2).any((1, 2)).sum())}; depth range {depth_all[depth_all > 0].min():.2f}-{depth_all.max():.2f} m (scaled world)")
