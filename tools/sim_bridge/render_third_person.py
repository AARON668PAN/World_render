"""Third-person view of a camera-consistent retarget (ik_camera_track.py output: motion.parquet + camera_scaled.npz) with
Newton's ray-traced camera: the robot, the scaled object and the table slab of render_sim_depth.py, seen from in front of
the robot (azimuth / elevation / distance around the point between the chest and the object). For figures.
usage: G1_URDF=... render_third_person.py <camtrack dir> <out dir> <frame,frame,...> [az_deg=150] [el_deg=22] [dist=1.9] [fov=38]
writes <out>/third_<frame>.png (960x720)"""
import os
import sys

import imageio.v2 as imageio
import newton
import numpy as np
import pyarrow.parquet as pq
import warp as wp
from newton.sensors import SensorTiledCamera

G1_URDF = os.environ.get("G1_URDF", "/data/repos/chord/local/assets/g1/main_with_hand_nohead.urdf")
d, out = sys.argv[1], sys.argv[2]
frames = [int(x) for x in sys.argv[3].split(",")]
az, el, dist, fov = [float(x) for x in (sys.argv[4:8] + ["150", "22", "1.9", "38"][len(sys.argv[4:8]):])]
os.makedirs(out, exist_ok=True)
tab = pq.read_table(f"{d}/motion.parquet"); col = lambda n: tab.column(n)[0].as_py()
names, Q = col("robot_joint_names"), np.asarray(col("robot_joint_positions"))
root_p, root_q = np.asarray(col("robot_root_position")), np.asarray(col("robot_root_wxyz"))
obj_p, obj_q = np.asarray(col("object_body_position"))[:, 0], np.asarray(col("object_body_wxyz"))[:, 0]
cam = np.load(f"{d}/camera_scaled.npz"); c2w = cam["cam_to_world"]
W, H = 960, 720; F = H / 2 / np.tan(np.radians(fov) / 2)

wp.init()
builder = newton.ModelBuilder()
builder.add_urdf(source=G1_URDF, floating=True, collapse_fixed_joints=False, enable_self_collisions=False)
builder.add_urdf(source=col("object_urdf_paths")[0], floating=True, collapse_fixed_joints=False)
fwd = c2w[:, :3, 2].mean(0); yaw = np.arctan2(fwd[1], fwd[0]); heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
top = float(cam["table_top"]); centre = root_p.mean(0) + heading * (0.15 + 0.6)
builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(centre[0], centre[1], top / 2), wp.quat_rpy(0.0, 0.0, float(yaw))), hx=0.6, hy=0.8, hz=top / 2)
TABLE = builder.shape_count - 1
builder.add_ground_plane()
model = builder.finalize(); state = model.state()
labels = list(model.joint_label); q_start = model.joint_q_start.numpy(); q = model.joint_q.numpy().copy()
bl = list(model.body_label); OBJ = [i for i, b in enumerate(model.shape_body.numpy()) if b >= 0 and "object" in bl[b]]
free = [j for j, n in enumerate(labels) if n.endswith("/floating_base")]
slots = [q_start[next(j for j, n in enumerate(labels) if n.endswith("/" + name))] for name in names]


def set_free(joint, pos, wxyz):
    s = q_start[joint]; q[s:s + 3] = pos; q[s + 3:s + 7] = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]


sensor = SensorTiledCamera(model, config=SensorTiledCamera.RenderConfig(enable_shadows=True))
sensor.utils.create_default_light(enable_shadows=True)
px, py = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
dirs = np.stack([(px - W / 2) / F, -(py - H / 2) / F, -np.ones_like(px)], -1); dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
rays_np = np.zeros((1, H, W, 2, 3), np.float32); rays_np[0, :, :, 1] = dirs
rays = wp.array(rays_np, dtype=wp.vec3f)
color = sensor.utils.create_color_image_output(W, H); depth = sensor.utils.create_depth_image_output(W, H)
index = sensor.utils.create_shape_index_image_output(W, H)
built = False
for t in frames:
    set_free(free[0], root_p[t], root_q[t]); q[slots] = Q[t]; set_free(free[1], obj_p[t], obj_q[t])
    model.joint_q.assign(q); newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    if not built:
        newton.geometry.build_bvh_shape(model, state); built = True
    else:
        newton.geometry.refit_bvh_shape(model, state)
    chest = root_p[t] + np.array([0, 0, 0.35]); target = 0.5 * chest + 0.5 * obj_p[t]
    a = yaw + np.radians(az); e = np.radians(el)
    eye = target + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = target - eye; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    rot = np.stack([r, u, -f], 1)  # OpenGL camera axes in the world
    ld = 0.55 * f + 0.35 * r + np.array([0, 0, -0.75]); ld /= np.linalg.norm(ld)  # key light from over the camera's right shoulder
    sensor.utils.create_default_light(enable_shadows=True, direction=wp.vec3f(*ld.tolist()))
    xyzw = wp.quat_from_matrix(wp.mat33(*rot.flatten().tolist()))
    tf = wp.array([[wp.transformf(wp.vec3f(*eye.tolist()), xyzw)]], dtype=wp.transformf)
    sensor.update(state, tf, rays, color_image=color, depth_image=depth, shape_index_image=index)
    rgb = color.numpy()[0, 0].view(np.uint8).reshape(H, W, 4)[..., :3].astype(np.float32)
    idx = index.numpy()[0, 0].astype(np.int64); idx[idx >= model.shape_count] = -1
    lum = rgb.mean(-1, keepdims=True) / max(1.0, float(rgb.mean(-1)[idx == TABLE].max() if (idx == TABLE).any() else 1.0))
    rgb = np.where((idx == TABLE)[..., None], lum * np.array([196, 160, 118]), rgb)  # table: light wood instead of the default colour
    ob = np.isin(idx, OBJ)
    if ob.any():  # object: the tissue box's purple, shaded by the render's own brightness
        lo = rgb.mean(-1, keepdims=True) / max(1.0, float(rgb.mean(-1)[ob].max()))
        rgb = np.where(ob[..., None], (0.35 + 0.65 * lo) * np.array([128, 78, 178]), rgb)
    imageio.imwrite(f"{out}/third_{t:04d}.png", np.clip(rgb, 0, 255).astype(np.uint8))
    print("wrote", f"{out}/third_{t:04d}.png", flush=True)
