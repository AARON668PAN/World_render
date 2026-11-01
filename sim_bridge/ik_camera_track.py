"""Camera-consistent whole-body retargeting of an ego video to G1 + Dex3.

The human scene (ego camera trajectory, Dex3 palm targets, object) is scaled uniformly by s about the floor
point under the mean camera, so that the scaled human camera sits at G1 head-camera height. A whole-body IK
then makes the calibrated G1 head camera follow the scaled human camera while both palms follow the scaled
hand targets; feet stay planted and the CoM stays over the feet. A uniformly scaled scene seen from a
uniformly scaled camera produces the same image, so the robot overlays the human video.

usage: ik_camera_track.py <dex3 parquet dir> <ego_camera_refined.npz> <head_cam_calib.json> <scale> <out dir>
"""

import glob
import json
import os
import sys

import numpy as np
import pink
import pinocchio as pin
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from pink.limits import ConfigurationLimit
from pink.tasks import ComTask, FrameTask, PostureTask
from scipy.spatial.transform import Rotation as R

# G1_URDF / PALM_LINKS (comma separated, left first) switch the hands, e.g. G1 + Inspire FTP from mano_to_inspire.py
URDF = os.environ.get("G1_URDF", "/data/repos/chord/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf")
PALMS = tuple(os.environ.get("PALM_LINKS", "left_hand_palm_link,right_hand_palm_link").split(","))
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
CV_FROM_OURS = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)  # columns: our x fwd / y left / z up in OpenCV axes

dex3_dir, cam_npz, calib_json, scale, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
os.makedirs(out_dir, exist_ok=True)
table = pq.read_table(glob.glob(f"{dex3_dir}/**/*.parquet", recursive=True)[0])
col = lambda n: table.column(n)[0].as_py()
cams = np.load(cam_npz)
c2w = cams["cam_to_world"]
T = len(c2w)
calib = json.load(open(calib_json))

center = np.array([c2w[:, 0, 3].mean(), c2w[:, 1, 3].mean(), 0.0])
S = lambda p: center + scale * (np.asarray(p) - center)
cam_pos = S(c2w[:, :3, 3])
cam_rot = c2w[:, :3, :3] @ CV_FROM_OURS  # world <- our camera axes
# PRESCALED_HANDS=1: the Dex3 retarget was already run on the scaled scene (ego_to_dex3.py --world_scale),
# so palms and object are taken as they are and only the camera is scaled here.
PRE = os.environ.get("PRESCALED_HANDS") == "1"
palm_p = {s: (np.asarray(col(f"robot_{s}_wrist_position")) if PRE else S(col(f"robot_{s}_wrist_position"))) for s in ("left", "right")}
palm_q = {s: np.asarray(col(f"robot_{s}_wrist_wxyz")) for s in ("left", "right")}
fingers = {s: (col(f"{s}_robot_finger_joint_names"), np.asarray(col(f"robot_{s}_finger_joints"))) for s in ("left", "right")}
# Dex3 finger joints are solved frame by frame and occasionally jump; smooth them in time (centred mean).
FW = int(os.environ.get("FINGER_SMOOTH", 7))
if FW > 1:
    kernel = np.ones(FW) / FW
    for side, (fn, fq) in fingers.items():
        padded = np.pad(fq, ((FW // 2, FW // 2), (0, 0)), mode="edge")
        fingers[side] = (fn, np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(fq.shape[1])], 1))
obj_p = np.asarray(col("object_body_position"))[:, 0] if PRE else S(np.asarray(col("object_body_position"))[:, 0])
obj_q = np.asarray(col("object_body_wxyz"))[:, 0]

# Scaled object mesh + URDF.
mesh_path = col("object_mesh_paths")[0].replace("/workspace/video_to_data", os.environ.get("CHORD", "/data/repos/chord"))
mesh = trimesh.load(mesh_path.replace(".obj", "_visual.stl"), force="mesh")
mesh.apply_scale(scale)
mesh.export(f"{out_dir}/object_scaled.stl")
open(f"{out_dir}/object_scaled.urdf", "w").write(f"""<?xml version="1.0"?>
<robot name="object_scaled"><link name="object">
<inertial><origin xyz="0 0 0"/><mass value="0.3"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
<visual><geometry><mesh filename="{out_dir}/object_scaled.stl"/></geometry></visual>
<collision><geometry><mesh filename="{out_dir}/object_scaled.stl"/></geometry></collision>
</link></robot>""")
rest = int(np.argmin(obj_p[:, 2]))
lowest = (R.from_quat(obj_q[rest], scalar_first=True).apply(mesh.vertices) + obj_p[rest])[:, 2].min()

# Robot model with a head_cam frame at the calibrated pose on torso_link; fingers locked.
full = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
torso_fid = full.getFrameId("torso_link")
torso_frame = full.frames[torso_fid]
cam_local = pin.SE3(R.from_euler("zyx", calib["euler_zyx_rad"]).as_matrix(), np.array(calib["position_torso"]))
full.addFrame(pin.Frame("head_cam", torso_frame.parentJoint, torso_fid, torso_frame.placement * cam_local, pin.FrameType.OP_FRAME))
finger_names = [n for s in ("left", "right") for n in fingers[s][0]]
q_full = pin.neutral(full)
model = pin.buildReducedModel(full, [full.getJointId(n) for n in finger_names], q_full)
data = model.createData()

# Initial stance: heading from the mean camera forward, camera above the pelvis, soles on z = 0.
fwd = cam_rot[:, :, 0].mean(axis=0)
yaw = np.arctan2(fwd[1], fwd[0])
q = pin.neutral(model)
q[3:7] = R.from_euler("z", yaw).as_quat()
pin.framesForwardKinematics(model, data, q)
cam0 = data.oMf[model.getFrameId("head_cam")].translation
sole = min(data.oMf[model.getFrameId(f)].translation[2] for f in FEET) - 0.03
# STANCE_DX shifts the planted feet along the heading (m); positive = closer to the scene.
shift = float(os.environ.get("STANCE_DX", 0.0)) * np.array([np.cos(yaw), np.sin(yaw)])
q[:3] = [cam_pos[:, 0].mean() - cam0[0] + shift[0], cam_pos[:, 1].mean() - cam0[1] + shift[1], -sole]
config = pink.Configuration(model, data, q)
feet = {f: FrameTask(f, position_cost=10.0, orientation_cost=10.0) for f in FEET}
for f, task in feet.items():
    task.set_target(config.get_transform_frame_to_world(f))
foot_mid = np.mean([config.get_transform_frame_to_world(f).translation for f in FEET], axis=0)
com = ComTask(cost=[5.0, 5.0, 0.0])
com.set_target(np.array([foot_mid[0], foot_mid[1], 0.0]))
camera = FrameTask("head_cam", position_cost=1.0, orientation_cost=float(os.environ.get("CAM_ORI", 0.5)))
palms = {p: FrameTask(p, position_cost=float(os.environ.get("PALM_POS", 1.0)),
                      orientation_cost=float(os.environ.get("PALM_ORI", 0.3))) for p in PALMS}
posture = PostureTask(cost=1e-2)
posture.set_target(q.copy())
# Continuity: stay close to the previous frame's solution so the IK does not hop between equivalent arm
# configurations (elbow out / elbow down) from one frame to the next.
continuity = PostureTask(cost=float(os.environ.get("SMOOTH", 0.5)))
continuity.set_target(q.copy())
limits = [ConfigurationLimit(model)]

names = [n for n in model.names[1:] if n != "root_joint"]  # actuated joints of the reduced model
idx = [model.joints[model.getJointId(n)].idx_q for n in names]
out_q, out_root_p, out_root_q = [], [], []
cam_err, cam_ang, palm_err = np.zeros(T), np.zeros(T), np.zeros((T, 2))
STRIDE = int(os.environ.get("FRAME_STRIDE", 1))  # >1: quick evaluation on a subset of frames
ITERS = int(os.environ.get("IK_ITERS", 80))
for t in range(0, T, STRIDE):
    camera.set_target(pin.SE3(cam_rot[t], cam_pos[t]))
    for k, side in enumerate(("left", "right")):
        palms[PALMS[k]].set_target(pin.SE3(R.from_quat(palm_q[side][t], scalar_first=True).as_matrix(), palm_p[side][t]))
    config = pink.Configuration(model, data, q)
    tasks = [camera, *palms.values(), *feet.values(), com, posture] + ([continuity] if t else [])
    for _ in range(ITERS if t else 4 * ITERS):
        vel = pink.solve_ik(config, tasks, dt=0.02, solver="daqp",
                            limits=limits, safety_break=False)
        config.integrate_inplace(vel, 0.02)
    q = config.q.copy()
    continuity.set_target(q.copy())
    M = config.get_transform_frame_to_world("head_cam")
    cam_err[t] = np.linalg.norm(M.translation - cam_pos[t])
    cam_ang[t] = np.degrees(np.linalg.norm(R.from_matrix(M.rotation.T @ cam_rot[t]).as_rotvec()))
    for k, p in enumerate(PALMS):
        palm_err[t, k] = np.linalg.norm(config.get_transform_frame_to_world(p).translation - [palm_p["left"], palm_p["right"]][k][t])
    out_root_p.append(q[:3].tolist())
    out_root_q.append([q[6], q[3], q[4], q[5]])
    row = [float(q[i]) for i in idx]
    for side in ("left", "right"):
        row += fingers[side][1][t].tolist()
    out_q.append(row)
sel = slice(0, T, STRIDE)
print(f"scale {scale} stance {os.environ.get('STANCE_DX', 0)}: camera position error mean {cam_err[sel].mean()*100:.1f} cm max {cam_err[sel].max()*100:.1f} cm |"
      f" camera angle error mean {cam_ang[sel].mean():.1f} deg max {cam_ang[sel].max():.1f} deg |"
      f" palm error mean {palm_err[sel].mean()*100:.1f} cm max {palm_err[sel].max()*100:.1f} cm")
if STRIDE > 1:
    sys.exit(0)

joint_names = names + [n for s in ("left", "right") for n in fingers[s][0]]
ee = np.stack([np.concatenate([palm_p[s], palm_q[s]], axis=1) for s in ("left", "right")], axis=1)
columns = {
    "robot_joint_names": [joint_names], "robot_joint_positions": [out_q],
    "robot_root_position": [out_root_p], "robot_root_wxyz": [out_root_q],
    "ee_link_names": [list(PALMS)], "ee_pose_w": [ee.tolist()],
    "object_urdf_paths": [[f"{out_dir}/object_scaled.urdf"]],
    "object_body_position": [obj_p[:, None].tolist()], "object_body_wxyz": [obj_q[:, None].tolist()],
}
pq.write_table(pa.table(columns), f"{out_dir}/motion.parquet")
scaled_c2w = c2w.copy()
scaled_c2w[:, :3, 3] = cam_pos
np.savez(f"{out_dir}/camera_scaled.npz", cam_to_world=scaled_c2w, focal=cams["focal"], width=cams["width"],
         height=cams["height"], table_top=lowest, scale=scale, center=center)
print(f"table top (scaled) {lowest:.3f} m; wrote {out_dir}/motion.parquet")
