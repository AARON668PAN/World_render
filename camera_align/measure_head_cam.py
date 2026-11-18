"""Measure the real G1 head camera and write it in the same format as head_cam_calib.json (main branch,
sim_bridge/assets/), so a simulator that reads that file can use the real camera instead.

What is measured, and how:
  - intrinsics (fx, fy, cx, cy, 640x480, distortion): read from the D435i itself (realsense_probe.py over ssh).
  - camera pitch and roll relative to torso_link: the D435i accelerometer gives "up" in the camera frame; the pelvis IMU
    quaternion + the three waist joints (rt/lowstate) give "up" in the torso frame. Both are read at the same time, so
    the result does not depend on how the robot is standing or hanging. Keep the robot still while it runs.
  - NOT measured: camera yaw relative to the torso (gravity cannot see it) and the camera position. Both are taken
    from --base (the training calibration) unless given with --yaw-deg / --position. Check them with overlay_check.py.

Camera frame convention (same as calibrate_head_cam.py): x forward, y left, z up, in the torso frame,
rotation = scipy R.from_euler("zyx", euler_zyx_rad).

Runs on the PC on the robot network, in a Python env with unitree_sdk2py (e.g. conda env groot_wbc).
usage: python measure_head_cam.py --iface enp11s0 --ssh g1 --out camera_align/head_cam_d435i.json
         [--base camera_align/head_cam_calib_training.json] [--yaw-deg Y] [--position X Y Z] [--secs 3]"""
import argparse, json, math, os, subprocess, sys, threading, time
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="enp11s0"); ap.add_argument("--ssh", default="g1")
ap.add_argument("--robot-python", default="~/GR00T-WholeBodyControl/.venv_camera/bin/python")
ap.add_argument("--base", default=os.path.join(HERE, "head_cam_calib_training.json"))
ap.add_argument("--out", default=os.path.join(HERE, "head_cam_d435i.json"))
ap.add_argument("--yaw-deg", type=float, default=None); ap.add_argument("--position", type=float, nargs=3, default=None)
ap.add_argument("--secs", type=float, default=3.0)
a = ap.parse_args()
base = json.load(open(a.base))

# 1) pelvis IMU + waist joints, collected while the probe runs on the robot
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
ChannelFactoryInitialize(0, a.iface)
quat, waist, joints, lock = [], [], [], threading.Lock()
def on_state(m):
    with lock:
        quat.append(list(m.imu_state.quaternion)); waist.append([m.motor_state[i].q for i in (12, 13, 14)])
        joints.append([m.motor_state[i].q for i in range(29)])
sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(on_state, 10)

# 2) D435i intrinsics + accelerometer, over ssh
probe = open(os.path.join(HERE, "realsense_probe.py")).read()
cmd = ["ssh", a.ssh, f"{a.robot_python} - {a.secs}"]
res = subprocess.run(cmd, input=probe, capture_output=True, text=True, timeout=60 + a.secs)
if res.returncode:
    sys.exit(f"probe failed on the robot:\n{res.stderr[-2000:]}")
cam = json.loads(res.stdout.strip().splitlines()[-1])
time.sleep(0.2)
with lock:
    if len(quat) < 50: sys.exit(f"only {len(quat)} lowstate messages on {a.iface}: is the robot on and the interface right?")
    q = np.array(quat); w = np.array(waist); J = np.array(joints)

# pelvis orientation (unitree quaternion is w, x, y, z); torso = pelvis * Rz(waist_yaw) Rx(waist_roll) Ry(waist_pitch)
q_mean = q.mean(0); q_mean /= np.linalg.norm(q_mean)
R_pelvis = R.from_quat(q_mean, scalar_first=True)
wy, wr, wp = w.mean(0)
R_torso = R_pelvis * R.from_euler("z", wy) * R.from_euler("x", wr) * R.from_euler("y", wp)
up_torso = R_torso.inv().apply([0, 0, 1.0])

# camera: optical frame (x right, y down, z fwd) -> body frame (x fwd, y left, z up)
acc = np.array(cam["accel_mean"]); up_opt = acc / np.linalg.norm(acc)
up_cam = np.array([up_opt[2], -up_opt[0], -up_opt[1]])

# solve R_tc = R.from_euler("zyx", [yaw, pitch, roll]) with yaw fixed, such that R_tc^T up_torso = up_cam
yaw = math.radians(a.yaw_deg) if a.yaw_deg is not None else base["euler_zyx_rad"][0]
from scipy.optimize import least_squares
def resid(p):
    Rtc = R.from_euler("zyx", [yaw, p[0], p[1]])
    return Rtc.inv().apply(up_torso) - up_cam
sol = least_squares(resid, x0=[base["euler_zyx_rad"][1], 0.0])
pitch, roll = sol.x

intr = cam["color_640x480"]; W, H = intr["width"], intr["height"]
out = {
    "position_torso": a.position if a.position is not None else base["position_torso"],
    "euler_zyx_rad": [yaw, float(pitch), float(roll)],
    "focal_px": (intr["fx"] + intr["fy"]) / 2,
    "image_size": [W, H],
    "hfov_deg": math.degrees(2 * math.atan(W / 2 / intr["fx"])),
    "vfov_deg": math.degrees(2 * math.atan(H / 2 / intr["fy"])),
    "fx": intr["fx"], "fy": intr["fy"], "cx": intr["cx"], "cy": intr["cy"],
    "distortion": {"model": intr["model"], "coeffs": intr["coeffs"]},
    "mujoco_fovy_deg": math.degrees(2 * math.atan(H / 2 / intr["fy"])),
    "isaac_focal_length_mm_at_aperture_20.955": intr["fx"] * 20.955 / W,
    "measured": {
        "what": "pitch/roll from D435i accelerometer vs pelvis IMU + waist joints; intrinsics from the camera",
        "not_measured": "yaw" + ("" if a.yaw_deg is not None else " (copied from base)") + ", position" + ("" if a.position is not None else " (copied from base)"),
        "base": os.path.basename(a.base),
        "camera_serial": cam["serial"],
        "camera_pitch_down_vs_gravity_deg": math.degrees(math.asin(-up_opt[2])),
        "torso_lean_forward_vs_gravity_deg": -math.degrees(math.asin(np.clip(up_torso[0], -1, 1))),
        "pelvis_quat_wxyz": q_mean.tolist(), "waist_yaw_roll_pitch_rad": [float(wy), float(wr), float(wp)],
        "joint_q_29_unitree_order": J.mean(0).tolist(),
        "accel_mean_optical": cam["accel_mean"], "accel_std": cam["accel_std"],
        "lowstate_msgs": len(q),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
}
json.dump(out, open(a.out, "w"), indent=2)
print(f"camera pitch relative to torso: {math.degrees(pitch):.1f} deg down (training: {math.degrees(base['euler_zyx_rad'][1]):.1f})")
print(f"camera roll relative to torso:  {math.degrees(roll):+.1f} deg (training: {math.degrees(base['euler_zyx_rad'][2]):+.1f})")
print(f"hfov x vfov: {out['hfov_deg']:.1f} x {out['vfov_deg']:.1f} deg (training: {base['hfov_deg']:.1f} x {base['vfov_deg']:.1f})")
print(f"accel std {np.round(cam['accel_std'], 3).tolist()} m/s^2 (above ~0.3 = the robot was moving; measure again)")
print(f"wrote {a.out}")
