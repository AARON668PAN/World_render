"""Runs ON THE ROBOT (pyrealsense2 + numpy). Stops nothing itself: run it while the camera server is stopped.
Captures raw depth (and the accelerometer), fits the table plane with RANSAC and prints one JSON line:
  - height_above_plane_m: perpendicular distance from the depth camera to the table top
  - pitch/roll of the camera relative to the table plane (an independent check of the accelerometer tilt)
Depth camera frame = optical (x right, y down, z forward); the colour camera sits ~1.5 cm to its side, which does not
change the height.
usage: python realsense_table_plane.py [frames=15] [inlier_mm=6]"""
import json, math, sys, time
import numpy as np
import pyrealsense2 as rs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
TOL = (float(sys.argv[2]) if len(sys.argv) > 2 else 6.0) / 1000

pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30); cfg.enable_stream(rs.stream.accel)
prof = pipe.start(cfg)
scale = prof.get_device().first_depth_sensor().get_depth_scale()
intr = prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
depths, acc = [], []
for _ in range(10): pipe.wait_for_frames()
while len(depths) < N:
    fs = pipe.wait_for_frames()
    for f in fs:
        if f.is_motion_frame():
            d = f.as_motion_frame().get_motion_data(); acc.append([d.x, d.y, d.z])
    d = fs.get_depth_frame()
    if d: depths.append(np.asanyarray(d.get_data()).astype(np.float32) * scale)
pipe.stop()

D = np.stack(depths); D[D == 0] = np.nan
z = np.nanmedian(D, 0)                                   # per-pixel median over frames
v, u = np.mgrid[0:480, 0:640]
ok = np.isfinite(z) & (z > 0.15) & (z < 1.5)
P = np.c_[(u[ok] - intr.ppx) / intr.fx * z[ok], (v[ok] - intr.ppy) / intr.fy * z[ok], z[ok]]
rng = np.random.default_rng(0); best = None
for _ in range(400):
    s = P[rng.choice(len(P), 3, replace=False)]
    n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
    if nn < 1e-9: continue
    n /= nn; dist = np.abs((P - s[0]) @ n); cnt = (dist < TOL).sum()
    if best is None or cnt > best[0]: best = (cnt, n, s[0])
inl = P[np.abs((P - best[2]) @ best[1]) < TOL]
c = inl.mean(0); _, _, Vt = np.linalg.svd(inl - c, full_matrices=False); n = Vt[2]
if n @ c > 0: n = -n                                    # normal points from the plane towards the camera ("up")
h = float(-(n @ c))                                      # distance from camera origin to the plane
acc = np.array(acc).mean(0); up_acc = acc / np.linalg.norm(acc)
print(json.dumps(dict(
    height_above_plane_m=h, inlier_fraction=float(len(inl) / len(P)), n_points=int(len(P)),
    plane_up_optical=n.tolist(), accel_up_optical=up_acc.tolist(),
    pitch_down_vs_plane_deg=math.degrees(math.asin(-n[2])), roll_vs_plane_deg=math.degrees(math.asin(n[0])),
    pitch_down_vs_gravity_deg=math.degrees(math.asin(-up_acc[2])), roll_vs_gravity_deg=math.degrees(math.asin(up_acc[0])),
    plane_vs_gravity_deg=math.degrees(math.acos(np.clip(n @ up_acc, -1, 1))),
    rms_mm=float(np.sqrt(np.mean(((inl - c) @ n) ** 2)) * 1000), depth_scale=scale, frames=N, time=time.time())))
