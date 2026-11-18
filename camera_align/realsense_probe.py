"""Runs ON THE ROBOT (any Python with pyrealsense2, e.g. ~/GR00T-WholeBodyControl/.venv_camera). Prints one JSON line: the
D435i's serial, the 640x480 colour intrinsics and the mean accelerometer reading over a few seconds (camera frame: x right,
y down, z forward; at rest it reads the reaction to gravity, i.e. it points up). Works while the camera server is streaming
colour, because the accelerometer is a separate sensor.
usage: python realsense_probe.py [seconds] [serial]"""
import json, sys, time
import numpy as np
import pyrealsense2 as rs

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
dev = rs.context().query_devices()[0] if len(sys.argv) < 3 else next(d for d in rs.context().query_devices()
                                                                     if d.get_info(rs.camera_info.serial_number) == sys.argv[2])
serial = dev.get_info(rs.camera_info.serial_number)
intr = None
for s in dev.query_sensors():
    for p in s.get_stream_profiles():
        if p.stream_type() == rs.stream.color and p.format() == rs.format.rgb8 and p.fps() == 30:
            v = p.as_video_stream_profile()
            if (v.width(), v.height()) == (640, 480):
                i = v.get_intrinsics()
                intr = dict(width=i.width, height=i.height, fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy, model=str(i.model), coeffs=list(i.coeffs))
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_device(serial); cfg.enable_stream(rs.stream.accel)
pipe.start(cfg); acc, t0 = [], time.time()
while time.time() - t0 < secs:
    for f in pipe.wait_for_frames():
        if f.is_motion_frame():
            d = f.as_motion_frame().get_motion_data(); acc.append([d.x, d.y, d.z])
pipe.stop()
acc = np.array(acc)
print(json.dumps(dict(serial=serial, color_640x480=intr, accel_mean=acc.mean(0).tolist(), accel_std=acc.std(0).tolist(),
                      n_accel=len(acc), time=time.time())))
