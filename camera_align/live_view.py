"""Live head-camera view in a browser while you adjust the camera, with the camera tilt drawn on the image.

  - video: the robot's camera server (gear_sonic composed_camera, port 5555), through the client's own reader
  - tilt vs gravity: the D435i accelerometer, streamed over ssh (realsense accel works while colour is streaming)
  - tilt vs torso: also needs the pelvis IMU + waist joints, read by a helper in a Python that has unitree_sdk2py
    (--sdk-python, e.g. the groot_wbc conda env). Without it only the gravity numbers are shown.

usage (repo root): GR00T_WBC=<GR00T-WholeBodyControl> .venv/bin/python camera_align/live_view.py \
         [--port 8090] [--sdk-python ~/miniforge3/envs/groot_wbc/bin/python] [--iface enp11s0] [--ssh g1]
then open http://<this machine>:8090"""
import argparse, json, math, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "client"))
from g1_client import Camera

ap = argparse.ArgumentParser()
ap.add_argument("--camera", default="official:192.168.123.164:5555")
ap.add_argument("--gr00t-wbc", default=os.environ.get("GR00T_WBC", ""))
ap.add_argument("--port", type=int, default=8090); ap.add_argument("--ssh", default="g1")
ap.add_argument("--robot-python", default="~/GR00T-WholeBodyControl/.venv_camera/bin/python")
ap.add_argument("--sdk-python", default=os.path.expanduser("~/miniforge3/envs/groot_wbc/bin/python"))
ap.add_argument("--iface", default="enp11s0")
ap.add_argument("--ref", default=os.path.join(ROOT, "camera_align", "head_cam_d435i.json"), help="json whose pitch is shown for comparison")
a = ap.parse_args()

state = {"up_cam": None, "up_torso": None, "frame": None, "jpeg": None}
lock = threading.Lock()

ACCEL_LOOP = r'''
import pyrealsense2 as rs, numpy as np, time, sys
p = rs.pipeline(); c = rs.config(); c.enable_stream(rs.stream.accel); p.start(c); buf = []; t = time.time()
while True:
    for f in p.wait_for_frames():
        if f.is_motion_frame():
            d = f.as_motion_frame().get_motion_data(); buf.append((d.x, d.y, d.z))
    if time.time() - t > 0.2 and buf:
        m = np.mean(buf, 0); print("%.4f %.4f %.4f" % tuple(m), flush=True); buf = []; t = time.time()
'''
LOWSTATE_LOOP = r'''
import sys, time, numpy as np
from scipy.spatial.transform import Rotation as R
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
ChannelFactoryInitialize(0, sys.argv[1]); last = [None]
def cb(m): last[0] = (list(m.imu_state.quaternion), [m.motor_state[i].q for i in (12, 13, 14)])
s = ChannelSubscriber("rt/lowstate", LowState_); s.Init(cb, 10)
while True:
    time.sleep(0.2)
    if last[0] is None: continue
    q, (wy, wr, wp) = last[0]
    Rt = R.from_quat(q, scalar_first=True) * R.from_euler("z", wy) * R.from_euler("x", wr) * R.from_euler("y", wp)
    print("%.5f %.5f %.5f" % tuple(Rt.inv().apply([0, 0, 1.0])), flush=True)
'''

def reader(cmd, key, stdin=None):
    while True:
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            if stdin: p.stdin.write(stdin); p.stdin.close()
            for line in p.stdout:
                v = np.array([float(x) for x in line.split()])
                with lock: state[key] = (v / np.linalg.norm(v), time.time())
        except Exception as e:
            print(key, "reader error:", e)
        time.sleep(2)

threading.Thread(target=reader, args=(["ssh", a.ssh, f"{a.robot_python} -u -"], "up_cam", ACCEL_LOOP), daemon=True).start()
if a.sdk_python and os.path.exists(a.sdk_python):
    threading.Thread(target=reader, args=([a.sdk_python, "-I", "-c", LOWSTATE_LOOP, a.iface], "up_torso"), daemon=True).start()

ref_pitch = None
if os.path.exists(a.ref):
    ref_pitch = math.degrees(json.load(open(a.ref))["euler_zyx_rad"][1])

def tilt_text():
    now = time.time(); lines = []
    with lock: uc, ut = state["up_cam"], state["up_torso"]
    if uc and now - uc[1] < 2:
        u = uc[0]  # optical frame: x right, y down, z fwd; points up
        lines.append(f"camera vs gravity: {math.degrees(math.asin(-u[2])):5.1f} deg down, roll {math.degrees(math.asin(u[0])):+4.1f}")
        if ut and now - ut[1] < 2:
            up_cam = np.array([u[2], -u[0], -u[1]]); up_t = ut[0]
            # pitch/roll of the camera in the torso frame with yaw ~ 0: solve on the x-z plane (small-roll approximation)
            pitch_t = math.degrees(math.asin(-np.clip(up_t[0], -1, 1)))      # torso lean forward
            lines.append(f"torso lean forward: {pitch_t:+5.1f} deg")
            lines.append(f"camera vs torso:   {math.degrees(math.asin(-np.clip(up_cam[0], -1, 1))) - pitch_t:5.1f} deg down"
                         + (f"   (file: {ref_pitch:.1f})" if ref_pitch is not None else ""))
        else:
            lines.append("torso IMU: no data (camera vs torso unavailable)")
    else:
        lines.append("camera accelerometer: no data yet")
    return lines

def restart_camera_server():
    # the D435i colour stream sometimes stalls (seen on USB 2.0) while the server keeps re-sending the last image
    subprocess.run(["ssh", a.ssh, "bash ~/GR00T-WholeBodyControl/start_gear_sonic_camera.sh"], capture_output=True, timeout=60)

def render_loop():
    cam = Camera(a.camera, a.gr00t_wbc or None)
    prev, changed_at, restarting = None, time.time(), [False]
    while True:
        rgb, age = cam.read()
        if rgb is None: time.sleep(0.02); continue
        small = rgb[::8, ::8].astype(np.int16)
        if prev is None or np.abs(small - prev).max() > 0: changed_at = time.time()
        prev = small; frozen = time.time() - changed_at
        if frozen > 2 and not restarting[0]:
            restarting[0] = True
            def go():
                restart_camera_server(); time.sleep(3); restarting[0] = False
            threading.Thread(target=go, daemon=True).start()
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); h, w = img.shape[:2]
        for k in (1, 2):  # thirds grid + centre cross
            cv2.line(img, (w * k // 3, 0), (w * k // 3, h), (200, 200, 200), 1)
            cv2.line(img, (0, h * k // 3), (w, h * k // 3), (200, 200, 200), 1)
        cv2.drawMarker(img, (w // 2, h // 2), (0, 220, 255), cv2.MARKER_CROSS, 30, 2)
        lines = tilt_text() + [f"{w}x{h}  frame age {age * 1000:.0f} ms"]
        if frozen > 1:
            lines.append(f"IMAGE FROZEN {frozen:.0f} s - restarting camera server")
        cv2.rectangle(img, (0, 0), (w, 22 * len(lines) + 8), (0, 0, 120) if frozen > 1 else (0, 0, 0), -1)
        for i, t in enumerate(lines):
            cv2.putText(img, t, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with lock: state["jpeg"] = jpg.tobytes()
        time.sleep(0.03)

PAGE = b"""<!doctype html><meta charset=utf-8><title>G1 head camera</title>
<body style="margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center">
<p>G1 head camera, live. Yellow cross = image centre.</p><img src="/stream" style="max-width:100%;height:auto"></body>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
            try:
                while True:
                    with lock: j = state["jpeg"]
                    if j:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(PAGE)

threading.Thread(target=render_loop, daemon=True).start()
print(f"open http://localhost:{a.port}  (or http://<this machine's IP>:{a.port})", flush=True)
ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
