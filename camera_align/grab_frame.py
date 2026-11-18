"""Save frames from the robot's head camera server (gear_sonic composed_camera, port 5555), through the same reader the
client uses, so the saved image is exactly what the student would see.
usage (repo root, .venv): GR00T_WBC=<GR00T-WholeBodyControl checkout> .venv/bin/python camera_align/grab_frame.py \
         [--camera official:192.168.123.164:5555] [--n 3] [--out camera_align/reference]"""
import argparse, os, sys, time
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "client"))
from g1_client import Camera

ap = argparse.ArgumentParser()
ap.add_argument("--camera", default="official:192.168.123.164:5555")
ap.add_argument("--gr00t-wbc", default=os.environ.get("GR00T_WBC", ""))
ap.add_argument("--n", type=int, default=3); ap.add_argument("--every", type=float, default=1.0)
ap.add_argument("--out", default=os.path.join(ROOT, "camera_align", "reference"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cam = Camera(a.camera, a.gr00t_wbc or None)
stamp = time.strftime("%Y%m%d_%H%M%S"); saved, last, t0 = 0, 0.0, time.time()
while saved < a.n and time.time() - t0 < 10 + a.n * a.every:
    rgb, age = cam.read()
    if rgb is not None and age < 0.5 and time.time() - last >= a.every:
        last = time.time(); p = os.path.join(a.out, f"real_{stamp}_{saved}.jpg")
        cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); print("saved", p, rgb.shape); saved += 1
    time.sleep(0.01)
if not saved: sys.exit("no frame: is the camera server running on the robot (start_gear_sonic_camera.sh)?")
