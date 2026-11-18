"""Check that the simulated head camera matches the real one.

Two uses:
  1) real frame + camera json: draw where the README layout (table edge, plate, orange, put-back spot) should land in the
     real image, if the json is right. Lines that miss the real plate = the json (or the plate placement) is off.
  2) real frame + sim frame (same scene, same robot pose, rendered with the json): side by side, 50/50 blend, and the sim's
     edges drawn on the real image. Aligned = the plate rim and the table edge sit on top of each other.

World frame (README section 3): origin on the floor under the pelvis, x forward, y left, z up. The torso is taken as
upright at --torso-z (pelvis height + 0.054 m); for use 1 put the robot in the start pose.

usage: python camera_align/overlay_check.py --real real.jpg --cam camera_align/head_cam_d435i.json [--sim sim.png]
         [--torso-z 0.834] [--plate X Y] [--orange X Y] [--out camera_align/reference/check]"""
import argparse, json, os
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--real", required=True); ap.add_argument("--cam", required=True); ap.add_argument("--sim")
ap.add_argument("--torso-z", type=float, default=0.834)
ap.add_argument("--table-z", type=float, default=0.76); ap.add_argument("--table-near-x", type=float, default=0.12)
ap.add_argument("--plate", type=float, nargs=2, default=[0.35, 0.04]); ap.add_argument("--plate-r", type=float, default=0.09)
ap.add_argument("--orange", type=float, nargs=2, default=[0.35, -0.13]); ap.add_argument("--orange-r", type=float, default=0.039)
ap.add_argument("--target", type=float, nargs=2, default=[0.40, -0.16])
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.real)), "check")
os.makedirs(out, exist_ok=True)

c = json.load(open(a.cam)); W, H = c["image_size"]
fx, fy = c.get("fx", c["focal_px"]), c.get("fy", c["focal_px"]); cx, cy = c.get("cx", W / 2), c.get("cy", H / 2)
def euler_zyx(yaw, pitch, roll):                                       # = scipy R.from_euler("zyx", [yaw, pitch, roll])
    cz, sz, cy_, sy, cx_, sx = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    # lowercase "zyx" in scipy = extrinsic (fixed axes): R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    return (np.array([[1, 0, 0], [0, cx_, -sx], [0, sx, cx_]]) @ np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))
R_tc = euler_zyx(*c["euler_zyx_rad"])                                  # camera body (x fwd, y left, z up) in torso
cam_pos = np.array([0, 0, a.torso_z]) + np.array(c["position_torso"])
R_wo = R_tc @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])           # optical (x right, y down, z fwd) in world

def project(P):
    pc = (np.atleast_2d(P) - cam_pos) @ R_wo
    return np.c_[fx * pc[:, 0] / pc[:, 2] + cx, fy * pc[:, 1] / pc[:, 2] + cy], pc[:, 2]

def ring(x, y, r, z, n=72):
    t = np.linspace(0, 2 * np.pi, n); return np.c_[x + r * np.cos(t), y + r * np.sin(t), np.full(n, z)]

def draw_layout(img):
    def poly(P, col, closed=True, th=2):
        uv, z = project(P); keep = z > 0
        if keep.sum() > 1: cv2.polylines(img, [np.round(uv[keep]).astype(np.int32)], closed, col, th, cv2.LINE_AA)
    poly(np.c_[np.full(50, a.table_near_x), np.linspace(-0.6, 0.6, 50), np.full(50, a.table_z)], (255, 255, 0), False)
    poly(ring(*a.plate, a.plate_r, a.table_z + 0.005), (255, 0, 255))
    poly(ring(*a.target, 0.05, a.table_z), (0, 200, 0))
    uv, z = project(np.array([*a.orange, a.table_z + a.orange_r]))
    if z[0] > 0: cv2.circle(img, tuple(np.round(uv[0]).astype(int)), int(round(fx * a.orange_r / z[0])), (0, 140, 255), 2, cv2.LINE_AA)
    return img

def caption(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (255, 255, 255), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA); return img

real = cv2.imread(a.real)
assert real.shape[:2] == (H, W), f"real frame is {real.shape[1]}x{real.shape[0]}, camera json says {W}x{H}"
layout = caption(draw_layout(real.copy()), "real + layout from json (magenta plate, orange, green put-back, cyan table edge)")
cv2.imwrite(os.path.join(out, "real_layout.png"), layout); print("wrote", os.path.join(out, "real_layout.png"))

if a.sim:
    sim = cv2.imread(a.sim)
    if sim.shape[:2] != (H, W):
        print(f"note: sim frame is {sim.shape[1]}x{sim.shape[0]}, resized to {W}x{H}"); sim = cv2.resize(sim, (W, H))
    blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120)
    edge_img = real.copy(); edge_img[cv2.dilate(edges, np.ones((2, 2), np.uint8)) > 0] = (255, 0, 255)
    top = np.hstack([caption(real.copy(), "real"), caption(sim.copy(), "sim")])
    bot = np.hstack([caption(blend, "50/50 blend"), caption(edge_img, "sim edges on real")])
    cv2.imwrite(os.path.join(out, "real_vs_sim.png"), np.vstack([top, bot])); print("wrote", os.path.join(out, "real_vs_sim.png"))
