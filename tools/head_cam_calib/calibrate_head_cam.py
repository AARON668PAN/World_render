"""Calibrate the Unitree G1 head camera (cam_left_high) from real teleoperation frames.

For each frame the recorded arm + Dex3 joint angles pose the G1 model (waist and legs at zero: teleop
moves the arms only). Points sampled on the arm and hand meshes are projected with a pinhole camera rigidly
attached to torso_link and compared with SAM3 'robot arm'/'robot hand' masks via a symmetric chamfer
distance. Optimised: camera position (3), yaw/pitch/roll (3) in the torso frame and focal length (1);
principal point at the image centre.

usage: calibrate_head_cam.py <calib dir with frames.json + sam3_masks.npz> <out.json>
"""

import json
import sys
import tempfile

import numpy as np
import pinocchio as pin
import trimesh
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation as R

from g1_state_names import STATE_NAMES, STATE_TO_JOINT

URDF = "/data/repos/chord/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf"
MESHES = "/data/repos/chord/robotic_grounding/flash_chord/src/flash_chord/assets/meshes"
W, H = 640, 480
ARM_KEYS = ("shoulder", "elbow", "wrist", "hand", "rubber")

def load_model():
    text = open(URDF).read().replace('filename="../../meshes/', f'filename="{MESHES}/')
    path = tempfile.mktemp(suffix=".urdf")
    open(path, "w").write(text)
    model = pin.buildModelFromUrdf(path)  # fixed base: only the torso-relative geometry matters
    geom = pin.buildGeomFromUrdf(model, path, pin.GeometryType.VISUAL)
    return model, geom


def sample_arm_points(geom, per_mesh=1200):
    samples = []
    for k, g in enumerate(geom.geometryObjects):
        if not any(key in g.name for key in ARM_KEYS):
            continue
        mesh = trimesh.load(g.meshPath, force="mesh")
        pts, _ = trimesh.sample.sample_surface(mesh, per_mesh, seed=k)
        samples.append((k, np.asarray(pts) * np.asarray(g.meshScale)))
    return samples


def main():
    calib_dir, out_path = sys.argv[1], sys.argv[2]
    rows = json.load(open(f"{calib_dir}/frames.json"))[::2]
    # Work at half resolution for speed; the focal length is reported at full resolution.
    masks = np.load(f"{calib_dir}/sam3_masks.npz")["masks"][::2][:, ::2, ::2]
    model, geom = load_model()
    data, gdata = model.createData(), pin.GeometryData(geom)
    torso = model.getFrameId("torso_link")
    samples = sample_arm_points(geom)

    # Arm/hand surface points in the torso frame, per frame.
    clouds = []
    for row in rows:
        q = pin.neutral(model)
        for name, value in zip(STATE_NAMES, row["state"]):
            j = model.getJointId(STATE_TO_JOINT[name])
            q[model.joints[j].idx_q] = value
        pin.framesForwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, geom, gdata)
        torso_inv = data.oMf[torso].inverse()
        pts = []
        for k, local in samples:
            M = torso_inv * gdata.oMg[k]
            pts.append(local @ M.rotation.T + M.translation)
        clouds.append(np.concatenate(pts))
    # Distance maps of the real masks (and their complements) for a smooth chamfer loss.
    dist_to_mask = [ndimage.distance_transform_edt(~m) for m in masks]
    mask_pixels = [np.argwhere(m)[:: max(1, int(m.sum()) // 3000)] for m in masks]

    def project(params, cloud):
        pos, euler, f = params[:3], params[3:6], params[6]
        rot = R.from_euler("zyx", euler).as_matrix()  # camera x forward, y left, z up in the torso frame
        cam = (cloud - pos) @ rot
        front = cam[:, 0] > 0.05
        cam = cam[front]
        u = W / 4 - (f / 2) * cam[:, 1] / cam[:, 0]
        v = H / 4 - (f / 2) * cam[:, 2] / cam[:, 0]
        return u, v

    def loss(params):
        total = 0.0
        for cloud, dmap, pix in zip(clouds, dist_to_mask, mask_pixels):
            u, v = project(params, cloud)
            inside = (u >= 0) & (u < W // 2) & (v >= 0) & (v < H // 2)
            if inside.sum() < 50:
                return 1e3
            ui, vi = u[inside].astype(int), v[inside].astype(int)
            d_pred = dmap[vi, ui].mean()  # projected robot points far from the real robot
            proj = np.zeros((H // 2, W // 2), bool)
            proj[vi, ui] = True
            proj = ndimage.binary_dilation(proj, iterations=1)
            d_real = ndimage.distance_transform_edt(~proj)[pix[:, 0], pix[:, 1]].mean()  # real robot not covered
            total += d_pred + d_real
        return total / len(clouds)

    best = None
    for pitch, f in ((0.9, 350.0), (1.1, 300.0), (1.0, 450.0)):
        if True:
            x0 = np.array([0.07, 0.03, 0.45, 0.0, pitch, 0.0, f])
            res = optimize.minimize(loss, x0, method="Powell", options={"maxiter": 4000, "xtol": 1e-4, "ftol": 1e-4})
            print(f"start pitch {pitch} f {f}: loss {res.fun:.2f} px -> {np.round(res.x, 3)}", flush=True)
            if best is None or res.fun < best.fun:
                best = res
    p = best.x
    hfov = np.degrees(2 * np.arctan(W / 2 / p[6]))
    vfov = np.degrees(2 * np.arctan(H / 2 / p[6]))
    result = {"position_torso": p[:3].tolist(), "euler_zyx_rad": p[3:6].tolist(), "focal_px": float(p[6]),
              "image_size": [W, H], "hfov_deg": hfov, "vfov_deg": vfov, "chamfer_px": float(best.fun)}
    json.dump(result, open(out_path, "w"), indent=2)
    result["chamfer_px_full_res"] = 2 * float(best.fun)
    json.dump(result, open(out_path, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
