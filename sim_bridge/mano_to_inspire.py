"""Retarget the ego MANO hands to G1 + Inspire FTP hands (fingers + hand-base target pose), in the same parquet schema as
ego_to_dex3.py, so ik_camera_track.py can run the camera-consistent whole-body IK on it (G1_URDF / PALM_LINKS env).

Fingers follow Unitree's own teleoperation (xr_teleoperate, the pipeline that recorded the G1_WBT_Inspire data):
DexPilot (dex_retargeting) on the Inspire DFX/DFQ hand model from assets/inspire_hand/inspire_hand.yml (wrist + 5 tips),
6 actuated joints -> normalised 0..1 like the teleop's hand command (fingers 0..1.7 rad, thumb pitch 0..0.5,
thumb yaw -0.1..1.3) -> FTP joint angles over the FTP URDF limits, mimic joints filled in from the URDF multipliers.
The human key points are rotated into the robot hand frame by matching a canonical hand frame built the same way on
both (x: wrist -> middle-finger base, z: palm normal from index base - little base, y = z x x).
Hand base pose: rotation from the same canonical frames; translation by weighted least squares so that the robot palm
centre (weight 2) and the 5 robot fingertips (weight 1) land on the human ones.
The input MANO joints are the scaled (world_scale) and surface-projected ones from ego_to_dex3.py, so the DexPilot
scaling is the teleop's 1.2 divided by the world scale.

usage: isaaclab_newton python mano_to_inspire.py <dex3 parquet dir> <out dir> <world scale, e.g. 0.75>
writes <out>/data.parquet (+ inspire_debug.npz: per-frame fingertip / palm-centre errors)"""
import glob
import os
import sys

import numpy as np
import pinocchio as pin
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
import yaml
from dex_retargeting.retargeting_config import RetargetingConfig
from scipy.spatial.transform import Rotation as R

TELEOP = os.environ.get("XR_TELEOP", "/data/repos/unitree_assets/xr_teleoperate") + "/assets"  # unitreerobotics/xr_teleoperate (DexPilot inspire_hand.yml)
G1I = os.environ.get("UNITREE_ROS", "/data/repos/unitree_assets/unitree_ros") + "/robots/g1_description"  # unitreerobotics/unitree_ros
FTP_URDF = f"{G1I}/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
MANO_BASES, MANO_TIPS = [5, 9, 13, 17], [4, 8, 12, 16, 20]  # index/middle/ring/pinky base; thumb..pinky tips
DFQ_ORDER = ["thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal"]
# teleop normalisation ranges (rad) of the 6 DFQ joints -> closure 0 (open) .. 1 (closed)
NORM = {"thumb_proximal_yaw": (-0.1, 1.3), "thumb_proximal_pitch": (0.0, 0.5), "index_proximal": (0.0, 1.7),
        "middle_proximal": (0.0, 1.7), "ring_proximal": (0.0, 1.7), "pinky_proximal": (0.0, 1.7)}
FTP_DRIVE = {"thumb_proximal_yaw": "thumb_1", "thumb_proximal_pitch": "thumb_2", "index_proximal": "index_1",
             "middle_proximal": "middle_1", "ring_proximal": "ring_1", "pinky_proximal": "little_1"}
FTP_FINGERS = ["thumb", "index", "middle", "ring", "little"]


def canon(wrist, bases):
    """3x3, columns = canonical hand axes; bases = index, middle, ring, little finger bases."""
    x = bases[1] - wrist; x /= np.linalg.norm(x)
    z = np.cross(bases[0] - bases[3], x); z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1)


def ftp_hand(side):
    """FTP hand of one side from the G1 URDF: model, data, joint names, mimic table, fingertip points per distal link."""
    model = pin.buildModelFromUrdf(FTP_URDF)
    geom = pin.buildGeomFromUrdf(model, FTP_URDF, pin.GeometryType.VISUAL, package_dirs=[G1I])
    names = [n for n in model.names[1:] if n.startswith(f"{side}_") and any(f in n for f in FTP_FINGERS) and "force" not in n]
    import xml.etree.ElementTree as ET
    mimic = {}
    for j in ET.parse(FTP_URDF).getroot().iter("joint"):
        m = j.find("mimic")
        if m is not None and j.get("name") in names:
            mimic[j.get("name")] = (m.get("joint"), float(m.get("multiplier")), float(m.get("offset", 0)))
    lim = {n: (model.lowerPositionLimit[model.joints[model.getJointId(n)].idx_q], model.upperPositionLimit[model.joints[model.getJointId(n)].idx_q]) for n in names}
    tips = {}  # fingertip = vertex farthest from the distal joint over the distal link and its fixed children (sensor pads)
    for f in FTP_FINGERS:
        jid = model.getJointId(f"{side}_{f}_4_joint" if f == "thumb" else f"{side}_{f}_2_joint")
        v = np.concatenate([trimesh.load(g.meshPath, force="mesh").vertices @ g.placement.rotation.T + g.placement.translation
                            for g in geom.geometryObjects if g.parentJoint == jid])  # in the distal joint frame
        tips[f] = (jid, v[np.argmax(np.linalg.norm(v, axis=1))])
    return model, model.createData(), names, mimic, lim, tips


def ftp_q(model, names, mimic, lim, closure):
    """FTP joint vector (all joints of the G1 model) from the 6 closures; mimic joints from the URDF."""
    q = pin.neutral(model); vals = {}
    for k, c in closure.items():
        n = next(n for n in names if n.endswith(f"_{FTP_DRIVE[k]}_joint")); lo, hi = lim[n]
        vals[n] = lo + float(np.clip(c, 0, 1)) * (hi - lo)
    for _ in range(3):  # chains (thumb_4 mimics thumb_3 mimics thumb_2)
        for n, (src, mul, off) in mimic.items():
            if src in vals:
                vals[n] = float(np.clip(mul * vals[src] + off, *lim[n]))
    for n, v in vals.items():
        q[model.joints[model.getJointId(n)].idx_q] = v
    return q, [vals.get(n, 0.0) for n in names]


def main():
    src, out, wscale = sys.argv[1], sys.argv[2], float(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    tab = pq.read_table(glob.glob(f"{src}/**/*.parquet", recursive=True)[0])
    col = lambda n: tab.column(n)[0].as_py()
    cfg_all = yaml.safe_load(open(f"{TELEOP}/inspire_hand/inspire_hand.yml"))
    RetargetingConfig.set_default_urdf_dir(TELEOP)
    columns, debug = {}, {}
    for side, S in (("right", "R"), ("left", "L")):
        J = np.asarray(col(f"mano_{side}_joints"))  # (T, 21, 3), scaled world
        T = len(J)
        c = cfg_all[side]
        cfg = dict(type="DexPilot", urdf_path=c["urdf_path"], target_joint_names=c["target_joint_names"], wrist_link_name=c["wrist_link_name"],
                   finger_tip_link_names=c["finger_tip_link_names"], scaling_factor=float(os.environ.get("HAND_SCALE", 1.2)) / wscale, low_pass_alpha=1.0)
        retarget = RetargetingConfig.from_dict(cfg).build()
        hi = retarget.optimizer.target_link_human_indices  # default DexPilot pairs for the 21-point MANO layout
        # canonical frame of the DFQ hand (teleop model) in its base frame, zero pose
        rb = retarget.optimizer.robot
        rb.compute_forward_kinematics(np.zeros(rb.dof))
        pos = lambda l: rb.get_link_pose(rb.get_link_index(l))[:3, 3]
        C_dfq = canon(pos(c["wrist_link_name"]), np.stack([pos(f"{S}_{f}_proximal") for f in ("index", "middle", "ring", "pinky")]))
        # FTP hand
        model, data, fnames, mimic, lim, tips = ftp_hand(side)
        base_fid = model.getFrameId(f"{side}_base_link")
        q0 = pin.neutral(model); pin.framesForwardKinematics(model, data, q0)
        B0inv = data.oMf[base_fid].inverse()
        fb = lambda f: (B0inv * data.oMi[model.getJointId(f"{side}_{f}_1_joint")]).translation
        C_ftp = canon(np.zeros(3), np.stack([fb(f) for f in ("index", "middle", "ring", "little")]))
        jn = list(retarget.joint_names); qs, fing, wp_, wq_, e_tip, e_pc = [], [], [], [], [], []
        for t in range(T):
            j = J[t]; C_h = canon(j[0], j[MANO_BASES])
            jr = (j - j[0]) @ (C_dfq @ C_h.T).T  # human key points in the DFQ base frame
            qd = retarget.retarget(jr[hi[1]] - jr[hi[0]])
            closure = {k: (qd[jn.index(f"{S}_{k}_joint")] - NORM[k][0]) / (NORM[k][1] - NORM[k][0]) for k in DFQ_ORDER}
            q, fv = ftp_q(model, fnames, mimic, lim, closure)
            pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
            Binv = data.oMf[base_fid].inverse()
            r_tips = np.stack([(Binv * data.oMi[tips[f][0]]).act(tips[f][1]) for f in FTP_FINGERS])
            r_pc = np.mean([(Binv * data.oMi[model.getJointId(f"{side}_{f}_1_joint")]).translation for f in ("index", "middle", "ring", "little")], 0)
            Rw = C_h @ C_ftp.T  # hand base orientation in the world
            h_pts = np.concatenate([j[MANO_BASES].mean(0, keepdims=True), j[MANO_TIPS]]); r_pts = np.concatenate([r_pc[None], r_tips]) @ Rw.T
            w = np.array([2.0, 1, 1, 1, 1, 1])[:, None]
            p = ((h_pts - r_pts) * w).sum(0) / w.sum()
            e = np.linalg.norm(r_pts + p - h_pts, axis=1); e_pc.append(e[0]); e_tip.append(e[1:])
            fing.append(fv); wp_.append(p); wq_.append(R.from_matrix(Rw).as_quat(scalar_first=True)); qs.append([closure[k] for k in DFQ_ORDER])
        e_tip = np.array(e_tip)
        print(f"{side}: fingertip error to the human tips median {np.median(e_tip) * 100:.1f} cm (per finger {np.round(np.median(e_tip, 0) * 100, 1)}), "
              f"palm centre {np.median(e_pc) * 100:.1f} cm; closure mean {np.round(np.mean(qs, 0), 2)} (order {DFQ_ORDER})")
        columns[f"{side}_robot_finger_joint_names"] = [fnames]
        columns[f"robot_{side}_finger_joints"] = [np.array(fing).tolist()]
        columns[f"robot_{side}_wrist_position"] = [np.array(wp_).tolist()]
        columns[f"robot_{side}_wrist_wxyz"] = [np.array(wq_).tolist()]
        columns[f"mano_{side}_joints"] = [J.tolist()]
        debug.update({f"{side}_tip_err": e_tip, f"{side}_pc_err": np.array(e_pc), f"{side}_closure": np.array(qs)})
    for n in ("object_body_position", "object_body_wxyz", "object_mesh_paths", "object_urdf_paths"):
        columns[n] = [col(n)]
    pq.write_table(pa.table(columns), f"{out}/data.parquet")
    np.savez(f"{out}/inspire_debug.npz", **debug)
    print("wrote", f"{out}/data.parquet")


if __name__ == "__main__":
    main()
