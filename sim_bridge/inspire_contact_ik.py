"""Inspire FTP hand retarget with absolute targets and box contact, after CHORD's Sharpa/Dex3 hand IK: the fingertips,
finger bases and palm orientation follow the human MANO hand where it is in the world (the MANO points are already
pushed out of the box by ego_to_dex3.py --surface_margin), instead of only the hand shape (DexPilot, mano_to_inspire.py,
which left the thumb 2.4-2.8 cm and the index 1.1-1.6 cm off the box while the human touched it).
Unlike Sharpa's 22 free joints, the Inspire has 6 motors driving 12 joints, so the unknowns per hand and frame are the
hand-base pose (6) and the 6 closures (0 open .. 1 closed; FTP angles and mimic joints from mano_to_inspire.ftp_q).
Least squares per frame (warm start from the previous frame; frame 0 from mano_to_inspire.py), residuals:
  fingertips -> human tips (w 1), finger bases -> human MCPs (w 0.3, thumb CMC 0.1), palm orientation (canonical hand
  frame, 0.03 m/rad), contact: for each finger whose human tip is within CONTACT of the box, the closest point of the
  robot's distal segment on the box surface (w 1), penetration of any hand surface sample deeper than 2 mm (w 3),
  continuity with the previous frame (base 0.2, rotation 0.01 m/rad, closures 0.01).
The box is CHORD's collision proxy: the minimum-volume oriented box of the (scaled) object mesh (object_collision.py).
usage: isaaclab_newton python inspire_contact_ik.py <mano_to_inspire out dir> <out dir> <world scale>
writes <out>/data.parquet in the same schema"""
import glob
import os
import sys

import numpy as np
import pinocchio as pin
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.environ.get("CHORD", "/data/repos/chord") + "/robotic_grounding/source/robotic_grounding")  # CHORD repo (env CHORD)
from robotic_grounding.retarget.object_collision import OrientedBox  # noqa: E402

import mano_to_inspire as MI  # noqa: E402

CONTACT = float(os.environ.get("CONTACT", 0.015))  # human tip (joint centre) to box surface, m
W_TIP, W_MCP, W_CMC, W_ORI, W_CON, W_PEN = 1.0, 0.3, 0.1, 0.03, 1.0, 3.0
W_ST, W_SR, W_SC = 0.2, 0.01, 0.01
DRIVE = ["thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal"]
FINGERS = MI.FTP_FINGERS  # thumb index middle ring little


class Hand:
    """FTP hand of one side: link transforms and surface samples in the hand-base frame as a function of the closures."""

    def __init__(self, side):
        self.side = side
        self.model, self.data, self.names, self.mimic, self.lim, self.tips = MI.ftp_hand(side)
        m = self.model
        self.base = m.getFrameId(f"{side}_base_link")
        geom = pin.buildGeomFromUrdf(m, MI.FTP_URDF, pin.GeometryType.VISUAL, package_dirs=[MI.G1I])
        self.jid = {f: m.getJointId(f"{side}_{f}_1_joint") for f in FINGERS}
        self.distal = {f: m.getJointId(f"{side}_{f}_4_joint" if f == "thumb" else f"{side}_{f}_2_joint") for f in FINGERS}
        finger_joints = {f: [m.getJointId(n) for n in self.names if f"_{f}_" in n] for f in FINGERS}
        rng = np.random.default_rng(0)
        self.samples = []  # (joint id, points in that joint frame, finger or "palm", is distal)
        base_jid = m.frames[self.base].parentJoint
        for g in geom.geometryObjects:
            f = next((f for f in FINGERS if g.parentJoint in finger_joints[f]), None)
            if f is None and not (g.parentJoint == base_jid and g.name.startswith((f"{side}_base_link", f"{side}_palm"))):
                continue
            v = trimesh.load(g.meshPath, force="mesh").vertices @ g.placement.rotation.T + g.placement.translation
            v = v[rng.choice(len(v), min(len(v), 40 if f else 120), replace=False)]
            self.samples.append((g.parentJoint, v, f or "palm", f is not None and g.parentJoint == self.distal[f]))

    def fk(self, closure):
        """closures (6, DRIVE order) -> tips (5,3), finger bases (5,3), surface points (N,3), owner list, base-frame."""
        q, fv = MI.ftp_q(self.model, self.names, self.mimic, self.lim, dict(zip(DRIVE, closure)))
        pin.forwardKinematics(self.model, self.data, q); pin.updateFramePlacements(self.model, self.data)
        Binv = self.data.oMf[self.base].inverse()
        M = lambda j: Binv * self.data.oMi[j]
        tips = np.stack([M(self.tips[f][0]).act(self.tips[f][1]) for f in FINGERS])
        bases = np.stack([M(self.jid[f]).translation for f in FINGERS])
        pts = [(lambda T: v @ T.rotation.T + T.translation)(M(j)) for j, v, _, _ in self.samples]
        return tips, bases, pts, fv


def solve(side, J, base_p0, base_q0, clos0, op, oq, box):
    hand = Hand(side)
    T = len(J)
    _, b0, _, _ = hand.fk(np.zeros(6))
    C_r = MI.canon(np.zeros(3), b0[1:])  # canonical frame of the robot hand (index..little bases), base frame
    owner = [(f, dist) for _, _, f, dist in hand.samples]
    out_p, out_q, out_f, out_c, stats = [], [], [], [], []
    Rb, tb, cl = R.from_quat(base_q0[0], scalar_first=True), np.asarray(base_p0[0]), np.clip(clos0[0], 0, 1)
    for t in range(T):
        j = J[t]
        C_h = MI.canon(j[0], j[MI.MANO_BASES]); R_tgt = R.from_matrix(C_h @ C_r.T)
        h_tips, h_mcp = j[MI.MANO_TIPS], j[[1, 5, 9, 13, 17]]
        Ro, po = R.from_quat(oq[t], scalar_first=True), op[t]
        sd = lambda P: box.signed_distance(Ro.inv().apply(P - po))
        contact = sd(h_tips) < CONTACT
        Rprev, tprev, cprev = Rb, tb, cl

        def res(x):
            Rw = R.from_rotvec(x[3:6]) * Rprev; tw = tprev + x[:3]
            tips, bases, pts, _ = hand.fk(x[6:])
            W = lambda P: Rw.apply(P) + tw
            r = [W_TIP * (W(tips) - h_tips).ravel(),
                 W_MCP * (W(bases[1:]) - h_mcp[1:]).ravel(), W_CMC * (W(bases[:1]) - h_mcp[:1]).ravel(),
                 W_ORI * (Rw * R_tgt.inv()).as_rotvec()]
            P = W(np.concatenate(pts)); d = sd(P); n = np.cumsum([0] + [len(p) for p in pts])
            con = np.zeros(5)
            for k, f in enumerate(FINGERS):
                if contact[k]:
                    con[k] = min(d[n[i]:n[i + 1]].min() for i, (g, dist) in enumerate(owner) if g == f and dist)
            r += [W_CON * con, W_PEN * np.maximum(0.0, -d - 0.002),
                  W_ST * x[:3], W_SR * x[3:6], W_SC * (x[6:] - cprev)]
            return np.concatenate(r)

        x0 = np.concatenate([np.zeros(6), cprev])
        lo = np.r_[np.full(6, -np.inf), np.zeros(6)]; hi = np.r_[np.full(6, np.inf), np.ones(6)]
        sol = least_squares(res, x0, bounds=(lo, hi), x_scale=np.r_[np.full(3, 0.01), np.full(3, 0.1), np.full(6, 0.1)],
                            max_nfev=200 if t else 600)
        x = sol.x; Rb = R.from_rotvec(x[3:6]) * Rprev; tb = tprev + x[:3]; cl = x[6:]
        tips, bases, pts, fv = hand.fk(cl)
        P = Rb.apply(np.concatenate(pts)) + tb; d = sd(P); dt = sd(Rb.apply(tips) + tb)
        stats.append((np.linalg.norm(Rb.apply(tips) + tb - h_tips, axis=1), contact, d.min(),
                      np.array([min(d[np.cumsum([0] + [len(p) for p in pts])[i]:np.cumsum([0] + [len(p) for p in pts])[i + 1]].min()
                                    for i, (g, dist) in enumerate(owner) if g == f) for f in FINGERS])))
        out_p.append(tb); out_q.append(Rb.as_quat(scalar_first=True)); out_f.append(fv); out_c.append(cl)
    e = np.array([s[0] for s in stats]); c = np.array([s[1] for s in stats]); fd = np.array([s[3] for s in stats])
    print(f"{side}: tip error median {np.median(e) * 100:.1f} cm (per finger {np.round(np.median(e, 0) * 100, 1)}); "
          f"in human-contact frames robot finger to box median {np.round([np.median(fd[c[:, k], k]) * 100 if c[:, k].any() else np.nan for k in range(5)], 1)} cm "
          f"(contact frames per finger {c.sum(0)}); deepest penetration {min(s[2] for s in stats) * 100:.1f} cm; closure mean {np.round(np.mean(out_c, 0), 2)}")
    return hand.names, np.array(out_f), np.array(out_p), np.array(out_q)


def main():
    src, out, wscale = sys.argv[1], sys.argv[2], float(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    tab = pq.read_table(glob.glob(f"{src}/**/*.parquet", recursive=True)[0]); col = lambda n: tab.column(n)[0].as_py()
    op, oq = np.asarray(col("object_body_position"))[:, 0], np.asarray(col("object_body_wxyz"))[:, 0]
    mesh_path = col("object_mesh_paths")[0].replace("/workspace/video_to_data", os.environ.get("CHORD", "/data/repos/chord"))
    box = OrientedBox.from_points(trimesh.load(mesh_path, force="mesh").vertices * wscale)
    print("box (scaled) extents", np.round(2 * box.half_extents, 3))
    columns = {n: [col(n)] for n in ("object_body_position", "object_body_wxyz", "object_mesh_paths", "object_urdf_paths")}
    for side in ("right", "left"):
        J = np.asarray(col(f"mano_{side}_joints"))
        F0 = np.asarray(col(f"robot_{side}_finger_joints")); names0 = col(f"{side}_robot_finger_joint_names")
        # frame-0 closures from the mano_to_inspire.py FTP angles of the drive joints
        lim = MI.ftp_hand(side)[4]
        clos0 = np.array([[F0[t, names0.index(n)] / lim[n][1] for n in
                           (f"{side}_{MI.FTP_DRIVE[k]}_joint" for k in DRIVE)] for t in range(len(F0))])
        names, fv, p, q = solve(side, J, col(f"robot_{side}_wrist_position"), col(f"robot_{side}_wrist_wxyz"), clos0, op, oq, box)
        columns[f"{side}_robot_finger_joint_names"] = [names]
        columns[f"robot_{side}_finger_joints"] = [fv.tolist()]
        columns[f"robot_{side}_wrist_position"] = [p.tolist()]
        columns[f"robot_{side}_wrist_wxyz"] = [q.tolist()]
        columns[f"mano_{side}_joints"] = [J.tolist()]
    pq.write_table(pa.table(columns), f"{out}/data.parquet")
    print("wrote", f"{out}/data.parquet")


if __name__ == "__main__":
    main()
