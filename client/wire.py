"""11-18: wire formats and joint orders of NVIDIA's GEAR-SONIC deployment (GR00T-WholeBodyControl, gear_sonic_deploy), as the
real-robot client needs them. Facts and their sources (repo /data/repos/gr00t_wbc):
  out  topic "pose", protocol v4: [b"pose"][1280-byte null-padded JSON header][raw little-endian fields]; fields token_state
       f32 [1, 64], frame_index i64 [1], left_hand_joints f32 [1, 7], right_hand_joints f32 [1, 7]
       (gear_sonic/utils/teleop/zmq/zmq_planner_sender.py pack_pose_message; C++ input_interface/zmq_endpoint_interface.hpp).
       The client BINDS the PUB socket (port 5556); the C++ side connects to it (--zmq-host).
  out  topic "command": u8 start, stop, planner (1 = the built-in planner stands the robot, 0 = streamed tokens).
  in   topic "g1_debug": [b"g1_debug"][msgpack map], 50 Hz, only while the control loop runs (zmq_output_handler.hpp):
       body_q [29] rad in MuJoCo = Unitree motor order, left_hand_q / right_hand_q [7] in Dex3 motor order, base_quat wxyz,
       token_state [64] = the token the decoder got that tick.
The C++ side does not validate tokens (no clamp, no NaN check) and holds the last token and hand targets forever when the
stream stops, so every check lives in the client. `python wire.py --check <gr00t_wbc dir>` compares the packers byte by byte
with the official ones."""
from __future__ import annotations

import json, struct
import numpy as np

HEADER_SIZE = 1280
TOPIC_POSE, TOPIC_COMMAND, TOPIC_STATE = b"pose", b"command", b"g1_debug"
PORT_ACTION, PORT_STATE, PORT_CAMERA = 5556, 5557, 5555

# body joints, MuJoCo / Unitree motor order (gear_sonic_deploy policy_parameters.hpp:210-240) = the order of g1_debug body_q
MUJOCO_BODY = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")] + \
    ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + \
    [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
# body joints, IsaacLab order = SONIC's tensor order = the student's state order (flash_chord sonic.py SONIC_POLICY_JOINT_NAMES)
SONIC_BODY = ["left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
              "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
              "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
              "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
              "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
              "left_wrist_yaw_joint", "right_wrist_yaw_joint"]
ISAACLAB_TO_MUJOCO = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]  # policy_parameters.hpp:99: MuJoCo index i -> IsaacLab index
assert [SONIC_BODY[k] for k in ISAACLAB_TO_MUJOCO] == MUJOCO_BODY
MJ_TO_SONIC = [MUJOCO_BODY.index(j) for j in SONIC_BODY]  # q_sonic = q_mujoco[MJ_TO_SONIC]

# Dex3 motor order, index 0..6 of rt/dex3/<side>/cmd. Thumb = 0..2 (dex3_hands.hpp). Which of 3-4 / 5-6 is the index finger is
# NOT stated in the C++ code and the repo's own files disagree; the default below is Unitree's SDK order (left: thumb, middle,
# index; right: thumb, index, middle; gear_sonic_deploy/g1/g1_29dof_with_hand.xml). CHECK ON THE ROBOT (g1_client.py --hand-check).
DEX3_ORDER = {"unitree": dict(left=["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
                              right=["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]),
              "middle_first": dict(left=["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
                                   right=["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]),
              "index_first": dict(left=["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"],
                                  right=["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"])}
# joint limits by joint name (dex3_hands.hpp:460-463, which lists them by motor index in the "unitree" order)
DEX3_LIMITS = {"left": dict(thumb_0=(-1.05, 1.05), thumb_1=(-0.724, 1.05), thumb_2=(0.0, 1.75), middle_0=(-1.57, 0.0), middle_1=(-1.75, 0.0), index_0=(-1.57, 0.0), index_1=(-1.75, 0.0)),
               "right": dict(thumb_0=(-1.05, 1.05), thumb_1=(-1.05, 0.742), thumb_2=(-1.75, 0.0), index_0=(0.0, 1.57), index_1=(0.0, 1.75), middle_0=(0.0, 1.57), middle_1=(0.0, 1.75))}


def hand_names(order: str, side: str) -> list:
    return [f"{side}_hand_{j}_joint" for j in DEX3_ORDER[order][side]]


def _header(fields: list, version: int) -> bytes:
    h = json.dumps({"v": version, "endian": "le", "count": 1, "fields": fields}, separators=(",", ":")).encode()
    if len(h) > HEADER_SIZE: raise ValueError("header too large")
    return h.ljust(HEADER_SIZE, b"\x00")


def pack_pose(token, frame_index: int, left_hand, right_hand) -> bytes:
    """One protocol-v4 pose message: token [64], Dex3 targets [7] + [7] (rad, motor order)."""
    arrs = [("token_state", np.asarray(token, np.float32).reshape(1, 64)), ("frame_index", np.asarray([frame_index], np.int64)),
            ("left_hand_joints", np.asarray(left_hand, np.float32).reshape(1, 7)), ("right_hand_joints", np.asarray(right_hand, np.float32).reshape(1, 7))]
    fields = [{"name": n, "dtype": "i64" if a.dtype == np.int64 else "f32", "shape": list(a.shape)} for n, a in arrs]
    return TOPIC_POSE + _header(fields, 4) + b"".join(np.ascontiguousarray(a).tobytes() for _, a in arrs)


def pack_command(start: bool, stop: bool, planner: bool) -> bytes:
    fields = [{"name": n, "dtype": "u8", "shape": [1]} for n in ("start", "stop", "planner")]
    return TOPIC_COMMAND + _header(fields, 1) + struct.pack("BBB", int(bool(start)), int(bool(stop)), int(bool(planner)))


def unpack(msg: bytes):
    """A pose / command message -> (topic, {field: array}) (what the C++ side reads; used by fake_robot.py and the tests)."""
    topic = TOPIC_POSE if msg.startswith(TOPIC_POSE) else TOPIC_COMMAND if msg.startswith(TOPIC_COMMAND) else None
    if topic is None: raise ValueError("unknown topic")
    head = json.loads(msg[len(topic):len(topic) + HEADER_SIZE].rstrip(b"\x00")); off = len(topic) + HEADER_SIZE; out = {}
    for f in head["fields"]:
        dt = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8", "u8": "u1", "bool": "?"}[f["dtype"]]; n = int(np.prod(f["shape"]))
        out[f["name"]] = np.frombuffer(msg, dt, n, off).reshape(f["shape"]); off += n * np.dtype(dt).itemsize
    return topic.decode(), out


def unpack_state(raw: bytes) -> dict:
    import msgpack
    m = msgpack.unpackb(raw[len(TOPIC_STATE):], raw=False)
    return {k: (np.asarray(v) if isinstance(v, (list, tuple)) else v) for k, v in m.items()}


def pack_state(body_q_mujoco, left_hand_q, right_hand_q, token, index: int, base_quat=(1.0, 0.0, 0.0, 0.0)) -> bytes:
    """The fields of g1_debug the client reads (fake_robot.py)."""
    import msgpack
    f = lambda a: [float(x) for x in np.asarray(a).ravel()]
    return TOPIC_STATE + msgpack.packb(dict(control_loop_type="cpp", index=int(index), base_quat=f(base_quat), body_q=f(body_q_mujoco), body_dq=[0.0] * 29,
                                            left_hand_q=f(left_hand_q), right_hand_q=f(right_hand_q), token_state=f(token)))


if __name__ == "__main__":
    import importlib.util, sys
    if len(sys.argv) > 2 and sys.argv[1] == "--check":
        spec = importlib.util.spec_from_file_location("official", f"{sys.argv[2]}/gear_sonic/utils/teleop/zmq/zmq_planner_sender.py"); off = importlib.util.module_from_spec(spec); spec.loader.exec_module(off)
        rng = np.random.default_rng(0); tok = rng.uniform(-0.5, 0.5, 64).astype(np.float32); lh = rng.uniform(-1, 0, 7).astype(np.float32); rh = rng.uniform(0, 1, 7).astype(np.float32)
        ref = off.pack_pose_message({"token_state": tok.reshape(1, 64), "frame_index": np.array([7], np.int64), "left_hand_joints": lh.reshape(1, 7), "right_hand_joints": rh.reshape(1, 7)}, topic="pose", version=4)
        assert ref == pack_pose(tok, 7, lh, rh), "pose message differs from the official packer"
        for s, p, pl in ((1, 0, 1), (1, 0, 0), (0, 1, 0)):
            assert off.build_command_message(bool(s), bool(p), bool(pl)) == pack_command(s, p, pl), "command message differs"
        t, d = unpack(ref); assert t == "pose" and np.array_equal(d["token_state"][0], tok) and np.array_equal(d["right_hand_joints"][0], rh) and int(d["frame_index"][0]) == 7
        t, d = unpack(pack_command(1, 0, 0)); assert t == "command" and (int(d["start"][0]), int(d["stop"][0]), int(d["planner"][0])) == (1, 0, 0)
        st = unpack_state(pack_state(np.arange(29) * 0.01, np.zeros(7), np.ones(7), tok, 3)); assert st["body_q"].shape == (29,) and st["index"] == 3
        print("wire check passed: pose and command messages are byte-identical to the official packers;", len(ref), "bytes a pose message")
        print("state order check: body_q[MJ_TO_SONIC] gives", [n.replace("_joint", "") for n in SONIC_BODY[:6]], "...")
