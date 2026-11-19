"""11-18: a stand-in for the C++ gear_sonic_deploy process in protocol tests of g1_client.py (no physics, no decoder): it speaks
the same sockets and formats (wire.py): SUB connects to the client's action port (topics command / pose), PUB sends g1_debug at
50 Hz once a `start` command arrived, plus a JPEG camera stream (zmqjpeg) from a video. In PLANNER mode it reports the default
standing pose and a fixed "planner" token; in POSE mode it reports the last token it received (held, as the real controller)
and joints drifting to the task's ready pose. Everything received is logged with times; `--report` prints the checks:
no pose message without an accepted instruction, message rate, value ranges, mode switches.
usage: fake_robot.py --config <json> --seconds 40 --log <npz> [--video <mp4>] [--action-port 5556 --state-port 5557 --camera-port 5558]"""
import argparse, json, time
import numpy as np, zmq, cv2
import wire

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--seconds", type=float, default=40); ap.add_argument("--log", required=True); ap.add_argument("--video", default="")
ap.add_argument("--stuck", action="store_true"); ap.add_argument("--frame-size", default="1280x720");  # 11-19: e.g. 640x480 to test that the client refuses a 4:3 stream
ap.add_argument("--action-port", type=int, default=wire.PORT_ACTION); ap.add_argument("--state-port", type=int, default=wire.PORT_STATE); ap.add_argument("--camera-port", type=int, default=5558)
a = ap.parse_args(); cfg = json.load(open(a.config)); T0 = next(iter(cfg["tasks"].values()))
ctx = zmq.Context.instance(); sub = ctx.socket(zmq.SUB); sub.setsockopt(zmq.SUBSCRIBE, b""); sub.connect(f"tcp://localhost:{a.action_port}")
pub = ctx.socket(zmq.PUB); pub.bind(f"tcp://*:{a.state_port}"); cam = ctx.socket(zmq.PUB); cam.bind(f"tcp://*:{a.camera_port}")
FS = tuple(int(v) for v in a.frame_size.split("x")); frames = []
if a.video:
    v = cv2.VideoCapture(a.video)
    while True:
        ok, f = v.read()
        if not ok: break
        frames.append(cv2.imencode(".jpg", cv2.resize(f, FS))[1].tobytes())  # 11-19: the robot's camera service must send the 16:9 stream
if not frames: frames = [cv2.imencode(".jpg", np.full((FS[1], FS[0], 3), 128, np.uint8))[1].tobytes()]
rng = np.random.default_rng(1); planner_token = (rng.integers(-5, 5, 64) / 16).astype(np.float32)
q_default = np.asarray(cfg["default_body"]); q_ready = q_default if a.stuck else np.asarray(T0.get("start_body", T0["ready_body"])); q = q_default.copy()  # SONIC order; --stuck: the arms never leave the standing pose
lh = np.zeros(7); rh = np.zeros(7); tok = planner_token.copy(); running = False; mode = "OFF"; log = dict(t=[], kind=[], token=[], left=[], right=[], cmd=[])
t0 = time.monotonic(); n = 0; tgt_l = lh.copy(); tgt_r = rh.copy()
while time.monotonic() - t0 < a.seconds:
    now = time.monotonic() - t0
    while True:
        try: msg = sub.recv(zmq.NOBLOCK)
        except zmq.Again: break
        topic, d = wire.unpack(msg)
        if topic == "command":
            s, p, pl = int(d["start"][0]), int(d["stop"][0]), int(d["planner"][0]); log["t"].append(now); log["kind"].append("command"); log["cmd"].append((s, p, pl))
            log["token"].append(np.zeros(64, np.float32)); log["left"].append(np.zeros(7, np.float32)); log["right"].append(np.zeros(7, np.float32))
            if p: running = False; mode = "OFF"
            elif s: running = True; mode = "PLANNER" if pl else "POSE"
            if pl: tok = planner_token.copy()
        else:
            log["t"].append(now); log["kind"].append("pose"); log["cmd"].append((0, 0, 0)); log["token"].append(d["token_state"][0].copy()); log["left"].append(d["left_hand_joints"][0].copy()); log["right"].append(d["right_hand_joints"][0].copy())
            if mode == "POSE": tok = d["token_state"][0].copy(); tgt_l, tgt_r = d["left_hand_joints"][0].astype(float), d["right_hand_joints"][0].astype(float)
    if running:
        goal = q_ready if mode == "POSE" else q_default; q = q + np.clip(goal - q, -0.02, 0.02)
        lh = lh + np.clip(tgt_l - lh, -0.1, 0.1); rh = rh + np.clip(tgt_r - rh, -0.1, 0.1); n += 1
        q_mj = np.zeros(29); q_mj[:] = [q[wire.SONIC_BODY.index(j)] for j in wire.MUJOCO_BODY]
        pub.send(wire.pack_state(q_mj, lh, rh, tok, n))
    if int(now * 30) != int((now - 0.02) * 30): cam.send(b"cam" + frames[int(now * 12.5) % len(frames)])
    time.sleep(max(0.0, 0.02 - ((time.monotonic() - t0) - now)))
np.savez_compressed(a.log, t=np.asarray(log["t"]), kind=np.asarray(log["kind"]), token=np.asarray(log["token"]).reshape(-1, 64), left=np.asarray(log["left"]).reshape(-1, 7), right=np.asarray(log["right"]).reshape(-1, 7),
                    cmd=np.asarray(log["cmd"]).reshape(-1, 3), planner_token=planner_token)
print("fake robot: logged", len(log["t"]), "messages ->", a.log)
