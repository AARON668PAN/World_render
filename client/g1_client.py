#!/usr/bin/env python
"""11-18: real-robot client for the vision student on a Unitree G1 + Dex3 hands behind NVIDIA's official GEAR-SONIC controller
(gear_sonic_deploy, `./deploy.sh --input-type zmq_manager real` with the DEFAULT `policy/release` checkpoint: the simulator's
SONIC is that one; the docs' sonic_v1_1 has another token space and must not be used with this student).
It replaces gear_sonic/scripts/run_vla_inference.py: the same wire protocol (wire.py), our student instead of GR00T.

The robot moves only on a typed command (gate.py). What is sent, by state:
  STANDBY    nothing. The controller is in its PLANNER mode and stands by itself. The student is not connected to the robot.
  `ready`    (11-18) the ONLY way from standing to the start pose: switch the controller to streamed tokens, blend from its
             current token into the precomputed approach tokens, then HOLD the ready token. The approach goes AROUND THE SIDES
             (standing -> arms out to the sides -> hands high over the table -> start pose, config approach_path), because a
             straight move would bring the hands up into the table top. It is made for the robot standing with the table's near
             edge 22 cm ahead of the pelvis centre (12 cm further back than in the user's video; closer, the controller's own
             standing pose already has the wrists at the table edge).
  HOLDING    the ready token and finger pose, every tick, until an instruction or `planner`.
  RUNNING    after an accepted instruction (one of the config's sentences), accepted only while HOLDING:
             approach  blend from the held token to this task's ready token (--blend-in-s), fingers to the start pose
             settle    the ready token for --settle-s, the first plan is computed from here
             act       the student: a new plan every 20 ticks (0.4 s) from the current camera image + joint angles, token
                       held 4 ticks, finger targets interpolated, latency-compensated hand-over as in the simulator exam
  RETURNING  when the episode time is up, on `stop`, or when camera / robot state go stale: blend back to the ready token,
             then HOLDING again (--after ready, the default). One instruction = one episode.
  `planner`  (asked twice) the arms go back down the way they came (config retreat_tokens), then the robot is handed back to the
             controller's standing loop. Never straight to the planner from the start pose: it would lower the hands onto the table.
             `quit` is refused while holding (`quit!` leaves the controller on the held token: it keeps the pose).
  `hold`     after a client restart while the robot still holds a pose: take over streaming the controller's current token.
Every outgoing message is checked here (the C++ side checks nothing): finite numbers, token within +-0.6, finger targets
inside the Dex3 joint limits and at most --finger-step rad per tick. If this program dies the controller keeps the last token:
the operator's emergency stop is the `O` key in the deploy.sh terminal (documented by NVIDIA; the robot goes limp).

Operator commands (type + Enter): start | ready [sentence] | <instruction sentence> | stop | planner (twice) | hold | status | estop | quit
usage: g1_client.py --config config_orange_plate.json --model <student pretrained_model dir> --robot-host 192.168.123.164
         [--camera official:192.168.123.164:5555 | cv:0 | zmqjpeg:host:port] [--hand-order unitree] [--dry-run] [--hand-check]
(python with numpy, pyzmq, msgpack, opencv; the student runs in its own process: --student-python <lerobot env python>)"""
from __future__ import annotations

import argparse, json, os, queue, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import wire  # noqa: E402
from gate import IDLE, RETURNING, RUNNING, InstructionGate  # noqa: E402

TICK = 0.02; STEP_TICKS = 4; EVERY = 20; TOKEN_LIMIT = 0.6
PRE_LIMIT = 0.75  # 11-18: the precomputed approach / retreat tokens come from SONIC's own encoder, which gives -0.625 with the arms straight out; only the student's output is held to +-0.6


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Student:
    """The inference process (runtime_server.py or a stand-in), one request at a time on a worker thread."""

    def __init__(self, cmd: list, shm: str):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.info = json.loads(self.p.stdout.readline()); self.shm = shm; self.busy = False; self._res = None; self._lock = threading.Lock()

    def request(self, tick, s_prev, s_cur, im_prev, im_cur, task, first, extra, reset=False) -> None:
        self.busy = True
        def work():
            t0 = time.monotonic()
            try:
                np.save(self.shm + "_cur.npy", im_cur[None]); np.save(self.shm + "_prev.npy", im_prev[None])
                self.p.stdin.write(json.dumps(dict(batch_state=[np.round(s_cur, 6).tolist()], batch_state_prev=[np.round(s_prev, 6).tolist()], images=self.shm + "_cur.npy",
                                                   images_prev=self.shm + "_prev.npy", first=first, extra=extra, rtc=1, blend=1, task=task, reset=int(reset))) + "\n")
                a = np.asarray(json.loads(self.p.stdout.readline())["actions"], np.float32)[0]
                with self._lock: self._res = (tick, a, time.monotonic() - t0)
            except Exception as e:  # a dead student ends the episode through the starvation check
                say(f"student error: {e!r}")
            self.busy = False
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        with self._lock:
            r, self._res = self._res, None
        return r

    def drop(self) -> None:
        with self._lock: self._res = None


class Camera:
    """Latest head-camera frame as RGB uint8 + its age (local receive time). kinds: official:<host>:<port> (NVIDIA's composed
    camera server), cv:<index or url> (OpenCV), zmqjpeg:<host>:<port> (tests: b'cam' + JPEG)."""

    def __init__(self, spec: str, gr00t_wbc: str | None):
        kind, _, rest = spec.partition(":"); self.kind = kind; self.frame = None; self.t = 0.0; self._n = -1
        if kind == "official":
            if gr00t_wbc: sys.path.insert(0, gr00t_wbc)
            from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
            host, _, port = rest.partition(":"); self.dev = ComposedCameraClientSensor(server_ip=host, port=int(port or wire.PORT_CAMERA))
        elif kind == "cv":
            import cv2
            self.dev = cv2.VideoCapture(int(rest) if rest.isdigit() else rest); threading.Thread(target=self._cv_loop, daemon=True).start()
        elif kind == "zmqjpeg":
            import zmq
            self.dev = zmq.Context.instance().socket(zmq.SUB); self.dev.setsockopt(zmq.SUBSCRIBE, b"cam"); self.dev.setsockopt(zmq.CONFLATE, 1); self.dev.connect(f"tcp://{rest}")
        else:
            raise ValueError(f"unknown camera '{spec}'")

    def _cv_loop(self):
        import cv2
        while True:
            ok, bgr = self.dev.read()
            if ok: self.frame, self.t = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), time.monotonic()
            else: time.sleep(0.01)

    def read(self):
        if self.kind == "official":
            m = self.dev.read()
            if m is not None and self.dev.idx != self._n: self._n = self.dev.idx; self.frame, self.t = np.asarray(m["images"]["ego_view"]), time.monotonic()
        elif self.kind == "zmqjpeg":
            import cv2, zmq
            try:
                raw = self.dev.recv(zmq.NOBLOCK); self.frame, self.t = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw[3:], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), time.monotonic()
            except zmq.Again: pass
        return self.frame, (time.monotonic() - self.t if self.frame is not None else 1e9)


def to_student_image(rgb: np.ndarray, hw=(168, 224)) -> np.ndarray:
    """Centre crop to the student's aspect ratio and resize (area filter). The client refuses to act when the stream's aspect is not the student's (see cam_shape_ok)."""
    import cv2
    h, w = rgb.shape[:2]; H, W = hw
    if w * H > h * W: x = (w - h * W // H) // 2; rgb = rgb[:, x:x + h * W // H]
    elif w * H < h * W: y = (h - w * H // W) // 2; rgb = rgb[y:y + w * H // W]
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA) if rgb.shape[:2] != (H, W) else np.ascontiguousarray(rgb)


def upsample(c: np.ndarray, ticks: int = STEP_TICKS, hold: int = 64) -> np.ndarray:
    """[n, A] dataset steps -> [(n - 1) * ticks, A] control ticks: token held, finger targets linear (as student_runtime.upsample)."""
    n = len(c) - 1; out = np.repeat(c[:n], ticks, 0).astype(np.float32)
    for i in range(n):
        for j in range(ticks): out[i * ticks + j, hold:] = c[i, hold:] + (c[i + 1, hold:] - c[i, hold:]) * j / ticks
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(HERE), "config", "config_orange_plate.json")); ap.add_argument("--model", default=os.path.join(os.path.dirname(HERE), "student")); ap.add_argument("--student-cmd", default="", help="override the inference process (tests)")
    ap.add_argument("--student-python", default=sys.executable)
    ap.add_argument("--robot-host", default="localhost"); ap.add_argument("--state-port", type=int, default=wire.PORT_STATE); ap.add_argument("--action-port", type=int, default=wire.PORT_ACTION)
    ap.add_argument("--camera", default="official:localhost:5555"); ap.add_argument("--gr00t-wbc", default=os.environ.get("GR00T_WBC", ""), help="checkout of NVIDIA's GR00T-WholeBodyControl (only for --camera official:...)")
    ap.add_argument("--hand-order", default="unitree", choices=list(wire.DEX3_ORDER)); ap.add_argument("--after", default="ready", choices=["planner", "ready"], help="where to wait after an episode (11-18: ready = keep holding the start pose; planner lowers the arms, only without a table)")
    ap.add_argument("--blend-in-s", type=float, default=0.5); ap.add_argument("--settle-s", type=float, default=1.0); ap.add_argument("--return-s", type=float, default=1.5)
    ap.add_argument("--finger-step", type=float, default=0.12, help="largest finger target change per tick, rad"); ap.add_argument("--max-latency-ticks", type=int, default=12)
    ap.add_argument("--start-tolerance", type=float, default=0.35, help="rad: before the student may act, arms + waist must be this close to where the simulated robot stood at the demos' first frame (0 = no check)")
    ap.add_argument("--state-stale-s", type=float, default=0.2); ap.add_argument("--camera-stale-s", type=float, default=0.5)
    ap.add_argument("--no-student", action="store_true", help="first test on the robot: approach + settle only, report how far the robot is from the demos' start pose, return; the student never acts")
    ap.add_argument("--dry-run", action="store_true", help="compute everything, send nothing to the robot"); ap.add_argument("--hand-check", action="store_true", help="only print the measured hand joints")
    ap.add_argument("--log-dir", default=os.path.join(HERE, "logs")); ap.add_argument("--script", default="", help="tests: 'seconds:command;...' typed by a timer instead of the keyboard")
    ap.add_argument("--exit-after", type=float, default=0.0, help="tests: quit after this many seconds")
    a = ap.parse_args()
    import zmq
    cfg = json.load(open(a.config)); tasks = cfg["tasks"]; fin = cfg["finger_names"]; names = cfg["state_names"]; os.makedirs(a.log_dir, exist_ok=True)
    assert names[:29] == wire.SONIC_BODY and names[29:43] == fin, "config state order is not 29 SONIC body joints + the 14 fingers"
    motor = {s: wire.hand_names(a.hand_order, s) for s in ("left", "right")}  # motor index -> joint name
    lim = {f"{s}_hand_{j}_joint": v for s in ("left", "right") for j, v in wire.DEX3_LIMITS[s].items()}; flo = np.array([lim[j][0] for j in fin]); fhi = np.array([lim[j][1] for j in fin])
    fidx = {s: [fin.index(j) for j in motor[s]] for s in ("left", "right")}  # student finger vector -> motor order
    ctx = zmq.Context.instance(); sub = ctx.socket(zmq.SUB); sub.setsockopt(zmq.SUBSCRIBE, wire.TOPIC_STATE); sub.setsockopt(zmq.CONFLATE, 1); sub.connect(f"tcp://{a.robot_host}:{a.state_port}")
    pub = ctx.socket(zmq.PUB); pub.bind(f"tcp://*:{a.action_port}"); time.sleep(0.3)
    say(f"state from tcp://{a.robot_host}:{a.state_port}; actions published on port {a.action_port} (start deploy.sh with --zmq-host <this machine>); hand order '{a.hand_order}'" + ("; DRY RUN: nothing is sent" if a.dry_run else ""))

    def read_state(st):
        m = wire.unpack_state(st); body = np.asarray(m["body_q"], np.float64)[wire.MJ_TO_SONIC]
        hq = dict(zip(motor["left"], np.asarray(m["left_hand_q"], np.float64))); hq.update(zip(motor["right"], np.asarray(m["right_hand_q"], np.float64)))
        return np.concatenate([body, [hq[j] for j in fin]]).astype(np.float32), np.asarray(m.get("token_state", np.zeros(64)), np.float32), m

    lines: queue.Queue = queue.Queue()
    if a.script:
        def typer():
            t0 = time.monotonic()
            for item in a.script.split(";"):
                at, _, text = item.partition(":"); time.sleep(max(0.0, float(at) - (time.monotonic() - t0))); lines.put(text)
        threading.Thread(target=typer, daemon=True).start()
    else:
        threading.Thread(target=lambda: [lines.put(l.rstrip("\n")) for l in sys.stdin], daemon=True).start()

    if a.hand_check:  # read-only: push one finger by hand and see which motor index moves
        say("hand check: start the control loop from the official side first; push ONE finger joint by hand and watch which entry changes. Ctrl-C to leave.")
        base = None
        while True:
            try: s43, _, m = read_state(sub.recv(zmq.NOBLOCK))
            except zmq.Again: time.sleep(0.2); continue
            cur = np.concatenate([m["left_hand_q"], m["right_hand_q"]]); base = cur if base is None else base
            print("  ".join(f"{'L' if i < 7 else 'R'}{i % 7}:{(motor['left'] + motor['right'])[i].split('_hand_')[1][:-6]}={cur[i]:+.2f}({cur[i] - base[i]:+.2f})" for i in range(14)), flush=True); time.sleep(0.5)

    cam = Camera(a.camera, a.gr00t_wbc)
    cmd = a.student_cmd.split() if a.student_cmd else [a.student_python, os.path.join(HERE, "runtime_server.py"), a.model]
    say("loading the student ..."); student = Student(cmd, f"/dev/shm/g1_client_{os.getpid()}"); say(f"student: {student.info}")
    if not student.info.get("official_format"): raise SystemExit("this student does not output 64 token + 14 finger targets (official format); refusing to run it on the robot")
    hw = tuple(student.info.get("image_hw", (168, 224)))
    gate = InstructionGate(allowed=tuple(tasks), blend_s=a.return_s)
    ARMS = [i for i, j in enumerate(wire.SONIC_BODY) if any(k in j for k in ("shoulder", "elbow", "wrist", "waist"))]
    n_in, n_set = int(round(a.blend_in_s / TICK)), int(round(a.settle_s / TICK)); extra = 1 + (a.max_latency_ticks + STEP_TICKS - 1) // STEP_TICKS

    started = False; mode = "OFF"; tick = 0; frame_no = 0; obs = []  # obs: the last ticks' (state43, image)
    planner_asked = -1e9; leave_holding = False; first_task = next(iter(tasks))  # 11-18: `planner` while holding must be typed twice; `quit!`
    retreat = None; hold_task = None; stand_fin = None; part_way = None  # 11-18: the way back down (retreat tokens of the task whose ready pose is held), fingers back to where they were when standing
    st_time = 0.0; s43 = None; tok_now = np.zeros(64, np.float32); last_tok = None; last_fin = None; ep = None; t_next = time.monotonic(); t_status = 0.0; clipped = 0; t_end = time.monotonic() + a.exit_after if a.exit_after else None

    def send_command(start, stop, planner):
        if a.dry_run: say(f"(dry run) command start={int(start)} stop={int(stop)} planner={int(planner)}"); return
        for _ in range(3): pub.send(wire.pack_command(start, stop, planner)); time.sleep(0.002)

    def send_pose(tok, f14, lim=TOKEN_LIMIT):
        nonlocal last_tok, last_fin, frame_no, clipped
        tok = np.asarray(tok, np.float32); f14 = np.asarray(f14, np.float32)
        if not (np.all(np.isfinite(tok)) and np.all(np.isfinite(f14))): gate.stop(time.monotonic(), "non-finite numbers from the student"); return
        if np.abs(tok).max() > lim: clipped += 1; tok = np.clip(tok, -lim, lim)
        f14 = np.clip(f14, flo, fhi)
        if last_fin is not None: f14 = last_fin + np.clip(f14 - last_fin, -a.finger_step, a.finger_step)
        last_tok, last_fin = tok, f14; frame_no += 1
        if ep is not None: ep["tok"].append(tok); ep["fin"].append(f14); ep["tick"].append(tick); ep["state"].append(s43 if s43 is not None else np.zeros(43, np.float32)); ep["phase"].append(ep["sub"])
        if not a.dry_run: pub.send(wire.pack_pose(tok, frame_no, f14[fidx["left"]], f14[fidx["right"]]))

    def to_planner(why):
        nonlocal mode, last_tok, last_fin
        send_command(True, False, True); mode = "PLANNER"; last_tok = last_fin = None; say(f"controller back in PLANNER mode ({why}); the student is disconnected")

    def near_ready():  # is the controller's own token one of the ready tokens (= the robot holds a start pose from an earlier run)?
        return any(np.abs(tok_now - np.asarray(T_["ready_token"], np.float32)).max() < 0.05 for T_ in tasks.values())

    say("client up. Order: robot at its mark (table edge 22 cm ahead of the pelvis centre) -> `start` (controller's own standing loop) -> `ready` (arms up around the sides to the start pose, held) -> an instruction: "
        + " | ".join(f"'{t}'" for t in tasks) + ". `planner` twice = arms back down the same way, then the controller stands by itself.")
    try:
        while True:
            now = time.monotonic(); tick += 1
            try: s43, tok_now, _ = read_state(sub.recv(zmq.NOBLOCK)); st_time = now
            except zmq.Again: pass
            rgb, cam_age = cam.read(); state_ok = now - st_time < a.state_stale_s
            cam_shape_ok = rgb is not None and abs(rgb.shape[1] / rgb.shape[0] - hw[1] / hw[0]) < 0.03  # 11-19: the student's view is 16:9 (D435i 1280x720); a 640x480 stream is the CENTRE of that view, cropping it gives a zoomed picture
            cam_ok = cam_age < a.camera_stale_s and cam_shape_ok
            cam_word = 'ok' if cam_ok else (f'WRONG SHAPE {rgb.shape[1]}x{rgb.shape[0]} (the student needs the 16:9 stream, 1280x720: set color_image_dim of the camera service)' if rgb is not None and not cam_shape_ok else 'missing')
            if state_ok and rgb is not None: obs.append((s43, to_student_image(rgb, hw))); obs = obs[-(STEP_TICKS + 1):]
            while not lines.empty():
                text = lines.get().strip(); low = text.lower()
                if not text: continue
                if low == "start":
                    if mode == "POSE": say("NOT sent: the client is streaming a pose; `planner` (twice) hands the robot back"); continue
                    if state_ok and near_ready(): say("NOT sent: the controller is holding a start pose from an earlier run, `start` would lower the arms. Type `hold` to take that pose over, or pull the table away and type `planner`"); continue
                    send_command(True, False, True); started = True; mode = "PLANNER"; say("sent: start the control loop in PLANNER mode (the robot stands by itself)")
                elif low in ("stop", "s"): gate.stop(now, "operator stop"); say("stop")
                elif low == "planner":
                    if mode == "POSE" and gate.state != IDLE: say("NOT yet: an episode is running; `stop` first, then `planner`"); continue
                    if mode == "POSE" and retreat is not None: say("already on the way down"); continue
                    if mode == "POSE" and now - planner_asked > 15.0:
                        planner_asked = now; say("`planner` takes the arms back down around the sides and hands the robot to the controller. Type `planner` again within 15 s to do it"); continue
                    planner_asked = -1e9
                    if mode == "POSE" and hold_task and "retreat_tokens" in tasks[hold_task] and last_tok is not None:
                        rt_ = np.asarray(tasks[hold_task]["retreat_tokens"], np.float32)
                        if part_way is not None: rt_ = rt_[max(0, len(rt_) - 1 - part_way):]  # from where `ready` was interrupted: the retreat is the approach backwards
                        retreat = dict(tok=rt_, k=0, t0=last_tok.copy(), f0=last_fin.copy()); say(f"RETREAT: arms back down the way they came, {0.5 + len(retreat['tok']) * TICK:.1f} s, then PLANNER")
                    else: gate.stop(now, "operator"); to_planner("operator")
                elif low == "hold":
                    if not state_ok: say("NOT holding: no robot state"); continue
                    if mode == "POSE": say("already streaming a pose"); continue
                    started = True; mode = "POSE"; last_tok = tok_now.copy(); last_fin = s43[29:43].copy(); send_command(True, False, False); say("holding the controller's current token and the measured finger pose")
                elif low == "estop": send_command(False, True, False); say("EMERGENCY STOP sent (the robot goes limp, deploy.sh exits)"); started = False; mode = "OFF"
                elif low in ("quit", "q", "exit"):
                    if mode == "POSE": say("NOT leaving: the robot is holding a pose with its hands over the table. Type `planner` twice first (arms down, hand back); `quit!` leaves it holding (the controller keeps the last token)"); continue
                    raise KeyboardInterrupt
                elif low == "quit!": leave_holding = True; raise KeyboardInterrupt
                elif low == "ready" or low.startswith("ready "):
                    want = text[5:].strip(); key = next((t_ for t_ in tasks if t_.lower() == " ".join(want.lower().rstrip(".!").split())), first_task if not want else None)
                    if key is None: say(f"NOT moving: `ready` takes no sentence or one of: " + " | ".join(tasks)); continue
                    if not (started and state_ok and mode == "PLANNER"):
                        say(f"NOT moving: `ready` refused (control loop started: {started}, robot state: {'ok' if state_ok else 'missing'}, controller mode: {mode}; it starts from the controller's standing loop)"); continue
                    ok, msg = gate.submit(key, now)
                    if not ok: say("NOT moving: " + msg); continue
                    T = tasks[key]; app = np.asarray(T["approach_tokens"], np.float32); gate.episode_s = a.blend_in_s + len(app) * TICK + a.settle_s + 2.0
                    ep = dict(task=key, t0=now, tick0=tick, app=app, ready=np.asarray(T["ready_token"], np.float32), fready=np.array([T["ready_fingers"][j] for j in fin], np.float32), ready_only=True,
                              tok0=tok_now.copy(), f0=s43[29:43].copy(), sub="approach", act0=None, chunk=None, chunk0=0, asked=False, starve=0, lat=[], tok=[], fin=[], tick=[], state=[], phase=[], first=None)
                    last_fin = s43[29:43].copy(); stand_fin = s43[29:43].copy(); student.drop(); send_command(True, False, False); mode = "POSE"
                    say(f"READY: arms up around the sides to the start pose of '{key}' in {a.blend_in_s + len(app) * TICK:.1f} s, then held")
                elif low == "status": say(f"gate {gate.state} ({gate.reason}); controller mode {mode}; state {'ok' if state_ok else 'STALE'}, camera {cam_word}; tokens clipped {clipped}")
                else:
                    if not (started and state_ok and cam_ok and len(obs) > STEP_TICKS):
                        say(f"NOT moving: instruction '{text}' refused, the robot is not ready (control loop started: {started}, robot state: {'ok' if state_ok else 'missing'}, camera: {cam_word})"); continue
                    if not (mode == "POSE" and last_tok is not None) and gate.state == IDLE and " ".join(low.rstrip(".!").split()) in [t_.lower() for t_ in tasks]:
                        say(f"NOT moving: instruction '{text}' refused, the robot is not holding the start pose. Type `ready` first, then give the instruction"); continue
                    if gate.state == IDLE and " ".join(low.rstrip(".!").split()) in [t_.lower() for t_ in tasks]:  # 11-19: measured, before anything moves
                        if part_way is not None: say(f"NOT moving: instruction '{text}' refused, `ready` was interrupted with the arms part-way. Type `planner` twice, then `ready` again"); continue
                        dev_ = np.abs(s43[:29] - np.asarray(tasks[hold_task if hold_task in tasks else first_task]["start_body"], np.float32))[ARMS]
                        if a.start_tolerance > 0 and dev_.max() > a.start_tolerance:
                            say(f"NOT moving: instruction '{text}' refused, the robot is not at the start pose ({wire.SONIC_BODY[ARMS[int(dev_.argmax())]]} is {dev_.max():.2f} rad off, limit {a.start_tolerance}). Type `planner` twice, then `ready` again"); continue
                    ok, msg = gate.submit(text, now)
                    if ok:  # 11-18: always from the held pose: a short blend to this task's ready token, never the approach from standing (it would pass through the table)
                        T = tasks[gate.task]; app = np.zeros((0, 64), np.float32); gate.episode_s = a.blend_in_s + a.settle_s + float(T["episode_s"])
                        ep = dict(task=gate.task, t0=now, tick0=tick, app=app, ready=np.asarray(T["ready_token"], np.float32), fready=np.array([T["ready_fingers"][j] for j in fin], np.float32), ready_only=False,
                                  tok0=last_tok.copy(), f0=s43[29:43].copy(), sub="approach", act0=None, chunk=None, chunk0=0, asked=False, starve=0, lat=[], tok=[], fin=[], tick=[], state=[], phase=[], first=None)
                        last_fin = s43[29:43].copy(); student.drop()
                        send_command(True, False, False); mode = "POSE"  # streamed tokens (already so while holding)
                    say(f"ACCEPTED: '{gate.task}': blend to its start pose {a.blend_in_s:.1f} s, settle {a.settle_s:.1f} s, then the student for at most {float(tasks[gate.task]['episode_s']):.1f} s" if ok else "NOT moving: " + msg)
            g = gate.tick(now, sensors_ok=state_ok and (cam_ok or (ep is not None and ep["ready_only"])))  # 11-19: `ready` (arms up) needs the robot state only, not the camera
            if g == RUNNING and ep is not None:
                k = tick - ep["tick0"]; n_app = len(ep["app"]); n_pre = n_in + n_app + n_set; wf = min(1.0, (k + 1) / (n_in + n_app))
                if k < n_in:
                    w = (k + 1) / n_in; ep["sub"] = "approach"; send_pose((1 - w) * ep["tok0"] + w * (ep["app"][0] if n_app else ep["ready"]), (1 - wf) * ep["f0"] + wf * ep["fready"], PRE_LIMIT)
                elif k < n_in + n_app:
                    send_pose(ep["app"][k - n_in], (1 - wf) * ep["f0"] + wf * ep["fready"], PRE_LIMIT)
                elif ep["act0"] is None:  # settle: hold the ready command; ask for the first plan 0.2 s before the end; act when it is there
                    ep["sub"] = "settle"; send_pose(ep["ready"], ep["fready"])
                    if ep["ready_only"]:  # `ready`: no student; report where the robot stands against the demos' start pose, then hold
                        if k >= n_pre + 50:
                            dev = np.abs(s43[:29] - np.asarray(tasks[ep["task"]]["start_body"], np.float32))[ARMS]
                            gate.stop(now, f"ready pose reached: within {dev.max():.2f} rad of the demos' start pose (worst {wire.SONIC_BODY[ARMS[int(dev.argmax())]]}). Place plate and orange, then give the instruction")
                    elif k >= n_pre - 10 and not ep["asked"] and not student.busy:
                        ep["asked"] = True; student.request(tick, obs[0][0], obs[-1][0], obs[0][1], obs[-1][1], ep["task"], EVERY // STEP_TICKS, extra, reset=True)
                    r = student.poll() if k >= n_pre and not ep["ready_only"] else None
                    if r is not None:
                        dev = np.abs(s43[:29] - np.asarray(tasks[ep["task"]]["start_body"], np.float32))[ARMS]; worst = wire.SONIC_BODY[ARMS[int(dev.argmax())]]
                        if a.no_student:
                            gate.stop(now, f"--no-student test: start pose is within {dev.max():.2f} rad of the demos' (worst {worst}); per joint: " + ", ".join(f"{wire.SONIC_BODY[i][:-6]} {s43[i] - tasks[ep['task']]['start_body'][i]:+.2f}" for i in ARMS if abs(s43[i] - tasks[ep['task']]['start_body'][i]) > 0.1))
                        elif a.start_tolerance > 0 and dev.max() > a.start_tolerance:
                            gate.stop(now, f"start pose is {dev.max():.2f} rad away from the demos' ({worst}); the student was NOT started (see --start-tolerance)")
                        else:
                            ep["chunk"], ep["chunk0"], ep["act0"] = upsample(r[1]), tick + 1, tick + 1; ep["lat"].append(r[2]); ep["first"] = obs[-1][1]
                            say(f"student acting (first plan took {1000 * r[2]:.0f} ms; start pose within {dev.max():.2f} rad of the demos', worst {worst})")
                    elif k > n_pre + 100 and not ep["ready_only"]: gate.stop(now, "the student did not answer within 2 s")
                else:
                    ep["sub"] = "act"; j = tick - ep["act0"]
                    if j > 0 and j % EVERY == 0 and not student.busy:
                        student.request(tick, obs[0][0], obs[-1][0], obs[0][1], obs[-1][1], ep["task"], EVERY // STEP_TICKS, extra)
                    r = student.poll()
                    if r is not None: ep["chunk"], ep["chunk0"] = upsample(r[1]), r[0]; ep["lat"].append(r[2])  # indexed from its observation's tick: the latency is skipped
                    i = tick - ep["chunk0"]
                    if i < len(ep["chunk"]): ep["starve"] = 0; send_pose(ep["chunk"][i, :64], ep["chunk"][i, -14:])
                    else:
                        ep["starve"] += 1; send_pose(last_tok, last_fin)
                        if ep["starve"] > 25: gate.stop(now, "no new plan for 0.5 s")
            elif g == RETURNING and ep is not None and last_tok is not None:
                if ep["sub"] != "return":
                    ep["cut"] = bool(ep["ready_only"] and ep["sub"] == "approach")  # 11-19: `ready` interrupted on the way up: the short way to the ready pose passes through the table, so the arms stay where they are
                    ep["k_cut"] = int(max(0, min(tick - ep["tick0"] - n_in, len(ep["app"]) - 1))) if ep["cut"] else 0
                    ep["sub"] = "return"; ep["rt"], ep["rf"] = last_tok.copy(), last_fin.copy()
                    say(f"READY INTERRUPTED ({gate.reason}): the arms stay where they are (step {ep['k_cut']} of {len(ep['app'])}). `planner` twice takes them back down the way they came" if ep["cut"] else f"returning to the ready pose: {gate.reason}")
                if ep.get("cut"): send_pose(ep["rt"], ep["rf"], PRE_LIMIT)
                else: w = gate.blend_weight(now); send_pose((1 - w) * ep["rt"] + w * ep["ready"], (1 - w) * ep["rf"] + w * ep["fready"])
            elif g == IDLE:
                if ep is not None:  # the episode just ended
                    lat = np.asarray(ep["lat"]); f = os.path.join(a.log_dir, time.strftime("%Y%m%d_%H%M%S") + "_" + ep["task"].replace(" ", "_") + ".npz")
                    np.savez_compressed(f, task=ep["task"], token=np.asarray(ep["tok"]), fingers=np.asarray(ep["fin"]), tick=np.asarray(ep["tick"]), state=np.asarray(ep["state"]), phase=np.asarray(ep["phase"]),
                                        latency_s=lat, finger_names=np.asarray(fin), first_image=ep["first"] if ep["first"] is not None else np.zeros((1, 1, 3), np.uint8))
                    say(f"{'ready' if ep['ready_only'] else 'episode'} over ({gate.log[-2][1] if len(gate.log) > 1 else ''}); {len(ep['tok'])} messages; plan latency median {1000 * np.median(lat) if len(lat) else 0:.0f} ms, max {1000 * lat.max() if len(lat) else 0:.0f} ms; log {f}")
                    was_ready = ep["ready_only"]; hold_task = ep["task"]; part_way = ep["k_cut"] if ep.get("cut") else None; ep = None
                    if mode == "POSE" and part_way is not None: say("NOT at the start pose: `ready` was interrupted with the arms part-way. Instructions are refused. Type `planner` twice (arms back down the way they came), then `ready` again")
                    elif mode == "POSE" and a.after == "planner" and not was_ready: to_planner("episode over")
                    elif mode == "POSE": say("HOLDING the ready pose; waiting for an instruction (`planner` twice = arms down, hand back)")
                if retreat is not None and mode == "POSE":  # the way back down: 0.5 s onto the first retreat token, the retreat tokens, 0.5 s at rest, then the controller's own loop
                    k = retreat["k"]; n_r = len(retreat["tok"]); wf = min(1.0, (k + 1) / (25 + n_r)); fin_t = (1 - wf) * retreat["f0"] + wf * (stand_fin if stand_fin is not None else retreat["f0"])
                    if k < 25: w = (k + 1) / 25; send_pose((1 - w) * retreat["t0"] + w * retreat["tok"][0], fin_t, PRE_LIMIT)
                    else: send_pose(retreat["tok"][min(k - 25, n_r - 1)], fin_t, PRE_LIMIT)
                    retreat["k"] = k + 1
                    if k >= 25 + n_r + 25: retreat = None; hold_task = None; part_way = None; to_planner("retreat done")
                elif mode == "POSE" and last_tok is not None: send_pose(last_tok, last_fin)  # --after ready: keep holding
            if now - t_status > 10.0 and g == IDLE:
                t_status = now; say(f"STANDBY ({mode}): not moving; state {'ok' if state_ok else 'missing'}, camera {cam_word}")
            if t_end and now > t_end: raise KeyboardInterrupt
            t_next += TICK; d = t_next - time.monotonic()
            if d > 0: time.sleep(d)
            else: t_next = time.monotonic()
    except KeyboardInterrupt:
        say("leaving")
    finally:
        # 11-18: with a table under the hands, handing back to the planner (arms down) is the dangerous move, a held token is not:
        # the controller keeps the last token, the robot keeps its pose. Hand back only if asked for (--after planner).
        if mode == "POSE" and a.after == "planner" and not leave_holding: to_planner("client exit")
        elif mode == "POSE": say("LEFT HOLDING: the controller keeps the last token, the robot keeps this pose. Restart the client, type `hold`, then `ready` is not needed: give an instruction or `planner` twice; emergency stop = `O` in the deploy.sh terminal")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
