#!/usr/bin/env python
"""11-18: real-robot client for the vision student on a Unitree G1 + Dex3 hands behind NVIDIA's official GEAR-SONIC controller
(gear_sonic_deploy, `./deploy.sh --input-type zmq_manager real` with the DEFAULT `policy/release` checkpoint: the simulator's
SONIC is that one; the docs' sonic_v1_1 has another token space and must not be used with this student).
It replaces gear_sonic/scripts/run_vla_inference.py: the same wire protocol (wire.py), our student instead of GR00T.

The robot moves only on an instruction (gate.py). What is sent, by state:
  STANDBY    nothing. The controller is in its PLANNER mode and stands by itself. The student is not connected to the robot.
  RUNNING    after an accepted instruction (one of the config's sentences, typed in this terminal):
             approach  switch the controller to streamed tokens, blend from its current token into the precomputed approach
                       tokens (default standing pose -> the demos' start pose), fingers to the start pose
             settle    the ready token for --settle-s, the first plan is computed from here
             act       the student: a new plan every 20 ticks (0.4 s) from the current camera image + joint angles, token
                       held 4 ticks, finger targets interpolated, latency-compensated hand-over as in the simulator exam
  RETURNING  when the episode time is up, on `stop`, or when camera / robot state go stale: blend back to the ready token,
             then the controller's PLANNER mode again -> STANDBY. One instruction = one episode.
Every outgoing message is checked here (the C++ side checks nothing): finite numbers, token within +-0.6, finger targets
inside the Dex3 joint limits and at most --finger-step rad per tick. If this program dies the controller keeps the last token:
the operator's emergency stop is the `O` key in the deploy.sh terminal (documented by NVIDIA; the robot goes limp).

Operator commands (type + Enter): start | <instruction sentence> | stop | planner | status | estop | quit
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
    """Centre crop to the student's 4:3 and resize (area filter) to 168 x 224."""
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
    ap.add_argument("--hand-order", default="unitree", choices=list(wire.DEX3_ORDER)); ap.add_argument("--after", default="planner", choices=["planner", "ready"], help="where to wait after an episode")
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
    st_time = 0.0; s43 = None; tok_now = np.zeros(64, np.float32); last_tok = None; last_fin = None; ep = None; t_next = time.monotonic(); t_status = 0.0; clipped = 0; t_end = time.monotonic() + a.exit_after if a.exit_after else None

    def send_command(start, stop, planner):
        if a.dry_run: say(f"(dry run) command start={int(start)} stop={int(stop)} planner={int(planner)}"); return
        for _ in range(3): pub.send(wire.pack_command(start, stop, planner)); time.sleep(0.002)

    def send_pose(tok, f14):
        nonlocal last_tok, last_fin, frame_no, clipped
        tok = np.asarray(tok, np.float32); f14 = np.asarray(f14, np.float32)
        if not (np.all(np.isfinite(tok)) and np.all(np.isfinite(f14))): gate.stop(time.monotonic(), "non-finite numbers from the student"); return
        if np.abs(tok).max() > TOKEN_LIMIT: clipped += 1; tok = np.clip(tok, -TOKEN_LIMIT, TOKEN_LIMIT)
        f14 = np.clip(f14, flo, fhi)
        if last_fin is not None: f14 = last_fin + np.clip(f14 - last_fin, -a.finger_step, a.finger_step)
        last_tok, last_fin = tok, f14; frame_no += 1
        if ep is not None: ep["tok"].append(tok); ep["fin"].append(f14); ep["tick"].append(tick); ep["state"].append(s43 if s43 is not None else np.zeros(43, np.float32)); ep["phase"].append(ep["sub"])
        if not a.dry_run: pub.send(wire.pack_pose(tok, frame_no, f14[fidx["left"]], f14[fidx["right"]]))

    def to_planner(why):
        nonlocal mode, last_tok, last_fin
        send_command(True, False, True); mode = "PLANNER"; last_tok = last_fin = None; say(f"controller back in PLANNER mode ({why}); the student is disconnected")

    say("ready. Type `start` to start the controller's own standing loop, then an instruction: " + " | ".join(f"'{t}'" for t in tasks))
    try:
        while True:
            now = time.monotonic(); tick += 1
            try: s43, tok_now, _ = read_state(sub.recv(zmq.NOBLOCK)); st_time = now
            except zmq.Again: pass
            rgb, cam_age = cam.read(); state_ok = now - st_time < a.state_stale_s; cam_ok = cam_age < a.camera_stale_s
            if state_ok and rgb is not None: obs.append((s43, to_student_image(rgb, hw))); obs = obs[-(STEP_TICKS + 1):]
            while not lines.empty():
                text = lines.get().strip(); low = text.lower()
                if not text: continue
                if low == "start":
                    send_command(True, False, True); started = True; mode = "PLANNER"; say("sent: start the control loop in PLANNER mode (the robot stands by itself)")
                elif low in ("stop", "s"): gate.stop(now, "operator stop"); say("stop: returning")
                elif low == "planner": gate.stop(now, "operator"); to_planner("operator")
                elif low == "estop": send_command(False, True, False); say("EMERGENCY STOP sent (the robot goes limp, deploy.sh exits)"); started = False; mode = "OFF"
                elif low in ("quit", "q", "exit"): raise KeyboardInterrupt
                elif low == "status": say(f"gate {gate.state} ({gate.reason}); controller mode {mode}; state {'ok' if state_ok else 'STALE'}, camera {'ok' if cam_ok else 'STALE'}; tokens clipped {clipped}")
                else:
                    if not (started and state_ok and cam_ok and len(obs) > STEP_TICKS):
                        say(f"NOT moving: instruction '{text}' refused, the robot is not ready (control loop started: {started}, robot state: {'ok' if state_ok else 'missing'}, camera: {'ok' if cam_ok else 'missing'})"); continue
                    ok, msg = gate.submit(text, now)
                    if ok:
                        T = tasks[gate.task]; app = np.asarray(T["approach_tokens"], np.float32); gate.episode_s = a.blend_in_s + len(app) * TICK + a.settle_s + float(T["episode_s"])
                        ep = dict(task=gate.task, t0=now, tick0=tick, app=app, ready=np.asarray(T["ready_token"], np.float32), fready=np.array([T["ready_fingers"][j] for j in fin], np.float32),
                                  tok0=tok_now.copy(), f0=s43[29:43].copy(), sub="approach", act0=None, chunk=None, chunk0=0, asked=False, starve=0, lat=[], tok=[], fin=[], tick=[], state=[], phase=[], first=None)
                        last_fin = s43[29:43].copy(); student.drop()
                        send_command(True, False, False); mode = "POSE"  # streamed tokens from now on
                    say(f"ACCEPTED: '{gate.task}': approach {a.blend_in_s + len(ep['app']) * TICK:.1f} s, settle {a.settle_s:.1f} s, then the student for at most {float(tasks[gate.task]['episode_s']):.1f} s" if ok else "NOT moving: " + msg)
            g = gate.tick(now, sensors_ok=state_ok and cam_ok)
            if g == RUNNING and ep is not None:
                k = tick - ep["tick0"]; n_app = len(ep["app"]); n_pre = n_in + n_app + n_set; wf = min(1.0, (k + 1) / (n_in + n_app))
                if k < n_in:
                    w = (k + 1) / n_in; ep["sub"] = "approach"; send_pose((1 - w) * ep["tok0"] + w * ep["app"][0], (1 - wf) * ep["f0"] + wf * ep["fready"])
                elif k < n_in + n_app:
                    send_pose(ep["app"][k - n_in], (1 - wf) * ep["f0"] + wf * ep["fready"])
                elif ep["act0"] is None:  # settle: hold the ready command; ask for the first plan 0.2 s before the end; act when it is there
                    ep["sub"] = "settle"; send_pose(ep["ready"], ep["fready"])
                    if k >= n_pre - 10 and not ep["asked"] and not student.busy:
                        ep["asked"] = True; student.request(tick, obs[0][0], obs[-1][0], obs[0][1], obs[-1][1], ep["task"], EVERY // STEP_TICKS, extra, reset=True)
                    r = student.poll() if k >= n_pre else None
                    if r is not None:
                        dev = np.abs(s43[:29] - np.asarray(tasks[ep["task"]]["start_body"], np.float32))[ARMS]; worst = wire.SONIC_BODY[ARMS[int(dev.argmax())]]
                        if a.no_student:
                            gate.stop(now, f"--no-student test: start pose is within {dev.max():.2f} rad of the demos' (worst {worst}); per joint: " + ", ".join(f"{wire.SONIC_BODY[i][:-6]} {s43[i] - tasks[ep['task']]['start_body'][i]:+.2f}" for i in ARMS if abs(s43[i] - tasks[ep['task']]['start_body'][i]) > 0.1))
                        elif a.start_tolerance > 0 and dev.max() > a.start_tolerance:
                            gate.stop(now, f"start pose is {dev.max():.2f} rad away from the demos' ({worst}); the student was NOT started (see --start-tolerance)")
                        else:
                            ep["chunk"], ep["chunk0"], ep["act0"] = upsample(r[1]), tick + 1, tick + 1; ep["lat"].append(r[2]); ep["first"] = obs[-1][1]
                            say(f"student acting (first plan took {1000 * r[2]:.0f} ms; start pose within {dev.max():.2f} rad of the demos', worst {worst})")
                    elif k > n_pre + 100: gate.stop(now, "the student did not answer within 2 s")
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
                if ep["sub"] != "return": ep["sub"] = "return"; ep["rt"], ep["rf"] = last_tok.copy(), last_fin.copy(); say(f"returning to the ready pose: {gate.reason}")
                w = gate.blend_weight(now); send_pose((1 - w) * ep["rt"] + w * ep["ready"], (1 - w) * ep["rf"] + w * ep["fready"])
            elif g == IDLE:
                if ep is not None:  # the episode just ended
                    lat = np.asarray(ep["lat"]); f = os.path.join(a.log_dir, time.strftime("%Y%m%d_%H%M%S") + "_" + ep["task"].replace(" ", "_") + ".npz")
                    np.savez_compressed(f, task=ep["task"], token=np.asarray(ep["tok"]), fingers=np.asarray(ep["fin"]), tick=np.asarray(ep["tick"]), state=np.asarray(ep["state"]), phase=np.asarray(ep["phase"]),
                                        latency_s=lat, finger_names=np.asarray(fin), first_image=ep["first"] if ep["first"] is not None else np.zeros((1, 1, 3), np.uint8))
                    say(f"episode over ({gate.log[-2][1] if len(gate.log) > 1 else ''}); {len(ep['tok'])} messages; plan latency median {1000 * np.median(lat) if len(lat) else 0:.0f} ms, max {1000 * lat.max() if len(lat) else 0:.0f} ms; log {f}")
                    ep = None
                    if a.after == "planner": to_planner("episode over")
                    else: say("holding the ready pose; waiting for an instruction")
                if mode == "POSE" and last_tok is not None: send_pose(last_tok, last_fin)  # --after ready: keep holding
            if now - t_status > 10.0 and g == IDLE:
                t_status = now; say(f"STANDBY ({mode}): not moving; state {'ok' if state_ok else 'missing'}, camera {'ok' if cam_ok else 'missing'}")
            if t_end and now > t_end: raise KeyboardInterrupt
            t_next += TICK; d = t_next - time.monotonic()
            if d > 0: time.sleep(d)
            else: t_next = time.monotonic()
    except KeyboardInterrupt:
        say("leaving")
    finally:
        if mode == "POSE": to_planner("client exit")  # never leave the controller on a stale token
        time.sleep(0.1)


if __name__ == "__main__":
    main()
