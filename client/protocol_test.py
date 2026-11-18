#!/usr/bin/env python
"""The whole robot client against a fake robot on this computer (11-18): no robot and no NVIDIA code needed.
fake_robot.py speaks the controller's sockets and message formats; g1_client.py runs as on the robot, with the real student,
and a timer types: an instruction BEFORE `start` (must be refused), `start`, the instruction again (refused: the robot does not hold
the start pose yet), `ready` (arms up, then held), the instruction (one episode, back to holding), `quit` (refused while holding),
`planner` twice (hand-back).
check_protocol.py then reads what the fake robot received: no motion message outside the episode, a steady 50 Hz inside it,
every value in range, no jump at the hand-over.
usage: protocol_test.py [--stub <teacher demo npz>] [extra g1_client.py options]   (--stub: replay recorded labels instead of
the student, for machines without a GPU)"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); PY = sys.executable
args = sys.argv[1:]; stub = None
if "--stub" in args: i = args.index("--stub"); stub = args[i + 1]; del args[i:i + 2]
cfg = os.path.join(ROOT, "config", "config_orange_plate.json"); logs = os.path.join(ROOT, "logs"); os.makedirs(logs, exist_ok=True); log = os.path.join(logs, "fake_robot.npz")
A, S, C = "15556", "15557", "15558"; sentence = "put the orange on the plate"
client = [PY, os.path.join(HERE, "g1_client.py"), "--config", cfg, "--robot-host", "localhost", "--action-port", A, "--state-port", S, "--camera", f"zmqjpeg:localhost:{C}",
          "--script", f"3:{sentence};5:start;7:{sentence};9:ready;17:{sentence};33:quit;34:planner;35:planner", "--exit-after", "38", "--log-dir", logs] + args
if stub: client += ["--student-cmd", f"{PY} {os.path.join(HERE, 'stub_server.py')} {stub}"]
c = subprocess.Popen(client, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # binds the action port first, as on the robot
time.sleep(1.0)
r = subprocess.Popen([PY, os.path.join(HERE, "fake_robot.py"), "--config", cfg, "--seconds", "44", "--log", log, "--action-port", A, "--state-port", S, "--camera-port", C],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
out_c = c.communicate()[0]; out_r = r.communicate()[0]
print("---- client"); print("\n".join(l for l in out_c.splitlines() if l.startswith("[")))
print("---- fake robot"); print(out_r.strip().splitlines()[-1] if out_r.strip() else "(no output)")
if not os.path.exists(log): sys.exit("FAIL: the fake robot wrote no log")
chk = subprocess.run([PY, os.path.join(HERE, "check_protocol.py"), log], capture_output=True, text=True, cwd=HERE)
print("---- checks"); print(chk.stdout.strip()); bad = "FAIL" in chk.stdout or chk.returncode != 0 or out_c.count("refused") < 2 or "HOLDING" not in out_c or "NOT leaving" not in out_c or "student acting" not in out_c
if chk.returncode != 0: print(chk.stderr[-800:])
print("PROTOCOL_TEST " + ("FAILED" if bad else "OK"))
sys.exit(1 if bad else 0)
