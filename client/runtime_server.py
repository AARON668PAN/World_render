"""11-18: the deployment inference core (student_runtime.StudentRuntime) behind the simulator's closed loop, as a drop-in for
teacher_multi/policy/policy_server.py (same line protocol: one JSON request per line on stdin {"batch_state": [W x S],
"batch_state_prev", "images": <npy path, W x H x W x 3 uint8>, "images_prev", "first": k, "extra": e, "rtc": 0/1, "blend": nb}
-> {"actions": [W x (k + e) x A]} on stdout), so the exam runs the code that goes on the robot.
The real-robot client adds "task": <sentence> (else CL_TASK_TEXT) and "reset": 1 at the start of an episode.
usage: runtime_server.py <student dir>   env CL_TASK_TEXT, RUNTIME_DEVICE (default: cuda when there is one, else cpu), STUDENT_OFFLINE=0\n(deployment package copy: + offline_assets.py and the device default; the inference itself is student_runtime.py, unchanged)"""
import json, os, sys
OUT = sys.stdout; sys.stdout = sys.stderr  # replies only on the real stdout; library prints go to stderr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lerobot_policy_dit_resnet"))  # the policy plugin of this repository
import offline_assets  # noqa: E402  (deployment package: no downloads; STUDENT_OFFLINE=0 turns it off)
offline_assets.set_env()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import lerobot_policy_dit_resnet  # noqa: E402,F401  (registers the policy type "dit_resnet")
from student_runtime import StudentRuntime  # noqa: E402

offline_assets.enable()
DEVICE = os.environ.get("RUNTIME_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
rt = StudentRuntime(sys.argv[1], DEVICE); TASK = os.environ.get("CL_TASK_TEXT", "")
print(f"student on {DEVICE}: {offline_assets.verify(rt.policy, sys.argv[1])} tensors equal to the checkpoint file", file=sys.stderr, flush=True)
# warm-up: the first call in a new process pays for CUDA / cuDNN start-up (250 ms measured on an RTX 5090, the client's limit is
# 240 ms); pay it here, before the client is told the student is ready, not in the first plan of an episode
_z = np.zeros((1, rt.state_dim), np.float32); _im = np.zeros((1, *rt.image_hw, 3), np.uint8)
for _ in range(3): rt.plan(_z, _z, _im, _im, TASK or "warm-up", 5, 4, 1, True)
rt.reset()
print(json.dumps(dict(ready=True, task=TASK, **rt.info())), file=OUT, flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req.get("reset"): rt.reset()  # start of an episode (g1_client.py): forget the previous plan
    sc = np.asarray(req["batch_state"], np.float32)[:, : rt.state_dim]; sp = np.asarray(req["batch_state_prev"], np.float32)[:, : rt.state_dim]
    act = rt.plan(sp, sc, np.load(req["images_prev"]), np.load(req["images"]), req.get("task", TASK), int(req.get("first", 1)), int(req.get("extra", 0)),
                  int(req.get("blend", 0)), bool(req.get("rtc", 0)))
    print(json.dumps(dict(actions=act.round(5).tolist())), file=OUT, flush=True)
