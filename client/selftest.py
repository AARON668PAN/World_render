#!/usr/bin/env python
"""Check of the student on this computer, without a robot (11-18).
  1. loads the student with no download (offline_assets) and proves every weight equals the checkpoint file,
  2. numbers: for fixed inputs, the student's internal features and one denoising step must equal the values computed on the
     training machine (student/selftest_reference.npz) -> the installation (libraries, weights, tokenizer) reproduces the model,
  3. speed: plans like the robot client asks for them (a new plan every 0.4 s); the client tolerates up to 0.24 s per plan,
  4. prints the range of the planned token and finger targets.
usage: selftest.py [student dir] [--make-reference]   env RUNTIME_DEVICE=cuda|cpu"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "lerobot_policy_dit_resnet"))
import offline_assets  # noqa: E402
offline_assets.set_env()
import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import lerobot_policy_dit_resnet  # noqa: E402,F401
from student_runtime import StudentRuntime  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]; MAKE = "--make-reference" in sys.argv
MODEL = args[0] if args else os.path.join(ROOT, "student"); REF = os.path.join(MODEL, "selftest_reference.npz")
if not os.path.exists(os.path.join(MODEL, "model.safetensors")):
    sys.exit(f"FAIL: {MODEL}/model.safetensors is missing (run ./assemble_weights.sh)")
offline_assets.enable()
DEVICE = os.environ.get("RUNTIME_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time(); rt = StudentRuntime(MODEL, DEVICE); n = offline_assets.verify(rt.policy, MODEL)
print(f"1. loaded on {DEVICE} in {time.time() - t0:.1f} s; {n} tensors equal to the checkpoint file; {rt.info()}")

cfg = json.load(open(os.path.join(ROOT, "config", "config_orange_plate.json"))); tasks = list(cfg["tasks"])
rng = np.random.default_rng(0); H, W = rt.image_hw
state = np.zeros((1, rt.state_dim), np.float32); state[0, :29] = np.asarray(cfg["tasks"][tasks[0]]["start_body"], np.float32)[:29]
yy, xx = np.mgrid[0:H, 0:W]; img = np.stack([xx * 255 // W, yy * 255 // H, (xx + yy) % 256], -1).astype(np.uint8)[None]  # a fixed colour ramp
tens = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.0
x_fixed = torch.from_numpy(rng.standard_normal((1, rt.horizon, rt.action_dim)).astype(np.float32)).to(DEVICE)
out = {}
with torch.no_grad():
    for i, task in enumerate(tasks):
        batch = {"observation.state": torch.from_numpy(np.stack([state, state], 1)), "task": [task]}
        for k in rt.img_keys: batch[k] = torch.stack([tens(img), tens(img)], 1)
        b = rt.pre(batch); b = {k: v for k, v in b.items() if k != "action" and v is not None}
        cond = rt.policy.observation_encoder.encode(rt.policy._prepare_batch(b))
        v = rt.policy.noise_predictor(x_fixed, torch.full((1,), 0.5, device=DEVICE), conditioning_vec=cond)
        out[f"cond_{i}"] = cond.float().cpu().numpy(); out[f"v_{i}"] = v.float().cpu().numpy()
if MAKE:
    np.savez(REF, tasks=np.array(tasks), **out); print(f"2. reference written: {REF}")
elif not os.path.exists(REF):
    print(f"2. SKIPPED: no {REF}")
else:
    R = np.load(REF); worst = max(float(np.abs(out[k] - R[k]).max()) for k in out); scale = max(float(np.abs(R[k]).max()) for k in out)
    ok = worst < 5e-3 * max(scale, 1.0)
    print(f"2. numbers against the training machine: largest difference {worst:.2e} (values up to {scale:.1f}) -> {'OK' if ok else 'FAIL'}")
    if not ok: sys.exit("FAIL: this installation does not reproduce the student")
lat = []; rt.reset(); a = None
for i in range(8):
    t0 = time.perf_counter(); a = rt.plan(state, state, img, img, tasks[0], first=5, extra=4, blend=1, rtc=True)
    if DEVICE == "cuda": torch.cuda.synchronize()
    lat.append(time.perf_counter() - t0)
med = 1000 * float(np.median(lat[2:])); print(f"3. time per plan: median {med:.0f} ms, slowest {1000 * max(lat[2:]):.0f} ms (first call {1000 * lat[0]:.0f} ms) -> {'OK' if med < 240 else 'TOO SLOW for the robot (limit 240 ms): use a GPU'}")
tok, fin = a[0, :, :64], a[0, :, -rt.finger_dim:]
print(f"4. plan {a.shape}: token {tok.min():+.2f}..{tok.max():+.2f}, finger targets {fin.min():+.2f}..{fin.max():+.2f} rad, all finite: {bool(np.isfinite(a).all())}")
print("SELFTEST_DONE" + ("" if med < 240 else " (but too slow)"))
