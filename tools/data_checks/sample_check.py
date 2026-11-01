"""Spot check of the downloaded G1 WBT Inspire videos: for every official dataset, 2 random leg-free training windows
(windows.json, 49 frames at 15 fps) from different episodes, played one after the other in one tile of a grid video;
each tile is labelled with how much the robot base moves during the window (odometry state_base_pose: distance,
yaw change, height change). Also a stills sheet of moments the leg classifier rejected, for contrast.
usage: isaaclab_newton python sample_check.py [seed=0]   -> check/windows_grid.mp4, check/rejected.jpg, check/motion.json"""
import glob, json, os, random, subprocess, sys
from collections import defaultdict
import cv2, numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

ROOT = "/data/datasets/unitree_g1_inspire"
OUT = f"{ROOT}/check"; os.makedirs(OUT, exist_ok=True)
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
TW, TH, COLS = 288, 216, 6
KEY = "videos/observation.images.head_stereo_left"
index = {(e["dataset"], e["episode"]): e for e in json.load(open(f"{ROOT}/index.json"))}
free = np.load(f"{ROOT}/leg_free.npz")


def vpath(e):
    return f"{ROOT}/official/{e['dataset']}/{KEY}/chunk-{e['chunk']:03d}/file-{e['file']:03d}.mp4"


def clip(e, t, n):
    """n frames at 15 fps from video time t (s), TWxTH."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", vpath(e), "-frames:v", str(n), "-vf", f"fps=15,scale={TW}:{TH}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, TH, TW, 3)
    return np.concatenate([f, np.repeat(f[-1:], n - len(f), 0)]) if 0 < len(f) < n else f


def base_motion(e, f0, f1):
    """Base displacement (m), |yaw change| (deg), height range (cm) over episode frames [f0, f1) (30 fps)."""
    ds = f"{ROOT}/official/{e['dataset']}"
    for p in sorted(glob.glob(f"{ds}/data/**/*.parquet", recursive=True)):
        if "observation.state.state_base_pose" not in pq.read_schema(p).names:
            return None
        t = pq.read_table(p, columns=["episode_index", "frame_index", "observation.state.state_base_pose"],
                          filters=[("episode_index", "=", e["episode"])])
        if t.num_rows:
            fi = t.column("frame_index").to_numpy(); bp = np.array(t.column("observation.state.state_base_pose").to_pylist())[np.argsort(fi)]
            bp = bp[f0:f1]
            yaw = np.unwrap(R.from_quat(bp[:, 3:7]).as_euler("xyz")[:, 2])
            return dict(dist=float(np.linalg.norm(bp[-1, :2] - bp[0, :2])), path=float(np.linalg.norm(np.diff(bp[:, :2], axis=0), axis=1).sum()),
                        yaw=float(np.degrees(abs(yaw[-1] - yaw[0]))), dz=float((bp[:, 2].max() - bp[:, 2].min()) * 100))
    return None


by_ds = defaultdict(list)
for w in json.load(open(f"{ROOT}/windows.json")):
    e = index[(w["dataset"], w["episode"])]
    if os.path.exists(vpath(e)):
        by_ds[w["dataset"]].append(w)
tiles, motion = [], {}
for ds in sorted(by_ds):
    ws = by_ds[ds]; random.shuffle(ws); pick, seen = [], set()
    for w in ws:
        if w["episode"] not in seen:
            pick.append(w); seen.add(w["episode"])
        if len(pick) == 2:
            break
    frames = []
    for w in pick:
        e = index[(ds, w["episode"])]
        f = clip(e, e["t0"] + 2 * w["start15"] / 30, 49)
        m = base_motion(e, 2 * w["start15"], 2 * (w["start15"] + 49))
        motion[f"{ds[15:]}/ep{w['episode']}/s{w['start15']}"] = m
        lab = f"move {m['path']:.2f}m yaw {m['yaw']:.0f}d" if m else "no odom"
        for im in f:
            im = im.copy()
            cv2.rectangle(im, (0, 0), (TW, 30), (0, 0, 0), -1)
            cv2.putText(im, ds[15:][:30], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(im, f"ep{w['episode']} {lab}", (3, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (120, 255, 120) if m and m["path"] < 0.1 and m["yaw"] < 10 else (80, 180, 255), 1, cv2.LINE_AA)
            frames.append(im)
    tiles.append(np.stack(frames))
    print(ds[15:], [motion[k] for k in list(motion)[-len(pick):]], flush=True)
n = max(len(t) for t in tiles); rows = (len(tiles) + COLS - 1) // COLS
wr = cv2.VideoWriter(f"{OUT}/windows_grid_tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (COLS * TW, rows * TH))
for k in range(n):
    g = np.zeros((rows * TH, COLS * TW, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, COLS); g[r * TH:(r + 1) * TH, c * TW:(c + 1) * TW] = t[min(k, len(t) - 1)]
    wr.write(g[..., ::-1])
wr.release()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{OUT}/windows_grid_tmp.mp4", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                f"{OUT}/windows_grid.mp4"]); os.remove(f"{OUT}/windows_grid_tmp.mp4")
json.dump(motion, open(f"{OUT}/motion.json", "w"), indent=1)

# rejected moments: frames the classifier marks as legs-visible (one per dataset, up to 18)
rej = []
for ds in sorted(by_ds):
    eps = [e for (d, _), e in index.items() if d == ds and os.path.exists(vpath(e)) and f"{d}/ep{e['episode']:05d}" in free.files]
    random.shuffle(eps)
    for e in eps:
        lf = free[f"{ds}/ep{e['episode']:05d}"]; bad = np.flatnonzero(~lf)
        if len(bad):
            fr = int(random.choice(bad)); im = clip(e, e["t0"] + fr / 30, 1)
            if len(im):
                im = im[0].copy(); cv2.putText(im, ds[15:][:30], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 0), 1, cv2.LINE_AA); rej.append(im)
            break
    if len(rej) == 18:
        break
rows = (len(rej) + COLS - 1) // COLS; sheet = np.zeros((rows * TH, COLS * TW, 3), np.uint8)
for i, im in enumerate(rej):
    r, c = divmod(i, COLS); sheet[r * TH:(r + 1) * TH, c * TW:(c + 1) * TW] = im
cv2.imwrite(f"{OUT}/rejected.jpg", sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
paths = [m["path"] for m in motion.values() if m]; yaws = [m["yaw"] for m in motion.values() if m]
print(f"SAMPLE_DONE {len(tiles)} datasets; base path per window: median {np.median(paths):.2f} m, >0.3 m in {np.mean(np.array(paths) > 0.3) * 100:.0f}%; "
      f"yaw median {np.median(yaws):.0f} deg; rejected stills {len(rej)}")
