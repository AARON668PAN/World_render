"""Cut training segments out of the G1 WBT Inspire left-head video files (each file decoded once): for each chosen window
(windows.json from leg_clf.py: 49 frames at 15 fps, leg-free by the joint-state classifier) the segment from 2 s before
to 2 s after it (15 fps, 448x336), so the source-video shift has room; the per-frame leg-free flags come along.
Choice: the first window of every episode, then further windows round-robin until N.
usage: python extract_windows.py [N=1300]   (sam3 venv: imageio, cv2)  -> segments/<id>.mp4 + segments.json"""
import json, os, sys
from collections import defaultdict
import cv2, imageio.v2 as imageio, numpy as np

ROOT = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire")  # dataset work dir
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1300
MARGIN = 30  # frames at 15 fps (2 s)
W, H = 448, 336
os.makedirs(f"{ROOT}/segments", exist_ok=True)
wins = json.load(open(f"{ROOT}/windows.json")); free = np.load(f"{ROOT}/leg_free.npz")
index = {(e["dataset"], e["episode"]): e for e in json.load(open(f"{ROOT}/index.json"))}
by_ep = defaultdict(list)
for w in wins:
    by_ep[(w["dataset"], w["episode"])].append(w)
chosen, rnd = [], 0
while len(chosen) < N and any(len(v) > rnd for v in by_ep.values()):
    for k, v in sorted(by_ep.items()):
        if len(v) > rnd and len(chosen) < N:
            chosen.append(v[rnd])
    rnd += 1
jobs = defaultdict(list)  # video file -> segments
segs = []
for w in chosen:
    e = index[(w["dataset"], w["episode"])]; f0 = round(e["t0"] * 30)  # video frame of episode frame 0
    n15 = (e["length"] + 1) // 2
    a = max(0, w["start15"] - 1 - MARGIN); b = min(n15, w["start15"] + 49 + MARGIN)  # 15-fps indices [a, b)
    sid = f"{w['dataset'][15:]}_ep{w['episode']:05d}_s{w['start15']:04d}"
    lf = free[f"{w['dataset']}/ep{w['episode']:05d}"][::2][a:b]
    seg = dict(id=sid, dataset=w["dataset"], episode=w["episode"], a15=a, b15=b, win_start=w["start15"] - a, leg_free=lf.astype(int).tolist(),
               raw=[f0 + 2 * i for i in range(a, b)])
    segs.append(seg)
    jobs[f"{ROOT}/official/{w['dataset']}/videos/observation.images.head_stereo_left/chunk-{e['chunk']:03d}/file-{e['file']:03d}.mp4"].append(seg)
done = 0
for vp, ss in sorted(jobs.items()):
    if not os.path.exists(vp):
        print("missing", vp, flush=True); continue
    need = {}
    for s in ss:
        if os.path.exists(f"{ROOT}/segments/{s['id']}.mp4"):
            continue
        for k, r in enumerate(s["raw"]):
            need.setdefault(r, []).append((s["id"], k))
    if not need:
        continue
    buf = defaultdict(dict); last = max(need)
    for v, fr in enumerate(imageio.get_reader(vp)):
        if v > last:
            break
        if v in need:
            img = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            for sid, k in need[v]:
                buf[sid][k] = img
    for s in ss:
        if s["id"] in buf and len(buf[s["id"]]) == len(s["raw"]):
            w_ = imageio.get_writer(f"{ROOT}/segments/{s['id']}.mp4", fps=15, codec="libx264", quality=9, macro_block_size=8)
            for k in range(len(s["raw"])):
                w_.append_data(buf[s["id"]][k])
            w_.close(); done += 1
    print(f"{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(vp)))))} {os.path.basename(vp)}: {len(ss)} segments", flush=True)
json.dump(segs, open(f"{ROOT}/segments.json", "w"))
print(f"EXTRACT_DONE {done} new segments, {len(segs)} chosen from {len(by_ep)} episodes")
