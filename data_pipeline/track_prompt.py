"""SAM3 video tracking of one text prompt over many short clips: union of every tracked instance seen in at least
MIN_FRAC of the frames. Used for the renderer-training data (robot = "robot", objects = per-dataset prompts).

usage: /data/repos/sam3/.venv/bin/python track_prompt.py <jobs.json> <out dir> <name>
  jobs.json: [{"clip": "PickApple_ep003", "rgb": ".../rgb/PickApple_ep003.mp4", "prompts": ["apple"]}, ...]
     or {"clip", "video", "start", "end", "size": [W, H], "prompts"}: frames start, start+2, ... < end of the video, resized
  env: JOB_RANK / JOB_WORLD split the jobs between processes; EDGE=1 keeps only tracks that touch the left / right / bottom
       image border in >= 20% of their frames (robot arms come in from the edges; banners, bags, tape on the floor do not)
  writes <out>/<name>/<clip>.npz (mask packed bits, shape, frac; INSTANCES=1 also lab: (T,H,W) uint8 instance ids,
  1..K in order of first appearance, one per kept track of every prompt) and <out>/qa_<name>/<clip>.jpg
"""
import json
import os
import shutil
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

jobs_json, out, name = sys.argv[1:4]
MIN_FRAC = float(os.environ.get("MIN_FRAC", 0.1))
RANK, WORLD = int(os.environ.get("JOB_RANK", 0)), int(os.environ.get("JOB_WORLD", 1))  # read before SAM3 sets RANK for its own use
os.makedirs(f"{out}/{name}", exist_ok=True)
os.makedirs(f"{out}/qa_{name}", exist_ok=True)
TMP = os.environ.get("SAM3_TMP", "/data/datasets/unitree_g1_dex3") + f"/_sam3_frames_{os.getpid()}"
from sam3.model_builder import build_sam3_predictor  # noqa: E402

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
pred = build_sam3_predictor(checkpoint_path=os.environ.get("SAM3_CKPT", "/data/models/sam3/sam3/sam3.pt"), version="sam3", async_loading_frames=False)
pred.model.score_threshold_detection = 0.4
pred.model.new_det_thresh = 0.4
pred.model.hotstart_delay = 0

EDGE = os.environ.get("EDGE") == "1"
TMP = TMP + f"_r{RANK}"


def edge_track(dd):
    vis = [m for m in dd.values() if m.sum() > 30]
    touch = sum(int(m[:, :3].any() or m[:, -3:].any() or m[-3:, :].any()) for m in vis)
    return len(vis) > 0 and touch >= 0.2 * len(vis)


for job in json.load(open(jobs_json))[RANK::WORLD]:
    try:
        dst = f"{out}/{name}/{job['clip']}.npz"
        if os.path.exists(dst):
            continue
        if "video" in job:
            frames = []
            rd = imageio.get_reader(job["video"])
            for k, fr in enumerate(rd):
                if k >= job["end"]:
                    break
                if k >= job["start"] and (k - job["start"]) % 2 == 0:
                    frames.append(cv2.resize(fr, tuple(job["size"]), interpolation=cv2.INTER_AREA))
            rd.close()
        else:
            frames = [f for f in imageio.get_reader(job["rgb"])]
        T, H, W = len(frames), frames[0].shape[0], frames[0].shape[1]
        shutil.rmtree(TMP, ignore_errors=True)
        os.makedirs(TMP)
        for k, f in enumerate(frames):
            Image.fromarray(f).save(f"{TMP}/{k:05d}.jpg", quality=95)
        M = np.zeros((T, H, W), bool)
        L = np.zeros((T, H, W), np.uint8)
        tracks = []
        for prompt in job["prompts"]:
            sid = pred.handle_request(dict(type="start_session", resource_path=TMP))["session_id"]
            pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=prompt))
            per = {}
            for resp in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
                o = resp["outputs"]
                ms = o["out_binary_masks"]
                ms = ms.cpu().numpy() if torch.is_tensor(ms) else np.asarray(ms)
                for oid, mk in zip(np.asarray(o["out_obj_ids"]).tolist(), ms):
                    per.setdefault(int(oid), {})[resp["frame_index"]] = mk
            pred.handle_request(dict(type="close_session", session_id=sid))
            for dd in per.values():
                if sum(int(m.sum() > 30) for m in dd.values()) >= MIN_FRAC * T and (not EDGE or edge_track(dd)):
                    tracks.append(dd)
                    for f, mk in dd.items():
                        M[f] |= mk
        tracks.sort(key=lambda dd: min(f for f, m in dd.items() if m.sum() > 30) if any(m.sum() > 30 for m in dd.values()) else T)
        for i, dd in enumerate(tracks[:255]):
            for f, mk in dd.items():
                L[f][mk & (L[f] == 0)] = i + 1
        torch.cuda.empty_cache()
        shutil.rmtree(TMP, ignore_errors=True)
        frac = M.reshape(T, -1).mean(1)
        extra = dict(lab=L) if os.environ.get("INSTANCES") == "1" else {}
        np.savez_compressed(dst, mask=np.packbits(M, axis=-1), shape=np.array(M.shape), frac=frac.astype(np.float16), **extra)
        sheet = []
        pal = np.array([[255, 0, 200], [0, 200, 255], [255, 200, 0], [0, 255, 100], [255, 80, 0], [120, 80, 255]], np.uint8)
        for k in np.linspace(0, T - 1, 6).astype(int):
            o = frames[k].copy()
            for i in range(1, int(L[k].max()) + 1):
                mk = L[k] == i
                o[mk] = (0.4 * o[mk] + 0.6 * pal[(i - 1) % len(pal)]).astype(np.uint8)
            sheet.append(cv2.resize(o, (224, 168)))
        imageio.imwrite(f"{out}/qa_{name}/{job['clip']}.jpg", np.hstack(sheet))
        print(f"{job['clip']} {name}: {len(tracks)} tracks, {T} frames, visible in {(frac > 0.0005).mean() * 100:.0f}% of frames, median area {np.median(frac[frac > 0.0005]) * 100 if (frac > 0.0005).any() else 0:.2f}%", flush=True)
    except torch.OutOfMemoryError:  # crowded scenes (a shelf of 'vegetable' / 'bottle') can track dozens of objects:
        print(f"{job['clip']} {name}: CUDA out of memory, skipped", flush=True)  # skip the clip, keep the worker alive
        try:
            pred.handle_request(dict(type="close_session", session_id=sid))
        except Exception:
            pass
        torch.cuda.empty_cache(); shutil.rmtree(TMP, ignore_errors=True)
print("TRACK_DONE")
