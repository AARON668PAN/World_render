"""Leg labels for the leg-visibility classifier: SAM3 on sampled left-head frames of the downloaded G1 WBT Inspire video files.
A frame shows the robot's legs if a "leg" / "robot foot" / "knee" mask (score >= 0.4, > 0.3% of the image) touches the
bottom image edge (the robot's own legs come in from the bottom; people's legs and chair legs elsewhere are ignored).
usage: /data/repos/sam3/.venv/bin/python leg_label.py [file indices, default 0] [every N video frames, default 90]
writes leg_labels.json: [{dataset, episode, frame, leg, area}]"""
import glob, json, os, sys
import cv2, imageio.v2 as imageio, numpy as np, torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

ROOT = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire")  # dataset work dir
FILES = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0").split(",")]
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 90
index = json.load(open(f"{ROOT}/index.json"))
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
proc = Sam3Processor(build_sam3_image_model(checkpoint_path=os.environ.get("SAM3_CKPT", "/data/models/sam3/sam3/sam3.pt"), load_from_HF=False))
out_path = f"{ROOT}/leg_labels.json"
labels = json.load(open(out_path)) if os.path.exists(out_path) else []
done = {(l["dataset"], l["episode"], l["frame"]) for l in labels}
for ds in sorted({e["dataset"] for e in index}):
    for fi in FILES:
        vp = f"{ROOT}/official/{ds}/videos/observation.images.head_stereo_left/chunk-000/file-{fi:03d}.mp4"
        if not os.path.exists(vp):
            print("missing", vp, flush=True); continue
        eps = [e for e in index if e["dataset"] == ds and e["file"] == fi]
        spans = [(round(e["t0"] * 30), round(e["t0"] * 30) + e["length"], e["episode"]) for e in eps]
        n = 0
        for v, frame in enumerate(imageio.get_reader(vp)):
            if v % EVERY:
                continue
            hit = [(a, ep) for a, b, ep in spans if a <= v < b]
            if not hit or (ds, hit[0][1], v - hit[0][0]) in done:
                continue
            a, ep = hit[0]
            img = cv2.resize(frame, (448, 336), interpolation=cv2.INTER_AREA)
            st = proc.set_image(Image.fromarray(img)); m = np.zeros((336, 448), bool)
            for p in ("leg", "robot foot", "knee"):
                r = proc.set_text_prompt(state=st, prompt=p)
                if len(r["scores"]):
                    for s, mk in zip(r["scores"].float().cpu().numpy(), r["masks"].cpu().numpy().reshape(-1, 336, 448) > 0):
                        if s >= 0.4 and mk[-6:].any() and mk.mean() > 0.003:
                            m |= mk
            labels.append(dict(dataset=ds, episode=ep, frame=v - a, leg=bool(m.any()), area=float(m.mean())))
            n += 1
        json.dump(labels, open(out_path, "w"))
        print(f"{ds} file {fi}: {n} frames labelled, legs in {np.mean([l['leg'] for l in labels if l['dataset'] == ds]) * 100:.0f}%", flush=True)
print("LEG_LABEL_DONE", len(labels))
