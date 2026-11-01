"""Object masks, step 1 (the user's pick, 10-29: Claude boxes the ONE manipulated object, SAM3 tracks it (box_sam3.py; SAM2 was the first plan); SAM3 per-noun
tracking found every bottle on the shelf). Per leg-free segment one sheet: 6 frames of the 49-frame training window
(448x336 each, 3 x 2), each labelled with its segment frame index and pixel ticks every 50 px, for a Claude agent that
returns the object, the frame and the box (obj_queue/README.md). Batches of 20 sheets. (The 9-frame sheets of the relabel
batches fix000..fix004, clips in fix_clips.json, were made by a variant that ran inline and is not in this repo.)
usage: python3 obj_sheets.py [--only a,b]   -> obj_queue/sheets/<clip>.jpg, obj_queue/batches/bNNN.txt"""
import json, os, sys
import cv2, imageio.v2 as imageio, numpy as np

R = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire"); Q = f"{R}/obj_queue"
os.makedirs(f"{Q}/sheets", exist_ok=True); os.makedirs(f"{Q}/batches", exist_ok=True); os.makedirs(f"{Q}/results", exist_ok=True)
segs = {s["id"]: s for s in json.load(open(f"{R}/segments.json"))}
clips = [j["clip"] for j in json.load(open(f"{R}/jobs_robot.json"))]
if "--only" in sys.argv:
    clips = sys.argv[sys.argv.index("--only") + 1].split(",")
W, H = 448, 336
for c in clips:
    out = f"{Q}/sheets/{c}.jpg"
    if os.path.exists(out):
        continue
    s0 = segs[c]["win_start"]; rd = imageio.get_reader(f"{R}/segments/{c}.mp4"); n = rd.count_frames()
    ks = [min(n - 1, s0 + k) for k in (0, 10, 19, 29, 38, 48)]
    tiles = []
    for k in ks:
        im = rd.get_data(k).copy()
        for x in range(0, W, 50):
            cv2.line(im, (x, 0), (x, 6), (255, 255, 0), 1); cv2.putText(im, str(x), (x + 2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1, cv2.LINE_AA)
        for y in range(50, H, 50):
            cv2.line(im, (0, y), (6, y), (255, 255, 0), 1); cv2.putText(im, str(y), (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(im, (W - 70, 0), (W, 20), (0, 0, 0), -1); cv2.putText(im, f"f{k}", (W - 64, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    rd.close()
    sheet = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
    cv2.imwrite(out, sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
done = sorted(f[:-4] for f in os.listdir(f"{Q}/sheets"))
labelled = set()
for f in os.listdir(f"{Q}/results"):
    labelled |= {json.loads(l)["clip"] for l in open(f"{Q}/results/{f}") if l.strip()}
order = [c for c in clips if c in done and c not in labelled]
if "--only" not in sys.argv:  # batches of 20 sheets (a labeller that reads many more images loses the old ones)
    for b in range(0, len(order), 20):
        open(f"{Q}/batches/b{b // 20:03d}.txt", "w").write("\n".join(order[b:b + 20]) + "\n")
print(len(done), "sheets,", len(order), "still to label,", (len(order) + 19) // 20, "batches")
