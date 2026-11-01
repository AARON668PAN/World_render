"""Which SAM3 text prompt finds the G1's own legs seen from the head camera? A few hand-picked frames (legs / no legs)."""
import json, subprocess, numpy as np, cv2, torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
R = "/data/datasets/unitree_g1_inspire"; idx = {(e["dataset"], e["episode"]): e for e in json.load(open(f"{R}/index.json"))}
def fr(ds, ep, s, k):
    e = idx[("G1_WBT_Inspire_" + ds, ep)]; vp = f"{R}/official/{e['dataset']}/videos/observation.images.head_stereo_left/chunk-{e['chunk']:03d}/file-{e['file']:03d}.mp4"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{e['t0'] + 2 * (s + k) / 30:.3f}", "-i", vp, "-frames:v", "1", "-vf", "scale=448:336", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(336, 448, 3)
CASES = [("Put_Clothes_into_Washing_Machine", 10, 326, 24, 1), ("Put_Clothes_into_Washing_Machine", 16, 190, 24, 1), ("Clean_Living_Room", 27, 110, 24, 1),
         ("Pick_Up_Drinks", 1, 187, 24, 1), ("Walk_To_Table_Put_Cups_And_Rack_Plates_In_Dishwasher", 10, 292, 0, 1), ("Pick_Up_Clothes", 7, 145, 24, -1),
         ("Pick_Up_Drinks", 7, 332, 24, -1), ("Put_Sink_Towel", 5, 432, 24, 0), ("Empty_Folder_In_Cabinet", 44, 153, 24, 0), ("Put_Plate_Into_Dishwasher", 4, 1, 24, 0),
         ("Push_Wheelchair", 28, 74, 24, 0), ("Plastic_Box_Parts", 2, 1508, 24, 0), ("Tidy_Living_Room_Slippers", 8, 456, 24, 0), ("Sort_Fruit", 8, 534, 24, 0)]
PROMPTS = ["leg", "knee", "robot leg", "robotic leg", "metal leg", "foot", "robot foot", "thigh"]
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
proc = Sam3Processor(build_sam3_image_model(checkpoint_path="/data/models/sam3/sam3/sam3.pt", load_from_HF=False))
rows = []
for ds, ep, s, k, gt in CASES:
    img = fr(ds, ep, s, k); st = proc.set_image(Image.fromarray(img)); row = [cv2.resize(img, (224, 168))]; line = []
    for p in PROMPTS:
        r = proc.set_text_prompt(state=st, prompt=p); m = np.zeros((336, 448), bool); best = 0.0
        if len(r["scores"]):
            for sc, mk in zip(r["scores"].float().cpu().numpy(), r["masks"].cpu().numpy().reshape(-1, 336, 448) > 0):
                if sc >= 0.3 and mk.mean() > 0.003: m |= mk; best = max(best, float(sc))
        v = img.copy(); v[m] = (0.4 * v[m] + [150, 0, 0]).astype(np.uint8); v = cv2.resize(v, (224, 168))
        lowest = np.flatnonzero(m.any(1)).max() / 336 if m.any() else 0
        cv2.putText(v, f"{p} {best:.2f} y{lowest:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
        row.append(v); line.append(f"{p}:{best:.2f}")
    cv2.putText(row[0], f"gt={gt} {ds[:18]} {ep}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    rows.append(np.concatenate(row, 1)); print(gt, ds, ep, " ".join(line), flush=True)
sheet = np.concatenate(rows, 0); cv2.imwrite(f"{R}/check/prompts/sheet_a.jpg", sheet[:7 * 168, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
cv2.imwrite(f"{R}/check/prompts/sheet_b.jpg", sheet[7 * 168:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
