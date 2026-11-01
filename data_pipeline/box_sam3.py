"""Object masks, step 2 with SAM3 only (user, 10-29: no SAM2): for every Claude box (obj_queue/results/*.jsonl, from
obj_sheets.py + obj_queue/README.md) segment the boxed object with the SAM3 image model (box prompt; of the instances it
returns, the one that fits the box best) in the chosen frame, then track that ONE object through the whole segment with
the SAM3 video predictor (add_mask on obj 1, propagate both ways from the seed frame).
Writes obj1/<clip>.npz (lab: (T,H,W) uint8, 1 = the object; mask, frac, name, frame, box) and obj_queue/qa/<clip>.jpg.
usage: /data/repos/sam3/.venv/bin/python box_sam3.py [results glob] [--follow]"""
import glob, json, os, shutil, sys
import cv2, imageio.v2 as imageio, numpy as np, torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model, build_sam3_predictor

R = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire"); Q = f"{R}/obj_queue"; OUT = f"{R}/obj1"
os.makedirs(OUT, exist_ok=True); os.makedirs(f"{Q}/qa", exist_ok=True)
CK = os.environ.get("SAM3_CKPT", "/data/models/sam3/sam3/sam3.pt")
TMP = os.environ.get("SAM3_TMP", R) + f"/_box_sam3_frames_{os.getpid()}"
pred = build_sam3_predictor(checkpoint_path=CK, version="sam3", async_loading_frames=False)
autocast = torch.autocast("cuda", dtype=torch.bfloat16)
proc = Sam3Processor(build_sam3_image_model(checkpoint_path=CK, load_from_HF=False))


def seed_mask(img, box):
    """SAM3 image model with the box as a positive geometric prompt; the returned instance that fits the box best"""
    H, W = img.shape[:2]; x0, y0, x1, y1 = box
    with autocast:
        st = proc.set_image(Image.fromarray(img))
        r = proc.add_geometric_prompt([(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H], True, st)
    if not len(r["scores"]):
        return None
    bm = np.zeros((H, W), bool); bm[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
    best, best_s = None, 0.0
    for sc, mk in zip(r["scores"].float().cpu().numpy(), r["masks"].cpu().numpy().reshape(-1, H, W) > 0):
        if mk.sum() < 30:
            continue
        inside = (mk & bm).sum() / mk.sum(); cover = (mk & bm).sum() / bm.sum()
        s = inside * cover * float(sc)
        if s > best_s:
            best, best_s = mk, s
    return best


def track(frames, k0, mask):
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    for k, f in enumerate(frames):
        Image.fromarray(f).save(f"{TMP}/{k:05d}.jpg", quality=95)
    T, H, W = len(frames), *frames[0].shape[:2]
    out = np.zeros((T, H, W), bool)
    sid = pred.handle_request(dict(type="start_session", resource_path=TMP))["session_id"]
    try:
        pred.handle_request(dict(type="add_mask", session_id=sid, frame_index=k0, obj_id=1, mask=mask))
        for resp in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid, propagation_direction="both", start_frame_index=k0)):
            o = resp["outputs"]; ms = o["out_binary_masks"]; ms = ms.cpu().numpy() if torch.is_tensor(ms) else np.asarray(ms)
            for oid, mk in zip(np.asarray(o["out_obj_ids"]).tolist(), ms):
                if int(oid) == 1:
                    out[resp["frame_index"]] |= mk
    finally:
        pred.handle_request(dict(type="close_session", session_id=sid)); shutil.rmtree(TMP, ignore_errors=True); torch.cuda.empty_cache()
    out[k0] |= mask
    return out


def process(r):
    global n_ok
    c = r["clip"]; dst = f"{OUT}/{c}.npz"
    frames = [f for f in imageio.get_reader(f"{R}/segments/{c}.mp4")]; T, H, W = len(frames), *frames[0].shape[:2]
    k0 = int(np.clip(r["frame"], 0, T - 1))
    box = [float(np.clip(r["box"][0], 0, W - 1)), float(np.clip(r["box"][1], 0, H - 1)), float(np.clip(r["box"][2], 0, W - 1)), float(np.clip(r["box"][3], 0, H - 1))]
    try:
        m0 = seed_mask(frames[k0], box)
        if m0 is None:
            print(f"{c}: SAM3 found nothing in the box", flush=True); return
        mk = track(frames, k0, m0)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); print(f"{c}: CUDA out of memory, skipped", flush=True); return
    frac = mk.reshape(T, -1).mean(1)
    np.savez_compressed(dst, lab=mk.astype(np.uint8), mask=np.packbits(mk, axis=-1), shape=np.array(mk.shape), frac=frac.astype(np.float16),
                        name=r.get("name", ""), frame=k0, box=np.array(box))
    sheet = []
    for k in sorted(set(np.linspace(0, T - 1, 5).astype(int)) | {k0}):
        o = frames[k].copy(); o[mk[k]] = (0.45 * o[mk[k]] + 0.55 * np.array([255, 0, 200])).astype(np.uint8)
        if k == k0:
            cv2.rectangle(o, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
        cv2.putText(o, f"f{k}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        sheet.append(cv2.resize(o, (224, 168)))
    cv2.putText(sheet[0], r.get("name", "")[:28], (5, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    imageio.imwrite(f"{Q}/qa/{c}.jpg", np.hstack(sheet))
    n_ok += 1; print(f"{c}: {r.get('name', '')}, seed f{k0}, visible in {(frac > 0.0005).mean() * 100:.0f}% of frames", flush=True)


FOLLOW = "--follow" in sys.argv  # keep picking up new results/bNNN.jsonl until every batch is labelled and tracked
pattern = next((x for x in sys.argv[1:] if not x.startswith("--")), f"{Q}/results/*.jsonl")
n_ok = n_none = 0; tried = set()
while True:
    rows = []
    for f in sorted(glob.glob(pattern)):
        try:
            rows += [json.loads(l) for l in open(f) if l.strip()]
        except json.JSONDecodeError:
            pass  # being written
    todo = [r for r in rows if not r.get("none") and r["clip"] not in tried and not os.path.exists(f"{OUT}/{r['clip']}.npz")]
    if not todo:
        if not FOLLOW:
            break
        import subprocess
        chk = subprocess.run(["python3", f"{Q}/check_results.py"], capture_output=True, text=True).stdout
        if " 0 to do" in chk.splitlines()[0]:
            break
        import time; time.sleep(60); continue
    for r in todo:
        tried.add(r["clip"])
        process(r)
n_none = sum(bool(r.get("none")) for r in rows)
print(f"BOX_SAM3_DONE {n_ok} tracked this run, {n_none} none, {len(os.listdir(OUT))} object maps in {OUT}")

