"""Ghost holes on 6 training windows next to the tissue-clip inference source (red = holed where no robot is): page image
img/holes.jpg (= check/ghost_holes_calibrated.jpg) + check/ghost_holes_calibrated.mp4. Ran inline on 10-31 (isaaclab_newton
python); recovered from the session transcript. usage: python ghost_holes_figure.py"""
import json, os, cv2, numpy as np, imageio.v2 as imageio, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "renderer"))  # ghost_holes of this repo
import ghost_holes as GH
R = "/data/datasets/unitree_g1_inspire"; E = "/data/repos/chord/local/ego/tissue_box"; H, W, G = 336, 448, 128
RED = np.array([230, 40, 40], np.uint8); K3 = np.ones((3, 3), np.uint8); dil = lambda m: cv2.dilate(m.astype(np.uint8), K3) > 0
def tag(im, s):
    im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(im, s, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA); return im
segs = {s["id"]: s for s in json.load(open(f"{R}/segments.json"))}
clips = ["Put_Vegetables_Into_Basket_ep00012_s0394", "Put_Plate_Into_Dishwasher_ep00010_s0051", "Put_Sink_Towel_ep00009_s0094",
         "Empty_Folder_In_Cabinet_ep00038_s0135", "Pick_Up_Drinks_v0_ep00046_s0466", "Sort_Fruit_ep00011_s0559"]
rng = np.random.default_rng(7); rows, vids = [], {}
for c in clips:
    fr = np.stack([f for f in imageio.get_reader(f"{R}/segments/{c}.mp4")]); s0 = segs[c]["win_start"]; seg = slice(s0, s0 + 49)
    rob = np.unpackbits(np.load(f"{R}/robot/{c}.npz")["mask"], axis=-1)[..., :W].astype(bool)[seg]; F = fr[seg]
    gh = GH.ghost(rob, rng); own = np.stack([dil(m) for m in rob])
    def frames(k):
        a = np.where(own[k][..., None], G, F[k]).astype(np.uint8); b = a.copy(); b[gh[k] & ~own[k]] = RED; return a, b
    a, b = frames(30)
    rows.append(np.hstack([tag(F[30], "training target"), tag(a, "exact: robot + 1 px"), tag(b, "+ ghost (red = no robot there)")]))
    vids[c] = [np.hstack([tag(F[k], f"target f{k}"), tag(frames(k)[1], "source: robot + ghost (red)")]) for k in range(49)]
z = np.load(f"{E}/human_g1i_letterbox_obj_realcol.npz"); band = z["band"]
lab = np.load(f"{E}/depth_compare/v7_letterbox/sim_depth.npz")["label"][::2]
ht = np.load(f"{E}/align/human_table_masks.npz"); vw = int(ht["shape"][2])
hum30 = (np.unpackbits(ht["hand"], axis=-1) | np.unpackbits(ht["forearm"], axis=-1))[..., :vw]
vid = [f for i, f in enumerate(imageio.get_reader("/data/repos/chord/reconstruction/assets/tissue_box.mp4")) if i % 2 == 0]
def inf(k):
    rb = np.isin(lab[k], (3, 4)); hm = np.zeros((H, W), bool); hm[42:294] = cv2.resize(hum30[2 * k], (W, 252), interpolation=cv2.INTER_NEAREST) > 0
    img = np.full((H, W, 3), G, np.uint8); img[42:294] = cv2.resize(vid[k], (W, 252), interpolation=cv2.INTER_AREA)
    own = dil(rb) | band; s = np.where(own[..., None], G, img).astype(np.uint8); s[dil(hm) & ~own] = RED; return img, s
img, s = inf(60)
rows.append(np.hstack([tag(img, "INFERENCE: human video"), tag(z["ctrl"][60], "control: sim robot"), tag(s, "source: red = human hole, no robot")]))
cv2.imwrite(f"{R}/check/ghost_holes_calibrated.jpg", np.vstack(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
w = imageio.get_writer(f"{R}/check/ghost_holes_calibrated.mp4", fps=8, codec="libx264", quality=7, macro_block_size=8)
for k in range(49):
    ii, ss = inf(49 + k)
    top = np.hstack([vids[clips[0]][k], vids[clips[2]][k]]); bot = np.hstack([tag(ii, f"INFERENCE human f{49 + k}"), tag(ss, "source: sim robot + human hole (red)"), vids[clips[1]][k]])
    w.append_data(np.vstack([top, bot]))
w.close(); print("ok")
