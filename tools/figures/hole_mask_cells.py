"""Page image img/hole_mask_cells.jpg (= check/hole_mask_cells.jpg): hole channel per 8x8 cell as max (any hole pixel, old)
vs mean (share of hole pixels, v6 --mask_pool mean). Ran inline on 10-31; recovered from the session transcript.
usage: python hole_mask_cells.py"""
import numpy as np, imageio.v2 as imageio, cv2
R = "/data/datasets/unitree_g1_inspire"; W, H, G = 448, 336, 128
c, t = "Put_Sink_Towel_ep00003_s0501", 56
rd = imageio.get_reader(f"{R}/segments/{c}.mp4"); fr = rd.get_data(t); rd.close()
rob = np.unpackbits(np.load(f"{R}/robot/{c}.npz")["mask"], axis=-1)[..., :W].astype(bool)[t]
hole = cv2.dilate(rob.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
cell = hole.reshape(42, 8, 56, 8)
mx = cell.max((1, 3)).astype(np.float32); mn = cell.mean((1, 3))
up = lambda m: cv2.resize((m * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
def grid(img, step):
    g = img.copy(); g[::step, :] = (g[::step, :] * 0.6).astype(np.uint8); g[:, ::step] = (g[:, ::step] * 0.6).astype(np.uint8); return g
def tag(im, s):
    im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(im, s, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA); return im
x0, y0, x1, y1 = 60, 120, 260, 270  # zoom on the hand
Z = lambda im: cv2.resize(im[y0:y1, x0:x1], ((x1 - x0) * 3, (y1 - y0) * 3), interpolation=cv2.INTER_NEAREST)
a = np.where(hole[..., None], G, fr).astype(np.uint8)
b = cv2.cvtColor(up(mx), cv2.COLOR_GRAY2RGB); cc = cv2.cvtColor(up(mn), cv2.COLOR_GRAY2RGB)
row = np.hstack([tag(grid(Z(a), 24), "source pixels (8x8 grid drawn)"), tag(grid(Z(b), 24), "now: cell = 1 if ANY pixel is hole"), tag(grid(Z(cc), 24), "proposed: cell = share of hole pixels")])
cv2.imwrite(f"{R}/check/hole_mask_cells.jpg", row[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
print("hole pixels %.1f%% of frame; max-mask covers %.1f%%; cells partly hole: %d of %d" % (hole.mean() * 100, mx.mean() * 100, int(((mn > 0) & (mn < 1)).sum()), int((mn > 0).sum())))
