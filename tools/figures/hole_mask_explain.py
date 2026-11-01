"""Page image img/hole_mask_explain.jpg (= check/hole_mask_explain.jpg): why the extra hole-mask channel is needed (real
grey surfaces vs grey holes), Put_Sink_Towel_ep00003_s0501 frame 56. Ran inline on 10-31; recovered from the session
transcript. usage: python hole_mask_explain.py"""
import numpy as np, imageio.v2 as imageio, cv2
R = "/data/datasets/unitree_g1_inspire"; W, H, G = 448, 336, 128
c, t = "Put_Sink_Towel_ep00003_s0501", 56
rd = imageio.get_reader(f"{R}/segments/{c}.mp4"); fr = rd.get_data(t); rd.close()
rob = np.unpackbits(np.load(f"{R}/robot/{c}.npz")["mask"], axis=-1)[..., :W].astype(bool)[t]
hole = cv2.dilate(rob.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
src = np.where(hole[..., None], G, fr).astype(np.uint8)
near = (np.abs(fr.astype(int) - 128).max(-1) < 14) & ~hole
def tag(im, s, col=(255, 255, 255)):
    im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(im, s, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA); return im
a = tag(fr, "1 real frame")
b = tag(src, "2 source: robot dug out (grey)")
# 3: highlight: real grey surfaces (yellow outline) vs holes (red outline)
cimg = src.copy()
for m, col in [(near, (255, 210, 0)), (hole, (230, 40, 40))]:
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(cimg, [x for x in cnt if cv2.contourArea(x) > 30], -1, col, 2)
cc = tag(cimg, "3 yellow: real grey stuff / red: holes")
m8 = hole.reshape(42, 8, 56, 8).max((1, 3)).astype(np.uint8) * 255
d = tag(cv2.cvtColor(cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB), "4 hole mask: white = hole")
out = np.vstack([np.hstack([a, b]), np.hstack([cc, d])])
cv2.imwrite(f"{R}/check/hole_mask_explain.jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
print("grey-like share of the real frame outside the holes: %.0f%%" % (near.mean() * 100))
