"""Page image img/channels.jpg (= check/channels_explain.jpg): control A, source and hole mask per 8x8 cell, training vs
inference. Reads page_images.py outputs from IMG_DIR. NOTE: draws the hole cell as max (pre-v6); v6 uses mean.
Ran inline on 10-31; recovered from the session transcript. usage: python channels_figure.py [IMG_DIR]"""
import sys
import cv2, numpy as np
P = sys.argv[1] if len(sys.argv) > 1 else "/data/datasets/unitree_g1_inspire/pipeline_page/img"
def tag(im, s):
    im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(im, s, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA); return im
r = lambda n: cv2.imread(f"{P}/{n}.jpg")
def cells(n):  # what the model gets: one value per 8x8 pixel cell (max), shown back at full size
    m = cv2.imread(f"{P}/{n}.jpg", 0) > 127
    c = m.reshape(42, 8, 56, 8).max((1, 3)).astype(np.uint8) * 255
    return cv2.cvtColor(cv2.resize(c, (448, 336), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
row1 = np.hstack([tag(r("tr_ctrlA"), "TRAIN control A (16 ch): robot + object"), tag(r("tr_source"), "source video (16 ch): scene, holes grey"), tag(cells("tr_holemask"), "hole mask (1 ch): where the holes are")])
row2 = np.hstack([tag(r("inf_ctrlA"), "INFER control A (16 ch)"), tag(r("inf_source"), "source video (16 ch)"), tag(cells("inf_holemask"), "hole mask (1 ch)")])
cv2.imwrite("/data/datasets/unitree_g1_inspire/check/channels_explain.jpg", np.vstack([row1, row2]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("ok")
