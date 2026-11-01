"""Ghost holes (10-31): at inference the source is holed where the HUMAN hand was and the sim robot is drawn where the
retargeted robot is; the two overlap but not exactly (tissue clip: per arm IoU ~0.55, the human about 0.7x the robot's
area, a pure shift gains only 0.03, a 1-3 frame lag nothing; 21 % of the human hole lies outside the robot, 90 % of it
within 18 px). Training imitates that with a second, finger-detailed copy of the robot's own mask per arm: rotated about
the point where the arm enters the frame (the hand moves most, like a differently posed human arm), scaled, shifted a
little, thinned, and drifting slowly over the window. Only the part outside the robot matters (the robot itself is holed
anyway). Defaults calibrated on 80 training windows against the tissue clip: ghost outside the robot 20 % (clip 21 %),
distance to the robot p50/75/90/95 = 5/10/17/23 px (5/11/18/23), IoU 0.51 (0.55), area ratio 0.71 (0.70). Not part of the
upstream repo."""
import cv2
import numpy as np

K3 = np.ones((3, 3), np.uint8)


def entry_point(m, H, W):
    """where an arm mask enters the frame: mean of its pixels on the image border (else its point nearest to the border)"""
    ys, xs = np.nonzero(m)
    edge = (ys >= H - 2) | (xs <= 1) | (xs >= W - 2) | (ys <= 1)
    if edge.any():
        return float(xs[edge].mean()), float(ys[edge].mean())
    d = np.minimum.reduce([ys, H - 1 - ys, xs, W - 1 - xs]); i = int(np.argmin(d))
    return float(xs[i]), float(ys[i])


def ghost(rob, rng, rot_deg=8.0, shift_px=6.0, scale=(0.82, 1.0), thin=(1, 3)):
    """rob (T,H,W) bool robot masks of the window -> ghost (T,H,W) bool (a thin-margin copy, per arm transformed)"""
    T, H, W = rob.shape
    out = np.zeros_like(rob)
    for side in (0, 1):  # left / right half of the image = left / right arm
        sl = slice(0, W // 2) if side == 0 else slice(W // 2, W)
        part = np.zeros_like(rob); part[:, :, sl] = rob[:, :, sl]
        ref = part.any(0)
        if ref.sum() < 100:
            continue
        cx, cy = entry_point(ref, H, W)
        a = [rng.normal(0, rot_deg, 2), rng.normal(0, shift_px, 2), rng.normal(0, shift_px, 2), rng.uniform(*scale, 2)]  # start, end
        er = int(rng.integers(thin[0], thin[1] + 1))
        for t in range(T):
            if not part[t].any():
                continue
            u = t / max(1, T - 1)
            th, dx, dy, s = [v[0] * (1 - u) + v[1] * u for v in a]
            M = cv2.getRotationMatrix2D((cx, cy), th, s); M[0, 2] += dx; M[1, 2] += dy
            g = cv2.warpAffine(part[t].astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
            if er:
                g = cv2.erode(g, np.ones((2 * er + 1, 2 * er + 1), np.uint8))
            out[t] |= cv2.dilate(g, K3) > 0
    return out
