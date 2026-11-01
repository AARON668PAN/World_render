"""C1 object-depth encoding for the control video (10-30), shared by cache_g1i.py (training, VDA inverse depth from
obj_depth_batch.py) and prep_human_g1i.py (inference, 1/sim depth), so both draw the object the same way:
  hue        = distance: inverse depth normalised once per window over object + robot pixels (keeps how far the object
               is from the hand), turbo colour map
  brightness = the object's own shape: its inverse depth normalised inside the object over the window, the span
               stretched over the object's size, normals -> Lambert shading with a fixed light (upper left, front)
Not part of the upstream repo."""
import cv2
import numpy as np


def window_ranges(inv, obj, rob):
    """(joint lo, hi) over object + robot and (local lo, hi) over the object, for a window of inverse depth (T,H,W).
    joint hi = the larger of the object's 98th and the robot's 60th percentile: the hand is usually the nearest thing, and
    letting its very nearest pixels set the top squeezed every object into the blue end of the colour map"""
    vo = inv[obj].astype(np.float32); vr = inv[rob].astype(np.float32)
    vj = np.concatenate([vo, vr])
    jl = np.percentile(vj, 2) if vj.size else 0.0
    jh = max(np.percentile(vo, 98) if vo.size else 1.0, np.percentile(vr, 60) if vr.size else 1.0)
    ol, oh = (np.percentile(vo, 2), np.percentile(vo, 98)) if vo.size else (0.0, 1.0)
    return (float(jl), float(max(jh, jl + 1e-3))), (float(ol), float(max(oh, ol + 1e-3)))


def shade(local, m, slope=1.2):
    if m.sum() < 20:
        return np.zeros(m.shape, np.float32)
    size = np.sqrt(m.sum())
    z = cv2.GaussianBlur(np.where(m, local, local[m].mean()).astype(np.float32), (0, 0), 1.5) * size * slope
    gy, gx = np.gradient(z)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array([-0.5, -0.6, 0.62]); L /= np.linalg.norm(L)
    return np.clip(n @ L, 0, 1)


def encode_frame(inv, obj, joint, local):
    """RGB uint8 colours for the object pixels of one frame (others untouched by the caller)"""
    jv = np.clip((inv.astype(np.float32) - joint[0]) / (joint[1] - joint[0]), 0, 1)
    lv = np.clip((inv.astype(np.float32) - local[0]) / (local[1] - local[0]), 0, 1)
    c = cv2.applyColorMap((jv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32)
    s = shade(lv, obj)[..., None]
    return np.clip(c * (0.25 + 1.0 * s), 0, 255).astype(np.uint8)


def contact_fill(inv, obj, rob, px=8):
    """object pixels within px of the robot take the value of the nearest object pixel further away: depth estimates
    bleed the (near) hand into the object where they touch, which showed as a red rim"""
    from scipy.ndimage import distance_transform_edt
    out = inv.copy(); k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    for t in range(len(inv)):
        near = obj[t] & (cv2.dilate(rob[t].astype(np.uint8), k) > 0); good = obj[t] & ~near
        if near.any() and good.sum() > 20:
            _, (iy, ix) = distance_transform_edt(~good, return_indices=True); out[t][near] = inv[t][iy[near], ix[near]]
    return out


def paint(ctrl, inv, obj, rob):
    """ctrl (T,H,W,3) uint8 in place: object pixels in C1 colours, with ranges from this window"""
    inv = contact_fill(inv.astype(np.float32), obj, rob)
    for t in range(len(inv)):  # thin streaks (depth bleeding along an edge the robot mask misses): 5x5 median inside the object
        if obj[t].sum() > 50:
            med = cv2.medianBlur(np.where(obj[t], inv[t], 0).astype(np.float32), 5); inv[t][obj[t]] = med[obj[t]]
    joint, local = window_ranges(inv, obj, rob)
    for t in range(len(ctrl)):
        if obj[t].any():
            col = encode_frame(inv[t], obj[t], joint, local); ctrl[t][obj[t]] = col[obj[t]]
    return ctrl
