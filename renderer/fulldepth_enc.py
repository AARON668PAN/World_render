"""Control B of the v6 renderer (10-31): whole-frame depth drawn as turbo(depth) x Lambert shading, normalised once per
49-frame window over the whole frame (hue = distance, brightness = shape). Training: VDA of the real robot video
(full_depth_batch.py, uint8 per segment) with the handled object replaced by its refined depth (obj_depth/, crop pass for
small objects) and contact fill near the robot, plus augmentations that imitate the inference composite (sim robot depth
fitted onto the VDA scale with a few % error, sharp sim edges, patches of filled background). Inference: the composite
made by prep_human_depth.py. Not part of the upstream repo."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

import objdepth_enc as OD

L = np.array([-0.5, -0.6, 0.62]); L /= np.linalg.norm(L)


def load_full(path, T=None):
    z = np.load(path)
    inv = z["d"].astype(np.float32) / 255 * (float(z["hi"]) - float(z["lo"])) + float(z["lo"])
    return inv if T is None else inv[:T]


def refine_object(inv, od, obj):
    """inv, od (T,H,W) one window; obj (T,H,W) bool: object pixels take the refined object depth, affine-fitted"""
    if obj.sum() < 200:
        return inv
    a, b = np.polyfit(od[obj][::3], inv[obj][::3], 1)
    out = inv.copy(); out[obj] = a * od[obj] + b
    return out


def shading(n01, slope=250.0):
    z = cv2.GaussianBlur(n01.astype(np.float32), (0, 0), 1.5) * slope
    gy, gx = np.gradient(z)
    nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    return np.clip(nrm @ L, 0, 1)


def encode_window(inv):
    """(T,H,W) inverse depth of one window -> (T,H,W,3) uint8 control frames"""
    lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
    n = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    out = np.empty(inv.shape + (3,), np.uint8)
    for t in range(len(inv)):
        c = cv2.applyColorMap((n[t] * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32)
        out[t] = np.clip(c * (0.35 + 0.8 * shading(n[t])[..., None]), 0, 255).astype(np.uint8)
    return out


def augment(inv, rob, rng, ghost=None):
    """training only, one window: make the real-robot VDA depth look like the inference composite
    - the robot's depth scaled about the window minimum by U(0.93, 1.07) (sim -> VDA fit error ~7 % of the range)
    - p .5: hard robot edges (a 3 px band outside the robot takes the background value next to it, as a sim render
      pasted on a background has no soft VDA transition)
    - p .3 with a ghost mask: the ghost-only part (where a human hand would have been) refilled from its surroundings"""
    out = inv.copy(); lo = float(np.percentile(inv, 2))
    s = rng.uniform(0.93, 1.07)
    out[rob] = lo + (out[rob] - lo) * s
    if rng.random() < 0.5:
        k = np.ones((7, 7), np.uint8)
        for t in range(len(out)):
            if not rob[t].any():
                continue
            band = (cv2.dilate(rob[t].astype(np.uint8), k) > 0) & ~rob[t]
            far = ~(band | rob[t])
            if band.any() and far.any():
                _, (iy, ix) = distance_transform_edt(~far, return_indices=True); out[t][band] = out[t][iy[band], ix[band]]
    if ghost is not None and rng.random() < 0.3:
        for t in range(len(out)):
            g = ghost[t] & ~rob[t]
            ok = ~(g | rob[t])
            if g.any() and ok.any():
                _, (iy, ix) = distance_transform_edt(~ok, return_indices=True); out[t][g] = out[t][iy[g], ix[g]]
    return out


def training_window(full_inv, od, obj, rob, rng, ghost=None):
    """full_inv, od (T,H,W) window slices; obj, rob bool -> (T,H,W,3) control B frames"""
    inv = full_inv if od is None else refine_object(full_inv, od, obj)
    if obj.any():
        inv = OD.contact_fill(inv, obj, rob)
    return encode_window(augment(inv, rob, rng, ghost))
