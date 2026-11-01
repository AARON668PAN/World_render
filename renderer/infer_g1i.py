"""Roll a train_src.py checkpoint (G1 + Inspire cache, cache_g1i.py, 448x336, 15 fps) over a sequence, chunk by chunk
(49 frames; the first chunk has no reference, later chunks take the last generated frame). Not part of the upstream repo.
  --val ID      held-out WBT segment (ROOT/segments/<ID>.mp4 + ROOT/robot/<ID>.npz), compared with the real frames;
                control = the handled object in one colour + the real robot cut out on top (as in training), source = the segment shifted by --shift frames with
                holes at the robot (source and target time, dilated). Starts at the segment's window start.
  --human NPZ   human clip prepared by prep_human_g1i.py (letterboxed 16:9 video, sim G1 + Inspire control, holes).
Writes OUT/<name>.mp4 (control | source | output [| real]) and OUT/results_<name>.json.
  --info K      v5 checkpoints: K scene (info) frames per chunk from outside it (val: the segment with the robot holed;
                human: the clip with the hands holed, its first frame always in - before the hands touch anything),
                picked greedily to show what the chunk's source hides.
  --depth       v6b checkpoints: control B (whole-frame depth) in the first-frame slot (val: from ROOT/full_depth; human: the
                npz's ctrlB from prep_human_depth.py). --mask_pool mean / --hole_px 1 for v6 checkpoints.
Usage: infer_g1i.py CKPT OUT (--val ID [--root DIR] | --human NPZ) [--steps 30] [--shift 15] [--max_chunks 3]"""
import argparse, json, os, sys
import cv2, imageio.v2 as imageio, numpy as np, torch
from einops import rearrange
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get("ROBORENDER_ROOT", "/data/repos/roborender")  # upstream RoboRender (core/)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)  # inst_model + sibling modules next to this script
from core.dit import WanModel, sinusoidal_embedding_1d  # noqa: E402
from core.loader import load_model, convert_dit  # noqa: E402
from core.pipeline import DIT_CONFIG  # noqa: E402
from core.video import tensor_to_video, video_to_tensor, image_to_tensor  # noqa: E402
import inst_model as IM  # noqa: E402
import objdepth_enc as OD  # noqa: E402
import fulldepth_enc as FE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--val", default=""); ap.add_argument("--root", default="/data/datasets/unitree_g1_inspire"); ap.add_argument("--human", default="")
ap.add_argument("--steps", type=int, default=30); ap.add_argument("--shift", type=int, default=15); ap.add_argument("--max_chunks", type=int, default=3)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--info", type=int, default=0, help="number of info frames per chunk (v5 checkpoints; 0 = none)")
ap.add_argument("--depth", action="store_true", help="v6b: control B in the first-frame slot")
ap.add_argument("--mask_pool", default="max", choices=["max", "mean"], help="hole channel per 8x8 cell (v6: mean)")
ap.add_argument("--hole_px", type=int, default=5, help="val: robot hole margin in px (v6: 1)")
ap.add_argument("--no_source", action="store_true", help="val ablation: source fully grey (all hole)")
ap.add_argument("--real_ref", action="store_true", help="val: the real frame before the window as the first chunk's reference (as iso/eval_mem.py did)")
ap.add_argument("--obj_enc", default="flat", choices=["flat", "c1"], help="val control: object as a flat colour or C1 depth (needs ROOT/obj_depth)")
ap.add_argument("--obj_dir", default="obj1", help="object maps under --root for the val control ('' = none, for the no-object model)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
W, H, WIN, GRAY = 448, 336, 49, 128
OBJ_COL = np.array([235, 60, 200], np.uint8)
K_ROB = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

ck = torch.load(a.ckpt, map_location="cpu")
dit = load_model(WanModel, f"{IM.MODEL_DIR}/diffusion_pytorch_model.safetensors", DIT_CONFIG, convert_dit, torch.bfloat16, dev)
old = dit.patch_embedding
dit.patch_embedding = torch.nn.Conv3d(ck["in_dim"], old.out_channels, kernel_size=old.kernel_size, stride=old.stride).to(dev, torch.bfloat16)
if "mem_embed" in ck["state"]:  # v5: info-frame memory embedding
    dit.mem_embed = torch.nn.Parameter(torch.zeros(1, 1, dit.dim, device=dev, dtype=torch.bfloat16))
assert not a.info or "mem_embed" in ck["state"], "--info needs a checkpoint trained with info frames"
dit.load_state_dict({k: v.to(dev) for k, v in ck["state"].items()}, strict=True); dit.eval(); del ck
pipe = IM.lite_pipe(dev); EMPTY = pipe.encode_prompt("").to(torch.bfloat16)


def fwd(m, latents, timestep, context, clip_feature, y, ref, info=None):  # same as train_src.py
    t = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, timestep)); t_mod = m.time_projection(t).unflatten(1, (6, m.dim))
    ctx = torch.cat([m.img_emb(clip_feature), m.text_embedding(context)], dim=1)
    x = m.patch_embedding(torch.cat([latents, y], dim=1)); f, h, w = x.shape[2:]; x = rearrange(x, "b c f h w -> b (f h w) c"); pre = []
    if info is not None:  # info frames first, then the reference frame, then the video
        r = m.ref_conv(info.flatten(0, 1)).flatten(2).transpose(1, 2); pre.append(r.reshape(info.shape[0], -1, r.shape[-1]) + m.mem_embed)
    if ref is not None:
        pre.append(m.ref_conv(ref[:, :, 0]).flatten(2).transpose(1, 2))
    nref = sum(p.shape[1] for p in pre); nf = nref // (h * w)
    if pre: x = torch.cat(pre + [x], 1); f += nf
    freqs = torch.cat([m.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), m.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                       m.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    for blk in m.blocks: x = blk(x, ctx, t_mod, freqs)
    x = m.head(x, t.unsqueeze(1))
    if nref: x = x[:, nref:]; f -= nf
    return m.unpatchify(x, (f, h, w))


def enc(frames):
    with torch.no_grad(): return pipe.encode_video([Image.fromarray(f) for f in frames])[0][None].to(dev, torch.bfloat16)


def cells_max(m):
    x = m.astype(np.float32).reshape(len(m), H // 8, 8, W // 8, 8)
    if a.mask_pool == "mean":  # share of hole pixels per cell (v6), as cache_g1i.py cells(.., "mean")
        x = x.mean((2, 4)); return np.concatenate([x[:1], x[1:].reshape((len(m) - 1) // 4, 4, H // 8, W // 8).mean(1)], 0)
    x = x.max((2, 4))
    return np.concatenate([x[:1], x[1:].reshape((len(m) - 1) // 4, 4, H // 8, W // 8).max(1)], 0)


def pick_info(scene_hole, lo, hi, need, first):
    """a.info frames of the scene video outside [lo, hi) (>= 6 frames off), greedy on what the chunk's source hides
    (as cache_g1i.py pick_info); first: frame 0 always in"""
    cell = lambda m: m.reshape(-1, H // 8, 8, W // 8, 8).mean((2, 4))
    need = cell(need).mean(0); cand = [t for t in range(len(scene_hole)) if t < lo - 7 or t >= hi + 6] or list(range(len(scene_hole)))
    vis = {t: 1 - cell(scene_hole[t:t + 1])[0] for t in set(cand) | {0}}
    pick = [0] if first else []
    for t in pick: need = need * (1 - vis[t])
    while len(pick) < a.info:
        t = max([t for t in cand if t not in pick] or cand, key=lambda t: (need * vis[t]).sum()); pick.append(t); need = need * (1 - vis[t])
    return sorted(pick)


def generate(ctrl, src, hole, ref0=None, scene=None, scene_hole=None, off=0, first=False, ctrlb=None):
    """scene / scene_hole: the clip for the info frames (--info), ctrl frame 0 = scene frame off"""
    n = min(a.max_chunks, len(ctrl) // WIN); out, ref_img = [], ref0  # ref0: a real frame for the first chunk (--real_ref)
    for c in range(n):
        seg = slice(c * WIN, c * WIN + WIN); info = None
        if a.info:
            it = pick_info(scene_hole, off + c * WIN, off + c * WIN + WIN, hole[seg], first); INFO_T.append(it)
            imgs = [np.where(scene_hole[t][..., None], GRAY, scene[t]).astype(np.uint8) for t in it]
            imageio.imwrite(f"{a.out}/info_chunk{c}.jpg", np.hstack(imgs))
            with torch.no_grad():
                info = torch.stack([pipe.vae.encode(video_to_tensor([Image.fromarray(im)], pipe.dtype, pipe.device), device=pipe.device)[0][:, 0] for im in imgs])[None].to(torch.bfloat16)
        slot = enc(ctrlb[seg]) if (a.depth and ctrlb is not None) else torch.zeros(1, 16, 13, H // 8, W // 8, device=dev, dtype=torch.bfloat16)
        y = torch.cat([enc(ctrl[seg]), slot, enc(src[seg]),
                       torch.from_numpy(cells_max(hole[seg]))[None, None].to(dev, torch.bfloat16)], 1)
        if ref_img is None: ref, clip = None, torch.zeros(1, 257, 1280, device=dev, dtype=torch.bfloat16)
        else:
            with torch.no_grad():
                ref = pipe.vae.encode(video_to_tensor([Image.fromarray(ref_img)], pipe.dtype, pipe.device), device=pipe.device).to(torch.bfloat16)
                clip = pipe.image_encoder.encode_image([image_to_tensor(Image.fromarray(ref_img), pipe.dtype, pipe.device)]).to(torch.bfloat16)
        pipe.scheduler.set_timesteps(a.steps, shift=5.0)
        lat = torch.randn((1, 16, 13, H // 8, W // 8), generator=torch.Generator("cpu").manual_seed(a.seed + c)).to(dev, torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in pipe.scheduler.timesteps:
                lat = pipe.scheduler.step(fwd(dit, lat, t.unsqueeze(0).to(dev), EMPTY, clip, y, ref, info).float(), t, lat.float()).to(torch.bfloat16)
            frames = np.stack([np.array(f) for f in tensor_to_video(pipe.vae.decode(lat, device=dev))])
        out.append(frames); ref_img = frames[-1]; print(f"chunk {c + 1}/{n}", flush=True)
    return np.concatenate(out)


def psnr(x, y, m=None):
    d = (x.astype(np.float32) - y.astype(np.float32)) ** 2; mse = d[m].mean() if m is not None else d.mean()
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-6)))


def write(name, cols):
    w = imageio.get_writer(f"{a.out}/{name}.mp4", fps=15, codec="libx264", quality=8, macro_block_size=8)
    for k in range(len(cols[-1])): w.append_data(np.hstack([c[k] for c in cols]))
    w.close()


INFO_T = []
dil = lambda m: np.stack([cv2.dilate(x.astype(np.uint8), K_ROB) > 0 for x in m])
if a.val:
    seg = [s for s in json.load(open(f"{a.root}/segments.json")) if s["id"] == a.val][0]
    rgb = np.stack([f for f in imageio.get_reader(f"{a.root}/segments/{a.val}.mp4")]); T = len(rgb)
    z = np.load(f"{a.root}/robot/{a.val}.npz"); rob = np.unpackbits(z["mask"], axis=-1)[..., :W].astype(bool)[:T]
    s0 = seg["win_start"]; sh = a.shift if s0 + a.shift + WIN <= T else -a.shift  # source later if it fits, else earlier
    Tn = min(T - s0, T - (s0 + sh)) if sh > 0 else T - s0
    Tn = (Tn // WIN) * WIN; tg = slice(s0, s0 + Tn); sr = slice(s0 + sh, s0 + sh + Tn)
    ctrl = np.full(rgb[tg].shape, GRAY, np.uint8)
    op = f"{a.root}/{a.obj_dir}/{a.val}.npz"  # the handled object (Claude box + SAM3) in one colour, as in training
    if a.obj_dir and os.path.exists(op):
        obj = np.load(op)["lab"][:T] > 0; ctrl[obj[tg] & ~rob[tg]] = OBJ_COL
        dp = f"{a.root}/obj_depth/{a.val}.npz"
        if a.obj_enc == "c1" and os.path.exists(dp):  # same C1 encoding as a --obj_enc c1 cache, per 49-frame chunk
            inv = np.load(dp)["inv"][:T].astype(np.float32)[tg]; ob = obj[tg] & ~rob[tg]
            for c0 in range(0, len(ctrl), WIN):
                sl = slice(c0, c0 + WIN); ctrl[sl][ob[sl]] = GRAY; OD.paint(ctrl[sl], inv[sl], ob[sl], rob[tg][sl])
    ctrl = np.where(rob[tg][..., None], rgb[tg], ctrl).astype(np.uint8)
    KH = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.hole_px + 1, 2 * a.hole_px + 1))
    dh = lambda m: np.stack([cv2.dilate(x.astype(np.uint8), KH) > 0 for x in m])
    hole = dh(rob[sr]) | dh(rob[tg]); src = np.where(hole[..., None], GRAY, rgb[sr]).astype(np.uint8)
    ctrlb = None
    if a.depth:  # control B as in training, without the training-only augmentations, per 49-frame chunk
        finv = FE.load_full(f"{a.root}/full_depth/{a.val}.npz", T)[tg]; dp = f"{a.root}/obj_depth/{a.val}.npz"
        od = np.load(dp)["inv"][:T].astype(np.float32)[tg] if os.path.exists(dp) else None
        ob = (np.load(op)["lab"][:T] > 0)[tg] & ~rob[tg] if (a.obj_dir and os.path.exists(op)) else np.zeros_like(rob[tg])
        ctrlb = np.zeros(ctrl.shape, np.uint8)
        for c0 in range(0, len(ctrl) - WIN + 1, WIN):
            sl = slice(c0, c0 + WIN); inv = finv[sl] if od is None else FE.refine_object(finv[sl], od[sl], ob[sl])
            if ob[sl].any():
                inv = OD.contact_fill(inv, ob[sl], rob[tg][sl])
            ctrlb[sl] = FE.encode_window(inv)
    if a.no_source:  # ablation: no scene information at all (all hole)
        hole[:] = True; src[:] = GRAY
    gen = generate(ctrl, src, hole, rgb[s0 - 1] if a.real_ref and s0 >= 1 else None, rgb, dh(rob), s0, ctrlb=ctrlb); n = len(gen); real = rgb[tg][:n]; rm = dil(rob[tg][:n])
    res = dict(segment=a.val, ckpt=a.ckpt, shift=sh, real_ref=a.real_ref, no_source=a.no_source, frames=n, psnr=psnr(gen, real), psnr_robot=psnr(gen, real, rm), psnr_background=psnr(gen, real, ~rm))
    try:
        import lpips
        LP = lpips.LPIPS(net="alex").to(dev).eval(); tt = lambda im: torch.from_numpy(im).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
        with torch.no_grad(): res["lpips"] = float(np.mean([float(LP(tt(gen[k]), tt(real[k]))) for k in range(0, n, 4)]))
    except ImportError:
        pass
    name = f"val_{a.val}"; write(name, [ctrl[:n], src[:n], gen, real])
else:
    z = np.load(a.human); ctrl, src, hole = z["ctrl"], z["src"], z["hole"]
    assert not a.info or "scene" in z.files, "--info needs a human npz from prep_human_g1i.py with scene / scene_hole"
    assert not a.depth or "ctrlB" in z.files, "--depth needs ctrlB in the human npz (prep_human_depth.py)"
    gen = generate(ctrl, src, hole, None, z["scene"] if a.info else None, z["scene_hole"] if a.info else None, 0, True, z["ctrlB"] if a.depth else None); n = len(gen)
    name = "human_" + os.path.basename(a.human).replace(".npz", ""); res = dict(human=a.human, ckpt=a.ckpt, frames=n); write(name, [ctrl[:n], src[:n], gen])
res["info_frames"] = INFO_T
print(json.dumps(res)); json.dump(res, open(f"{a.out}/results_{name}.json", "w"), indent=1)
