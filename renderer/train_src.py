"""Source-video renderer on Unitree G1 + Inspire (caches from cache_g1i.py; first written for the G1 Dex3 pilot, cache_src.py,
not shipped). Not part of the upstream repo. Like iso/train_mem.py of the working copy (not shipped) without information
frames, plus a per-frame source video: the input layer takes
    noisy latent 16 | control 16 | zeros 16 (base model's first-frame slot) | source 16 | hole mask 1  = 65 channels
(the 17 new input channels start at zero, so step 0 is the base model). Conditions: control = grey + sim robot (CG) +
object in a random colour; source = the same scene at another moment with grey holes; reference = real frame before
the window (ref_conv + CLIP), dropped with prob. --ref_drop (the first window at inference has none) and noised with
prob. --ref_noise. Fixed empty prompt. Flow matching as Wan (shift 5); loss up-weighted on object cells.
LoRA (rank 128, scale 1) on attention q,k,v,o and both FFN layers + trainable input layer and ref_conv.
v5 (10-31): a cache made with cache_g1i.py --info K carries K scene (info) frames per window; they go through ref_conv,
get a learned memory embedding (as iso/train_mem.py) and are put before the reference frame; dropped together with
prob. --info_drop.
v6 (10-31): a cache made with cache_g1i.py --depth carries control B (whole-frame depth latent "yd"); it goes into the
base model's first-frame slot (channels 32-47 of the input layer, zeros until v5; RoboRender also reuses this slot), whose
weights are zeroed when a run starts from another run's weights (--init_from), so step 0 is exactly that model; dropped
with prob. --depth_drop. --init_from: LoRA / input layer / ref_conv / mem_embed from a resume.pt or fetch_steps lora file.
Usage: torchrun --nproc_per_node N train_src.py RUN_DIR STEPS --cache DIR --val val_segments.json [flags]
(the flags of every real run: cluster/train_v5.sbatch, train_v6a.sbatch, train_v6b.sbatch: --balance --batch 8 --accum 1
--ckpt_grad --lora_rank 128 --save_every 250 on 2 GPUs)"""
import argparse
import glob
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist
from einops import rearrange
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ROBORENDER_ROOT", "/data/repos/roborender")  # upstream RoboRender clone (core/)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)  # inst_model + sibling modules live next to this script
from core.dit import WanModel, sinusoidal_embedding_1d  # noqa: E402
from core.loader import load_model, convert_dit  # noqa: E402
from core.pipeline import DIT_CONFIG  # noqa: E402
import inst_model as IM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("steps", type=int)
ap.add_argument("--cache", required=True)
ap.add_argument("--val", default="")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr_patch", type=float, default=2e-5)
ap.add_argument("--lora_rank", type=int, default=128)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--ref_drop", type=float, default=0.2)
ap.add_argument("--ref_noise", type=float, default=0.5)
ap.add_argument("--info_drop", type=float, default=0.2, help="v5: probability of training a batch without its info frames")
ap.add_argument("--depth_drop", type=float, default=0.2, help="v6b: probability of a window without its depth (control B)")
ap.add_argument("--init_from", default="", help="v6: start from another run's trained weights (resume.pt or lora_NNNNN.pt), fresh optimizer, step 0")
ap.add_argument("--ref_noise_max", type=float, default=0.5)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--save_every", type=int, default=1000)
ap.add_argument("--ckpt_grad", action="store_true")
ap.add_argument("--ckpt_blocks", type=int, default=30)
ap.add_argument("--opt8bit", action="store_true")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--balance", action="store_true", help="pick a dataset uniformly, then a window of it (cache_v2 clips are <Dataset>_epNNN)")
a = ap.parse_args()
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local)
dev = f"cuda:{local}"
random.seed(a.seed + rank)
torch.manual_seed(a.seed + rank)
SHIFT, AREF = 5.0, 40.0
if rank == 0:
    os.makedirs(a.run, exist_ok=True)
    json.dump(vars(a), open(f"{a.run}/args.json", "w"), indent=1)
val = set(json.load(open(a.val))) if a.val else set()
items, by_ds = [], {}
for f in sorted(glob.glob(f"{a.cache}/*.pt")):
    d = torch.load(f, map_location="cpu", mmap=True)
    if d.get("clip") in val or str(d.get("episode")) in val:  # val list: clip names (Unitree) or episode ids (AgiBot)
        continue
    new = [(f, k) for k in range(len(d["windows"]))]
    items += new
    by_ds.setdefault(d.get("dataset", "all"), []).extend(new)
if rank == 0:
    print(f"{len(items)} training windows, world {world}; per dataset: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(by_ds.items())), flush=True)
EMPTY = torch.load(IM.PROMPT_CACHE, map_location=dev)[""].to(torch.bfloat16)

dit = load_model(WanModel, f"{IM.MODEL_DIR}/diffusion_pytorch_model.safetensors", DIT_CONFIG, convert_dit, torch.bfloat16, dev)
for p in dit.parameters():
    p.requires_grad_(False)
old = dit.patch_embedding  # Conv3d(48, dim): noise 16 | control 16 | first-frame 16
new = torch.nn.Conv3d(old.in_channels + 17, old.out_channels, kernel_size=old.kernel_size, stride=old.stride).to(dev)
with torch.no_grad():
    new.weight.zero_()
    new.weight[:, :old.in_channels] = old.weight.float()
    new.bias.copy_(old.bias.float())
dit.patch_embedding = new
loras = []
for blk in dit.blocks:
    for parent in (blk.self_attn, blk.cross_attn):
        for n in "qkvo":
            lyr = IM.LoRALinear(getattr(parent, n), a.lora_rank, a.lora_rank).to(dev)
            setattr(parent, n, lyr)
            loras.append(lyr)
    for i in (0, 2):
        lyr = IM.LoRALinear(blk.ffn[i], a.lora_rank, a.lora_rank).to(dev)
        blk.ffn[i] = lyr
        loras.append(lyr)
for lyr in loras:
    for p in (lyr.A.weight, lyr.B.weight):
        p.data = p.data.float()
        p.requires_grad_(True)
for p in list(dit.patch_embedding.parameters()) + list(dit.ref_conv.parameters()):
    p.data = p.data.float()
    p.requires_grad_(True)
w0 = {}
for f_, k_ in items[:50]:  # a window that has info frames (a cache_g1i.py --info cache; a few windows may lack them)
    w0 = torch.load(f_, map_location="cpu", mmap=True)["windows"][k_]
    if w0.get("info") is not None:
        break
HAS_INFO = w0.get("info") is not None
HAS_DEPTH = any(torch.load(f_, map_location="cpu", mmap=True)["windows"][k_].get("yd") is not None for f_, k_ in items[:50])
if HAS_INFO:  # marks info-frame tokens as scene memory, not the previous frame
    dit.mem_embed = torch.nn.Parameter(torch.zeros(1, 1, dit.dim, device=dev))
if rank == 0:
    print(f"info frames: {tuple(w0['info'].shape) if HAS_INFO else 'none'}", flush=True)
start = 0
if a.init_from and not (a.resume and os.path.exists(f"{a.run}/resume.pt")):
    src_sd = torch.load(a.init_from, map_location="cpu")
    src_sd = src_sd.get("model", src_sd.get("keep"))
    missing = dit.load_state_dict({k: v.float() for k, v in src_sd.items()}, strict=False)
    if rank == 0:
        print(f"init from {a.init_from}: {len(src_sd)} tensors, unexpected {len(missing.unexpected_keys)}", flush=True)
    if HAS_DEPTH:
        with torch.no_grad():
            dit.patch_embedding.weight[:, 32:48] = 0  # first-frame slot -> control B, starts silent (step 0 = the source run)
if a.resume and os.path.exists(f"{a.run}/resume.pt"):
    rs = torch.load(f"{a.run}/resume.pt", map_location="cpu")
    dit.load_state_dict(rs["model"], strict=False)
    start = rs["step"]
TRAIN = [p for p in dit.parameters() if p.requires_grad]
if rank == 0:
    print("trainable params %.1fM" % (sum(p.numel() for p in TRAIN) / 1e6), flush=True)
groups = [dict(params=[p for lyr in loras for p in (lyr.A.weight, lyr.B.weight)], lr=a.lr, base_lr=a.lr),
          dict(params=list(dit.patch_embedding.parameters()) + list(dit.ref_conv.parameters()) + ([dit.mem_embed] if HAS_INFO else []), lr=a.lr_patch, base_lr=a.lr_patch)]
if a.opt8bit:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.999), weight_decay=0.0)
else:
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
if start:
    opt.load_state_dict(rs["opt"])
    del rs


def fwd(m, latents, timestep, context, clip_feature, y, ref, info=None):
    t = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, timestep))
    t_mod = m.time_projection(t).unflatten(1, (6, m.dim))
    ctx = torch.cat([m.img_emb(clip_feature), m.text_embedding(context)], dim=1)
    x = m.patch_embedding(torch.cat([latents, y], dim=1))
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c")
    pre = []
    if info is not None:  # (B,K,16,h,w): info frames first, then the reference frame, then the video
        B = info.shape[0]
        r = m.ref_conv(info.flatten(0, 1)).flatten(2).transpose(1, 2)
        pre.append(r.reshape(B, -1, r.shape[-1]) + m.mem_embed)
    if ref is not None:
        pre.append(m.ref_conv(ref[:, :, 0]).flatten(2).transpose(1, 2))
    nref = sum(p.shape[1] for p in pre)
    nf = nref // (h * w)
    if pre:
        x = torch.cat(pre + [x], 1)
        f += nf
    freqs = torch.cat([m.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), m.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                       m.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    for i, blk in enumerate(m.blocks):
        x = checkpoint(blk, x, ctx, t_mod, freqs, use_reentrant=False) if (a.ckpt_grad and i < a.ckpt_blocks) else blk(x, ctx, t_mod, freqs)
    x = m.head(x, t.unsqueeze(1))
    if nref:
        x = x[:, nref:]
        f -= nf
    return m.unpatchify(x, (f, h, w))


class Net(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, *args):
        return fwd(self.dit, *args)


dit.train()
model = torch.nn.parallel.DistributedDataParallel(Net(dit), device_ids=[local], find_unused_parameters=True)


def sample_window():
    f, k = random.choice(by_ds[random.choice(list(by_ds))]) if a.balance else random.choice(items)
    w = torch.load(f, map_location="cpu", mmap=True)["windows"][k]
    s = random.choice(w["srcs"])
    slot = w["yd"] if (HAS_DEPTH and w.get("yd") is not None and random.random() >= a.depth_drop) else torch.zeros_like(w["y"])
    y = torch.cat([w["y"], slot, s["lat"], s["hole"]], 0)
    return dict(z0=w["z0"], y=y, ref=w["ref"], clip=w["clip"], occ=w["occ"], info=w.get("info"))


def merged_state():
    sd = {}
    for k, v in dit.state_dict().items():
        if k.endswith((".A.weight", ".B.weight")):
            continue
        sd[k.replace(".base.", ".")] = v.detach().float()
    for name, mod in dit.named_modules():
        if isinstance(mod, IM.LoRALinear):
            sd[f"{name}.weight"] = sd[f"{name}.weight"] + (mod.B.weight.float() @ mod.A.weight.float()) * mod.scale
    return {k: v.to(torch.bfloat16).cpu() for k, v in sd.items()}


def save(step):
    if rank != 0:
        return
    torch.save(dict(mode="lora_merged", layout="g1src", in_dim=dit.patch_embedding.in_channels, state=merged_state(), step=step), f"{a.run}/ckpt_{step:05d}.pt")
    keep = {k: v for k, v in dit.state_dict().items() if k.endswith((".A.weight", ".B.weight")) or k.startswith(("patch_embedding.", "ref_conv.", "mem_embed"))}
    torch.save(dict(model=keep, opt=opt.state_dict(), step=step), f"{a.run}/resume.pt.tmp")
    os.replace(f"{a.run}/resume.pt.tmp", f"{a.run}/resume.pt")


log = open(f"{a.run}/log.jsonl", "a") if rank == 0 else None
t0 = time.time()
run_loss = []
for step in range(start + 1, a.steps + 1):
    lr = a.lr * min(1.0, step / a.warmup)
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * min(1.0, step / a.warmup)
    for micro in range(a.accum):
        ws = [sample_window() for _ in range(a.batch)]
        z0 = torch.stack([w["z0"] for w in ws]).to(dev, torch.float32)
        y = torch.stack([w["y"] for w in ws]).to(dev, torch.bfloat16)
        use_ref = random.random() >= a.ref_drop
        ref = torch.stack([w["ref"] for w in ws]).to(dev, torch.float32) if use_ref else None
        if ref is not None and random.random() < a.ref_noise:
            ref = ref + random.uniform(0, a.ref_noise_max) * torch.randn_like(ref)
        ref = ref.to(torch.bfloat16) if ref is not None else None
        clip = torch.stack([w["clip"] for w in ws]).to(dev, torch.bfloat16) if use_ref else torch.zeros(a.batch, 257, 1280, device=dev, dtype=torch.bfloat16)
        use_info = HAS_INFO and all(w["info"] is not None and len(w["info"]) == len(w0["info"]) for w in ws) and random.random() >= a.info_drop
        info = torch.stack([w["info"] for w in ws]).to(dev, torch.bfloat16) if use_info else None
        u = torch.rand(a.batch, device=dev)
        sigma = (SHIFT * u / (1 + (SHIFT - 1) * u)).view(-1, 1, 1, 1, 1)
        eps = torch.randn_like(z0)
        xt = (1 - sigma) * z0 + sigma * eps
        target = eps - z0
        ctx = model.no_sync() if micro < a.accum - 1 else torch.enable_grad()
        with ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(xt.to(torch.bfloat16), sigma.view(-1) * 1000, EMPTY.expand(a.batch, -1, -1), clip, y, ref, info)
            err = ((pred.float() - target) ** 2).mean(1)
            wgt = torch.ones_like(err)
            for b, w in enumerate(ws):
                occ = w["occ"].to(dev).float()
                if a.lam > 0 and occ.sum() > 0:
                    area = occ.sum((2, 3)).mean(1).clamp(min=1.0)
                    wgt[b] = wgt[b] + a.lam * (occ * (AREF / area).sqrt().clamp(max=5.0)[:, None, None, None]).sum(0)
            loss = (err * wgt).sum() / wgt.sum() / a.accum
            loss.backward()
            run_loss.append(loss.item() * a.accum)
    torch.nn.utils.clip_grad_norm_(TRAIN, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if rank == 0 and step % 25 == 0:
        r = dict(step=step, loss=sum(run_loss) / len(run_loss), lr=lr, s_per_step=(time.time() - t0) / (step - start), mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        print(json.dumps(r), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()
    if rank == 0:
        run_loss = run_loss[-200:]
    if step % a.save_every == 0 or step == a.steps:
        save(step)
if rank == 0:
    print("TRAIN_DONE", flush=True)
dist.destroy_process_group()
