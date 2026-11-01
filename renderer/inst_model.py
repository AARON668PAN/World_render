"""Instance-channel adapter for RoboRender's DiT. Not part of the upstream repo.
Instances arrive as an ID map on the latent grid: every instance k gets a random unit vector e_k (re-drawn for every
training sample, so the vector is a label, not an appearance) painted with its occupancy -> (B, D_ID, 21, 30, 52).
It enters through a zero-initialised patch conv added to the input tokens, and is re-injected before every block
with a zero-initialised per-block gate. LoRA (rank r) on q/k/v/o of self/cross attention and both FFN linears.
Optional appearance canvas (M1): per-instance reference latent warped into the instance's current box, masked by its
occupancy -> (B, 16, 21, 30, 52), through its own zero-initialised patch conv."""
import math, os, sys, torch, torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, os.environ.get("ROBORENDER_ROOT", "/data/repos/roborender"))
from core.dit import WanModel, sinusoidal_embedding_1d
from core.loader import load_model, convert_dit, fuse_lora, load_state_dict
from core.pipeline import DIT_CONFIG
D_ID = 16

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__(); self.base = base; self.scale = alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False); self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5)); nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return self.base(x) + self.B(self.A(x.to(self.A.weight.dtype))).to(x.dtype) * self.scale

class InstAdapter(nn.Module):
    def __init__(self, dim=1536, num_layers=30, appearance=False, presence=False):
        super().__init__(); self.presence = presence
        self.inst_embed = nn.Conv3d(D_ID + int(presence), dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        nn.init.zeros_(self.inst_embed.weight); nn.init.zeros_(self.inst_embed.bias)
        self.gates = nn.Parameter(torch.zeros(num_layers, dim))
        self.app_embed = None
        if appearance:
            self.app_embed = nn.Conv3d(16 + 1, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            nn.init.zeros_(self.app_embed.weight); nn.init.zeros_(self.app_embed.bias)

def build(lora_rank=32, appearance=False, device="cuda", presence=False):
    dit = load_model(WanModel, "/data/models/roborender/Wan2.1-Fun-V1.1-1.3B-Control/diffusion_pytorch_model.safetensors", DIT_CONFIG, convert_dit, torch.bfloat16, device)
    fuse_lora(dit, load_state_dict("/data/models/roborender/ckpt/roborender_lora.safetensors", torch.bfloat16, device), alpha=1.0, dtype=torch.bfloat16, device=device)
    for p in dit.parameters(): p.requires_grad_(False)
    loras = []
    if lora_rank > 0:
        for blk in dit.blocks:
            for parent, names in [(blk.self_attn, "qkvo"), (blk.cross_attn, "qkvo")]:
                for n in names:
                    l = LoRALinear(getattr(parent, n), lora_rank, lora_rank).to(device); setattr(parent, n, l); loras.append(l)
            for i in (0, 2):
                l = LoRALinear(blk.ffn[i], lora_rank, lora_rank).to(device); blk.ffn[i] = l; loras.append(l)
    adapter = InstAdapter(appearance=appearance, presence=presence).to(device)
    return dit, adapter, loras

def trainable_state(adapter, loras):
    sd = {f"adapter.{k}": v for k, v in adapter.state_dict().items()}
    for i, l in enumerate(loras): sd[f"lora.{i}.A"] = l.A.weight; sd[f"lora.{i}.B"] = l.B.weight
    return sd

def load_trainable(adapter, loras, sd):
    adapter.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("adapter.")})
    for i, l in enumerate(loras): l.A.weight.data.copy_(sd[f"lora.{i}.A"]); l.B.weight.data.copy_(sd[f"lora.{i}.B"])

def id_map_from_occ(occ, vecs, presence=False):
    """occ (K, T, H, W) float, vecs (K, D_ID) -> (1, D_ID [+1 presence], T, H, W)"""
    m = torch.einsum("kthw,kd->dthw", occ, vecs)
    if presence: m = torch.cat([m, occ.sum(0, keepdim=True).clamp(max=1.0)], 0)
    return m[None]

def build_from_ckpt(path, lora_rank=32, device="cuda"):
    """Build the model that matches a checkpoint (appearance / presence detected from tensor shapes)."""
    if path in (None, "none"): return build(lora_rank, False, device) + (None,)
    sd = torch.load(path, map_location=device)
    presence = sd["adapter.inst_embed.weight"].shape[1] == D_ID + 1
    appearance = any(k.startswith("adapter.app_embed") for k in sd)
    dit, adapter, loras = build(lora_rank, appearance, device, presence)
    load_trainable(adapter, loras, sd); return dit, adapter, loras, sd

def forward(dit, adapter, latents, timestep, context, clip_feature, y, ref=None, id_map=None, app=None, grad_ckpt=True):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    ctx = torch.cat([dit.img_emb(clip_feature), dit.text_embedding(context)], dim=1)
    x = dit.patch_embedding(torch.cat([latents, y], dim=1))
    f, h, w = x.shape[2:]
    inst = adapter.inst_embed(id_map.to(x.dtype)) if id_map is not None else torch.zeros_like(x)
    if app is not None and adapter.app_embed is not None: inst = inst + adapter.app_embed(app.to(x.dtype))
    x = rearrange(x + inst, "b c f h w -> b (f h w) c"); inst_tok = rearrange(inst, "b c f h w -> b (f h w) c")
    nref = 0
    if ref is not None:
        r = dit.ref_conv(ref[:, :, 0]).flatten(2).transpose(1, 2); nref = r.shape[1]
        x = torch.cat([r, x], 1); inst_tok = torch.cat([torch.zeros_like(r), inst_tok], 1); f += 1
    freqs = torch.cat([dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                       dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                       dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    gates = adapter.gates.to(x.dtype)
    for i, blk in enumerate(dit.blocks):
        x = x + gates[i] * inst_tok
        x = checkpoint(blk, x, ctx, t_mod, freqs, use_reentrant=False) if grad_ckpt else blk(x, ctx, t_mod, freqs)
    x = dit.head(x, t)
    if nref: x = x[:, nref:]; f -= 1
    return dit.unpatchify(x, (f, h, w))

def appearance_canvas(occ, z0):
    """Per instance: crop its cells from latent frame 0 of z0 (training: the previous real frame), resize the crop into the
    instance's box in every latent frame and keep only occupied cells. Returns (1, 17, T, H, W) = canvas + visibility."""
    K, T, H, W = occ.shape; can = torch.zeros(16, T, H, W, device=occ.device); vis = torch.zeros(1, T, H, W, device=occ.device)
    for k in range(K):
        m0 = occ[k, 0] > 0.5
        if m0.sum() < 2: continue
        ys, xs = torch.where(m0); src = z0[:, 0, ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # z0 here is the previous real frame
        for t in range(T):
            mt = occ[k, t] > 0.5
            if mt.sum() < 1: continue
            ys2, xs2 = torch.where(mt); hh, ww = int(ys2.max() - ys2.min() + 1), int(xs2.max() - xs2.min() + 1)
            patch = torch.nn.functional.interpolate(src[None].float(), size=(hh, ww), mode="bilinear", align_corners=False)[0]
            region = mt[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]
            can[:, t, ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1][:, region] = patch[:, region]
            vis[0, t] = torch.maximum(vis[0, t], occ[k, t])
    return torch.cat([can, vis], 0)[None]

import os
MODEL_DIR = os.environ.get("WAN_MODEL_DIR", "/data/models/roborender/Wan2.1-Fun-V1.1-1.3B-Control")
PROMPT_CACHE = os.environ.get("PROMPT_CACHE", "/data/models/roborender/prompt_cache.pt")

def lite_pipe(device="cuda"):
    """RoboRenderPipeline with only the VAE, CLIP image encoder and scheduler (no T5, no DiT); prompts come from PROMPT_CACHE."""
    from core.pipeline import RoboRenderPipeline, _find
    from core.vae import WanVideoVAE
    from core.clip import WanImageEncoder
    from core.loader import convert_vae, convert_clip
    from core.scheduler import FlowMatchScheduler
    p = object.__new__(RoboRenderPipeline); p.device = device; p.dtype = torch.bfloat16; p.scheduler = FlowMatchScheduler()
    p.vae = load_model(WanVideoVAE, _find(MODEL_DIR, "Wan2.1_VAE.safetensors", "Wan2.1_VAE.pth"), None, convert_vae, p.dtype, device)
    p.image_encoder = load_model(WanImageEncoder, _find(MODEL_DIR, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"), None, convert_clip, p.dtype, device)
    cache = torch.load(PROMPT_CACHE, map_location=device)
    p.encode_prompt = lambda prompt: cache[prompt]
    return p
