"""Policy "dit_resnet": multi_task_dit with the EgoEngine-style ResNet-18 vision stem (see configuration_dit_resnet.py)."""
import math

import torch
import torch.nn as nn
import torchvision
from torch import Tensor

from lerobot.policies.multi_task_dit import modeling_multi_task_dit as mtd
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy

from .configuration_dit_resnet import DiTResNetConfig


def _sincos_2d(h: int, w: int, dim: int, device) -> Tensor:
    """Fixed 2D sine-cosine position embedding [h * w, dim]."""
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, device=device, dtype=torch.float32) / d))
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    oy, ox = y.flatten()[:, None] * omega, x.flatten()[:, None] * omega
    return torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], 1)


class ResNet18TokenEncoder(nn.Module):
    """ImageNet normalisation -> ResNet-18 up to layer4 (before global average pooling) -> 1x1 projection -> cross-attention
    of `num_tokens` learnable queries over the spatial feature map (HPT-style stem) -> flattened tokens."""

    def __init__(self, num_tokens: int = 16, dim: int = 256, freeze: bool = False):
        super().__init__()
        rn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(rn.children())[:-2])
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.proj = nn.Conv2d(512, dim, 1)
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.num_tokens, self.dim = num_tokens, dim

    def forward(self, x: Tensor) -> Tensor:
        f = self.proj(self.backbone((x - self.mean) / self.std))  # [N, dim, h, w]
        n, c, h, w = f.shape
        tok = f.flatten(2).transpose(1, 2) + _sincos_2d(h, w, c, f.device)[None]
        q = self.queries.unsqueeze(0).expand(n, -1, -1)
        out, _ = self.attn(q, tok, tok)
        return self.norm(out + q).reshape(n, self.num_tokens * self.dim, 1, 1)

    def get_output_shape(self) -> tuple:
        return (self.num_tokens * self.dim, 1, 1)


class ResNetObservationEncoder(mtd.ObservationEncoder):
    """multi_task_dit's ObservationEncoder with the CLIP vision encoder replaced by ResNet18TokenEncoder."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self._setup_preprocessing(config)
        if config.use_separate_rgb_encoder_per_camera:
            raise NotImplementedError("dit_resnet shares one encoder across cameras")
        self.num_cameras = len(config.image_features); self.camera_names = list(config.image_features.keys())
        self.vision_encoder = ResNet18TokenEncoder(config.vision_num_tokens, config.vision_token_dim, config.freeze_backbone)
        self.vision_encoders = None
        self.robot_state_dim = config.robot_state_feature.shape[0] if getattr(config, "robot_state_feature", None) else 0
        self.text_dim = config.hidden_dim
        self.text_encoder = mtd.CLIPTextEncoder(model_name=config.text_encoder_name, projection_dim=self.text_dim)
        self._setup_vector_output()

    def encode(self, batch: dict) -> Tensor:
        p = getattr(self.config, "state_dropout", 0.0)
        if self.training and p > 0:  # state dropout (DiTResNetConfig.state_dropout); off at inference (policy.eval())
            s = batch["observation.state"]
            keep = (torch.rand(s.shape[0], *([1] * (s.dim() - 1)), device=s.device) >= p).to(s.dtype)
            batch = {**batch, "observation.state": s * keep}
        return super().encode(batch)


class DiTResNetPolicy(MultiTaskDiTPolicy):
    config_class = DiTResNetConfig
    name = "dit_resnet"

    def __init__(self, config: DiTResNetConfig, **kwargs):
        orig = mtd.ObservationEncoder
        mtd.ObservationEncoder = ResNetObservationEncoder  # MultiTaskDiTPolicy.__init__ builds its encoder by this name
        try:
            super().__init__(config, **kwargs)
        finally:
            mtd.ObservationEncoder = orig

    def get_optim_params(self) -> list:
        """The ResNet backbone at vision_encoder_lr_multiplier x lr (as multi_task_dit's CLIP); the stem and the rest at lr."""
        back, rest = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                (back if "observation_encoder.vision_encoder.backbone" in name else rest).append(p)
        return [{"params": rest}, {"params": back, "lr": self.config.optimizer_lr * self.config.vision_encoder_lr_multiplier}]


def rtc_prefix_weights(start: int, end: int, total: int) -> Tensor:
    """LeRobot RTCProcessor.get_prefix_weights, LINEAR schedule: 1 on [0, start), linear 1 -> 0 inside [start, end), 0 after."""
    w = torch.zeros(total)
    w[:start] = 1.0
    n = end - start
    if n > 0:
        w[start:end] = torch.linspace(1, 0, n + 2)[1:-1]
    return w


@torch.no_grad()
def encode_conditioning(policy: DiTResNetPolicy, batch: dict) -> Tensor:
    return policy.observation_encoder.encode(batch)


def sample_rtc(policy: DiTResNetPolicy, cond: Tensor, leftover: Tensor | None, weights: Tensor | None, max_guidance: float = 10.0) -> Tensor:
    """Real-Time Chunking (Physical Intelligence; LeRobot policies/rtc/modeling_rtc.py) on multi_task_dit's flow matching,
    whose time runs from 0 (noise) to 1 (data): x_t = t a + (1 - t) e, v = a - e, predicted actions a_hat = x_t + (1 - t) v.
    Each Euler step's velocity is corrected toward the previous chunk's leftover plan on the overlap:
    v += w(t) J^T ((leftover - a_hat) * weights), w(t) = min(c / r^2-term of the RTC paper (tau = t), max_guidance).
    Returns the full normalised chunk [B, horizon, A]. leftover None: plain sampling."""
    obj, model = policy.objective, policy.noise_predictor
    b = cond.shape[0]; dev, dt_ = cond.device, cond.dtype
    x = torch.randn((b, obj.horizon, obj.action_dim), dtype=dt_, device=dev)
    grid = torch.linspace(0, 1, policy.config.num_integration_steps + 1, device=dev)
    wts = None if leftover is None else weights.to(dev).view(1, -1, 1)
    for i in range(len(grid) - 1):
        t = grid[i].item(); dt = (grid[i + 1] - grid[i]).item()
        tb = torch.full((b,), t, dtype=dt_, device=dev)
        if leftover is None:
            with torch.no_grad():
                v = model(x, tb, conditioning_vec=cond)
        else:
            with torch.enable_grad():
                xg = x.detach().requires_grad_(True)
                v = model(xg, tb, conditioning_vec=cond)
                a_hat = xg + (1.0 - t) * v
                err = ((leftover - a_hat) * wts).detach()
                corr = torch.autograd.grad(a_hat, xg, err)[0]
            tau = t
            if tau <= 0.0:
                gw = max_guidance
            else:
                inv_r2 = ((1 - tau) ** 2 + tau ** 2) / max((1 - tau) ** 2, 1e-12)
                gw = min((1 - tau) / tau * inv_r2, max_guidance)
            v = v.detach() + gw * corr
        x = (x + dt * v).detach()
    return x
