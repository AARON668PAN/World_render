"""Policy "dit_resnet" (11-05): LeRobot's multi_task_dit (flow-matching diffusion transformer) with the vision encoder of
EgoEngine (arXiv 2606.12604, App. D): images ImageNet-normalised and encoded by a ResNet-18 truncated before global
average pooling; the feature map is read out by an HPT-style cross-attention stem (learnable query tokens) instead of
CLIP's single CLS token; flow matching with 10 integration steps at inference ("We use 10 inference steps in all
experiments", App. D.2). Not stated in the paper and chosen here: ImageNet weights, backbone trained at 0.1x the learning
rate (LeRobot's vision_encoder_lr_multiplier), 16 tokens of width 64, no resize or crop of the 168x224 image."""
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig


@PreTrainedConfig.register_subclass("dit_resnet")
@dataclass
class DiTResNetConfig(MultiTaskDiTConfig):
    objective: str = "flow_matching"
    num_integration_steps: int = 10
    image_resize_shape: tuple[int, int] | None = None
    image_crop_shape: tuple[int, int] | None = None
    vision_num_tokens: int = 16
    vision_token_dim: int = 64  # 16 x 256 made the per-layer adaLN projections of the 9k conditioning vector 171M parameters
    freeze_backbone: bool = False
    # 11-07, GR00T N1.7 style: while training, each sample's whole (normalised) state is replaced by zeros with this
    # probability (GR00T: state_dropout_prob 0.2), so the policy cannot lean on the joint positions alone
    state_dropout: float = 0.0
    # images enter the encoder in [0, 1]; the encoder applies the ImageNet normalisation itself
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
