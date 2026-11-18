"""Load the student without any download (11-18).

The checkpoint (student/model.safetensors) holds every weight of the model, but two constructors of the libraries fetch files
from the internet before the checkpoint is read:
  * the CLIP text encoder of the instruction (transformers CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16"), 600 MB)
    and its tokenizer,
  * the ResNet-18 image stem (torchvision ImageNet weights, 45 MB).
enable() makes both build the bare architecture instead: the tokenizer and the model's config come from the small cache in
assets/hf_home (4 MB, in this repository), the weights from the checkpoint. verify() then proves that nothing was left at its
random initial value: every tensor of the loaded model must be in the checkpoint file and equal to it.
STUDENT_OFFLINE=0 turns this off (the libraries then download as usual)."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_HOME = os.path.join(ROOT, "assets", "hf_home")


def wanted() -> bool:
    return os.environ.get("STUDENT_OFFLINE", "1") != "0" and os.path.isdir(HF_HOME)


def set_env() -> None:
    """Call before anything imports huggingface_hub / transformers (they read these variables at import)."""
    if wanted():
        os.environ["HF_HOME"] = HF_HOME
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def enable() -> None:
    if not wanted():
        return
    import torchvision
    from transformers import CLIPTextConfig, CLIPTextModel

    def text_model_from_config(cls, name, *args, **kwargs):  # weights: from the checkpoint (verify() checks it)
        return cls(CLIPTextConfig.from_pretrained(name))

    CLIPTextModel.from_pretrained = classmethod(text_model_from_config)
    resnet18 = torchvision.models.resnet18
    torchvision.models.resnet18 = lambda *args, **kwargs: resnet18(weights=None)  # weights: from the checkpoint


def verify(policy, model_dir: str) -> int:
    """Every tensor of the loaded policy equals the checkpoint file's. Returns the number of tensors; raises otherwise."""
    import torch
    from safetensors import safe_open

    state = policy.state_dict()
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt", device="cpu") as f:
        keys = set(f.keys())
        missing = sorted(k for k in state if k not in keys)
        if missing:
            raise RuntimeError(f"{len(missing)} tensors of the model are not in the checkpoint (they would be random): {missing[:5]}")
        for k, v in state.items():
            if not torch.equal(v.detach().cpu(), f.get_tensor(k).to(v.dtype)):
                raise RuntimeError(f"tensor {k} differs from the checkpoint")
    return len(state)
