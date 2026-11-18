"""11-18: the vision student's inference core, shared by the simulator exam (runtime_server.py behind closed_loop_mw.py) and the
real-robot client (g1_client.py), so that what is tested in simulation is the code that runs on the robot.

The student (LeRobot plugin policy "dit_resnet", trained from scratch) reads
  observation.state        [S]  joint positions: 29 body joints in SONIC's order + 14 Dex3 finger joints (rad)
  observation.images.head  168 x 224 RGB head-camera image
  task                     the instruction sentence
for the last n_obs_steps (2) observations, one dataset step apart (STEP_TICKS = 4 control ticks = 80 ms), and returns a chunk of
`horizon` (13) actions, one per dataset step. Chunk index 0 belongs to the OLDER observation, index 1 is the action for the tick
after the current observation. Official-format students return 78 numbers per step = 64-d SONIC motion token | 14 finger targets
(rad); older ones 92 = token | 14 arm residuals | 14 finger targets.

plan() = the Real-Time-Chunking branch of teacher_multi/policy/policy_server.py, unchanged in behaviour: each new chunk is
sampled with the unexecuted rest of the previous one as a soft prefix, and its first `blend` steps are cross-faded with it.
(ego2libero python: torch, lerobot 0.6.1, lerobot_policy_dit_resnet)"""
from __future__ import annotations

import numpy as np
import torch

STEP_TICKS = 4  # one dataset step = 4 control ticks of 20 ms
TOKEN_DIM = 64


class StudentRuntime:
    def __init__(self, model_dir: str, device: str = "cuda"):
        from lerobot.utils.import_utils import register_third_party_plugins
        register_third_party_plugins()
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        cfg = PreTrainedConfig.from_pretrained(model_dir)
        if cfg.type != "dit_resnet":
            raise ValueError(f"{model_dir}: policy type {cfg.type}, expected dit_resnet")
        self.policy = get_policy_class(cfg.type).from_pretrained(model_dir).to(device).eval()
        self.pre, self.post = make_pre_post_processors(policy_cfg=self.policy.config, pretrained_path=model_dir,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        c = self.policy.config
        self.img_keys = [k for k in c.input_features if k.startswith("observation.images")]
        self.image_hw = tuple(c.input_features[self.img_keys[0]].shape[1:])  # (168, 224)
        self.state_dim = int(c.input_features["observation.state"].shape[0])
        self.action_dim = int(c.output_features["action"].shape[0])
        self.n_obs, self.horizon = int(c.n_obs_steps), int(c.horizon)
        self.finger_dim = self.action_dim - TOKEN_DIM if self.action_dim == TOKEN_DIM + 14 else 14
        self.official_format = self.action_dim == TOKEN_DIM + self.finger_dim
        self.model_dir = model_dir
        self.reset()

    def reset(self) -> None:
        """Forget the previous plan (start of an episode)."""
        self._prev = None  # the previous full normalised chunk [B, horizon, A]
        self._prev_k = 0   # how many of its steps were executed before this plan

    def info(self) -> dict:
        return dict(type="dit_resnet", model=self.model_dir, state_dim=self.state_dim, action_dim=self.action_dim, image_hw=list(self.image_hw),
                    n_obs=self.n_obs, horizon=self.horizon, official_format=self.official_format)

    @torch.no_grad()
    def plan(self, state_prev, state_cur, image_prev, image_cur, task, first: int, extra: int = 0, blend: int = 0, rtc: bool = True) -> np.ndarray:
        """state_* [B, S] float, image_* [B, H, W, 3] uint8 RGB (previous = one dataset step earlier), task str or list of B strings.
        Returns [B, first + extra, A] in real units: the actions for the next `first` dataset steps (+ `extra` more that are
        returned but not counted as executed), starting with the step right after the current observation."""
        from lerobot_policy_dit_resnet.modeling_dit_resnet import rtc_prefix_weights, sample_rtc
        sp, sc = np.asarray(state_prev, np.float32), np.asarray(state_cur, np.float32)
        B = len(sc); tasks = [task] * B if isinstance(task, str) else list(task)
        tens = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.0
        batch = {"observation.state": torch.from_numpy(np.stack([sp, sc], 1)[:, -self.n_obs:]), "task": tasks}
        im = torch.stack([tens(image_prev), tens(image_cur)], 1)[:, -self.n_obs:]
        for k in self.img_keys:
            batch[k] = im
        b = self.pre(batch); b = {k: v for k, v in b.items() if k != "action" and v is not None}
        if not rtc:
            act = self.post(self.policy._generate_actions(self.policy._prepare_batch(b)))[:, : first + extra]
            return act.float().cpu().numpy()
        cond = self.policy.observation_encoder.encode(self.policy._prepare_batch(b))
        H = self.horizon; prev = self._prev; leftover = weights = None
        if prev is not None and prev.shape[0] == cond.shape[0]:
            E = self._prev_k; L = H - E  # the previous chunk's index E + i is this chunk's index i (both start one step in the past)
            leftover = torch.zeros_like(prev); leftover[:, :L] = prev[:, E:]; weights = rtc_prefix_weights(1, L, H)
        full = sample_rtc(self.policy, cond, leftover, weights)
        act = full[:, 1:1 + first + extra].clone()
        if leftover is not None:
            for j in range(min(blend, first)):
                if E + 1 + j < H:
                    w = (j + 1) / (blend + 1); act[:, j] = (1 - w) * prev[:, E + 1 + j] + w * act[:, j]
        self._prev, self._prev_k = full.detach(), first
        return self.post(act).float().cpu().numpy()


def split_action(a: np.ndarray, finger_dim: int = 14):
    """[..., A] -> (token [..., 64], finger targets [..., 14]); works for the 78-d official format and the older 92-d one."""
    return a[..., :TOKEN_DIM], a[..., -finger_dim:]


def upsample(chunk: np.ndarray, ticks: int = STEP_TICKS, hold: int = TOKEN_DIM) -> np.ndarray:
    """[n, A] dataset steps -> [(n - 1) * ticks, A] control ticks, as closed_loop_mw.py: the first `hold` numbers (the token, a
    quantised code that must not be interpolated) are held for `ticks` ticks, the rest (finger targets) run linearly to the next
    step. The last step is only an interpolation end point (ask plan() for one `extra` step)."""
    c = np.asarray(chunk, np.float32); n = len(c) - 1
    out = np.repeat(c[:n], ticks, 0)
    for i in range(n):
        for j in range(ticks):
            out[i * ticks + j, hold:] = c[i, hold:] + (c[i + 1, hold:] - c[i, hold:]) * j / ticks
    return out
