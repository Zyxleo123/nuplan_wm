"""FlowDrive's model at inference: the scene encoder and the DiT velocity field, integrated by 8 Euler steps.

Upstream ``FlowDrivePlanner.forward`` (no guidance, one sample) without upstream's training-time imports. The EMA
weights are copied as diffusers' ``EMAModel.copy_to`` copies them (the shadow params, in ``parameters()`` order),
and the Euler step is diffusers' ``FlowMatchEulerDiscreteScheduler`` (shift 1) written out; ``check.py`` holds
both to the originals bit for bit.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml

from flow_drive.model.module.dit import DiT
from flow_drive.model.module.encoder import Encoder
from flow_drive.utils.normalizer import ObservationNormalizer, StateNormalizer

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
ACTION_DIM = 4  # x, y, cos, sin


def _namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**{k: _namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_params(path: str | None = None) -> SimpleNamespace:
    with open(path or os.path.join(CONFIG_DIR, "config.yaml")) as f:
        params = _namespace(yaml.safe_load(f))
    if not params.data_processing.normalization_file_path:
        params.data_processing.normalization_file_path = os.path.join(CONFIG_DIR, "normalization.json")
    return params


def euler_sigmas(num_train_timesteps: int, num_inference_steps: int) -> torch.Tensor:
    """``FlowMatchEulerDiscreteScheduler(num_train_timesteps).set_timesteps(num_inference_steps).sigmas``: the
    steps' noise levels from 1 down to 1 / num_train_timesteps, then 0 (float32, on the CPU)."""
    n = num_train_timesteps
    train = torch.from_numpy(np.linspace(1, n, n, dtype=np.float32)[::-1].copy()) / n
    t = np.linspace(train[0].item() * n, train[-1].item() * n, num_inference_steps)
    sigmas = torch.from_numpy(t / n).to(dtype=torch.float32)
    return torch.cat([sigmas, torch.zeros(1)])


def load_ema_(module: nn.Module, ema_state: dict) -> None:
    """diffusers ``EMAModel.load_state_dict`` then ``copy_to(module.parameters())``."""
    shadow, params = ema_state["shadow_params"], list(module.parameters())
    if len(shadow) != len(params) or any(s.shape != p.shape for s, p in zip(shadow, params)):
        raise ValueError(f"EMA state has {len(shadow)} params, {type(module).__name__} has {len(params)} "
                         "(or their shapes differ): not this model's checkpoint")
    with torch.no_grad():
        for p, s in zip(params, shadow):
            p.copy_(s.to(p.device))


class FlowDrivePlanner(nn.Module):
    def __init__(self, params: SimpleNamespace, ckpt_path: str, device: str = "cpu"):
        super().__init__()
        self.params = params
        self.encoder = Encoder(params.encoder)
        self.decoder = DiT(n_blocks=params.diffuser.n_blocks, action_dim=ACTION_DIM,
                           pred_horizon=params.diffuser.pred_horizon + 1,  # + the current ego state
                           hidden_dim=params.diffuser.hidden_dim, heads=params.diffuser.num_heads,
                           dropout=params.diffuser.dropout)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        load_ema_(self.encoder, ckpt["ema_encoder_state_dict"])
        load_ema_(self.decoder, ckpt["ema_decoder_state_dict"])
        self.encoder = self.encoder.to(device).eval()
        self.decoder = self.decoder.to(device).eval()
        self.sigmas = euler_sigmas(params.inference.flow_train_iter, params.inference.flow_inference_iter)
        self.action_normalizer = StateNormalizer.from_json(params.data_processing)
        self.observation_normalizer = ObservationNormalizer.from_json(params.data_processing)

    def forward(self, inputs: dict) -> torch.Tensor:
        """Raw model inputs (``DataProcessor.observation_adapter``) -> the plan [B, T, (x, y, yaw)], ego frame."""
        with torch.inference_mode():
            inputs = self.observation_normalizer(inputs)
            cond = self.encoder(inputs)
            ego = inputs["ego_current_state"][:, :4]  # [B, 4] x, y, cos, sin (normalised)
            device = cond["encoding"].device
            B, T = cond["encoding"].shape[0], self.params.diffuser.pred_horizon

            x = torch.randn((B, T, ACTION_DIM), device=device)
            x = torch.cat([ego.unsqueeze(1), x], dim=1)  # [B, T + 1, 4]; index 0 stays the current state
            # The velocity field's time input runs 0 -> train_iter - 1 while sigma runs 1 -> 0, as upstream samples.
            t_int = torch.linspace(0, self.params.inference.flow_train_iter - 1, steps=len(self.sigmas) - 1,
                                   device=device).long()
            for i, t in enumerate(t_int):
                v = self.decoder(x, t, global_cond=cond)
                dt = self.sigmas[i + 1] - self.sigmas[i]
                x = (x.to(torch.float32) + dt * v).to(v.dtype)
                x[:, 0, :] = ego

            action = self.action_normalizer.inverse(x[:, 1:, :])
            heading = torch.arctan2(action[:, :, 3], action[:, :, 2])
            return torch.cat([action[:, :, :2], heading.unsqueeze(-1)], dim=-1)
