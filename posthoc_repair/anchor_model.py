"""AnchorWM at planning time, nuPlan-free: one model forward per replan on a cache-format scene dict, then its
forecast (``scene.forecast_boxes``'s ``predicted``) and its learned plan score (``PlanScorer``'s total head).

The scene dict is ``anchor_forecast.online_scene``'s (the cache's keys). The scorer reads the drivable-area
boundary (``da_*``), so a scene built for scoring must carry it; the forecast does not read it.
"""

from __future__ import annotations

import numpy as np
import torch

from anchor_wm.config import AnchorWMConfig
from anchor_wm.data import agent_arrays, build_sample
from anchor_wm.model import AnchorWorldModel, select_modes


def load_model(ckpt_path: str, device, need_scorer: bool = False) -> AnchorWorldModel:
    """An ``anchor_wm.train`` checkpoint (its anchors are a buffer of the state dict, so no anchor file is needed).
    ``need_scorer``: refuse a forecasting-only run (``--no_scorer``), whose scorer weights are its random init."""
    ck = torch.load(ckpt_path, map_location="cpu")
    if need_scorer and ck.get("args", {}).get("no_scorer", False):
        raise ValueError(f"{ckpt_path} was trained with --no_scorer: its PlanScorer is untrained")
    cfg = AnchorWMConfig.from_dict(ck["config"])
    anchors = ck["model"]["anchors"].float() * float(cfg.max_range)
    model = AnchorWorldModel(cfg, anchors)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


class AnchorWM:
    """``prepare(z)`` runs the encoder / decoder on one scene; ``forecast()`` and ``score(plans)`` reuse it."""

    def __init__(self, ckpt_path: str | None = None, device="cpu", model: AnchorWorldModel | None = None,
                 need_scorer: bool = False) -> None:
        self.device = torch.device(device)
        self.model = model if model is not None else load_model(ckpt_path, self.device, need_scorer)
        self.cfg = self.model.config
        self._z = self._f = self._dec = self._ctx = None

    @torch.no_grad()
    def prepare(self, z: dict) -> "AnchorWM":
        sample = build_sample(z, self.cfg, with_targets=False)
        f = {k: torch.as_tensor(np.asarray(v))[None].to(self.device) for k, v in sample.items()}
        enc, dec = self.model(f)                 # the encoder / decoder see the ego through cfg.ego_input
        self._z, self._f, self._dec = z, f, dec
        # The scorer reads the unmodified inputs (as in training); `scale` only scales gradients.
        self._ctx = self.model.scorer.context(enc, dec, f, 1.0)
        return self

    @torch.no_grad()
    def forecast(self) -> dict:
        """Per agent valid at t = 0 (the ego excluded), its ``num_out_modes`` NMS modes in the ego frame (box
        centres): ``xy [P, m, T, 2]``, ``yaw [P, m, T]``, ``prob [P, m]`` (renormalised), ``cur_xy``, ``lw``."""
        cfg, f, dec = self.cfg, self._f, self._dec
        picks = select_modes(dec, int(cfg.num_out_modes), float(cfg.nms_dist_m), float(cfg.max_range))   # [1,N,m]
        gather = lambda x: x.gather(2, picks.view(*picks.shape, *([1] * (x.dim() - 3))).expand(  # noqa: E731
            *picks.shape, *x.shape[3:]))
        rng = float(cfg.max_range)
        cur_xy, cur_yaw = f["current_local_xy"].float(), f["current_yaw_local"].float()
        xy = cur_xy[:, :, None, None] + gather(dec["pred_xy_k"]).float() * rng                  # [1,N,m,T,2]
        yv = gather(dec["yaw_vec_k"]).float()
        yaw = cur_yaw[:, :, None, None] + torch.atan2(yv[..., 0], yv[..., 1])                  # [1,N,m,T]
        prob = gather(dec["probs"]).float()
        prob = prob / prob.sum(-1, keepdim=True).clamp(min=1e-9)
        keep = f["agent_valid"][0].bool() & ~f["is_sdc"][0].bool()
        lw = agent_arrays(self._z, int(cfg.max_agents))["lw"]                                   # build_sample's order
        out = {"xy": xy[0, keep], "yaw": yaw[0, keep], "prob": prob[0, keep], "cur_xy": cur_xy[0, keep]}
        out = {k: v.cpu().numpy().astype(np.float64) for k, v in out.items()}
        out["lw"] = np.asarray(lw, np.float64)[keep.cpu().numpy()]
        return out

    @torch.no_grad()
    def score(self, plans: torch.Tensor, chunk: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Plans ``[K, 1 + T, 3]`` (rear axle, ego frame, row 0 = now; the scorer's 0.1 s x ``future_steps``
        grid) -> total score ``[K]`` (higher is better) and sub-score logits ``[K, S]`` (``SUB_SCORES``)."""
        t = int(self.cfg.future_steps)
        if plans.shape[1] - 1 < t:
            raise ValueError(f"plans have {plans.shape[1] - 1} steps, the scorer {t}")
        p = plans[None, :, 1:t + 1].to(self.device, torch.float32)
        total, sub, _ = self.model.scorer(self._ctx, p[..., :2], p[..., 2], chunk=chunk)
        return total[0], sub[0]
