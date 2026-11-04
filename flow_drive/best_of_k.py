"""FlowDrive best-of-K under AnchorWM's scorer, nuPlan-free.

One replan: K FlowDrive samples (one encoder pass, the DiT batched over K) -> each repaired (optional) -> each
extended from FlowDrive's 4 s to the scorer's 8 s (the last step's speed held along the last heading) -> the scorer's
total score -> the best candidate's own 4 s plan is executed; the extension is only for scoring.

Sample 0's noise is drawn exactly as ``FlowDrivePlanner.forward`` draws a single sample's, so under the same seed it
is FlowDrive's own plan: bit for bit at K = 1, up to batched-kernel rounding at K > 1.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch

from flow_drive.model.flow_drive_planner import ACTION_DIM, FlowDrivePlanner


@torch.no_grad()
def sample_population(model: FlowDrivePlanner, inputs: dict, k: int) -> torch.Tensor:
    """Raw model inputs of one scene -> ``k`` plans ``[k, T, (x, y, yaw)]`` (rear axle, ego frame).
    ``FlowDrivePlanner.forward`` over a population; under ``no_grad`` (not inference mode) so the repair can
    differentiate through its edit of these plans."""
    inputs = model.observation_normalizer(inputs)
    cond = model.encoder(inputs)
    ego = inputs["ego_current_state"][:, :4]
    device = cond["encoding"].device
    T = model.params.diffuser.pred_horizon

    noise = torch.randn((1, T, ACTION_DIM), device=device)                       # sample 0: forward's own draw
    if k > 1:
        noise = torch.cat([noise, torch.randn((k - 1, T, ACTION_DIM), device=device)])
    cond = {key: v.expand(k, *v.shape[1:]) for key, v in cond.items()}
    ego = ego.expand(k, -1)
    x = torch.cat([ego.unsqueeze(1), noise], dim=1)
    t_int = torch.linspace(0, model.params.inference.flow_train_iter - 1, steps=len(model.sigmas) - 1,
                           device=device).long()
    for i, t in enumerate(t_int):
        v = model.decoder(x, t, global_cond=cond)
        dt = model.sigmas[i + 1] - model.sigmas[i]
        x = (x.to(torch.float32) + dt * v).to(v.dtype)
        x[:, 0, :] = ego

    action = model.action_normalizer.inverse(x[:, 1:, :])
    heading = torch.arctan2(action[:, :, 3], action[:, :, 2])
    return torch.cat([action[:, :, :2], heading.unsqueeze(-1)], dim=-1)


EXTENSIONS = ("speed", "decel")


def extend_plans(plans: torch.Tensor, steps: int, mode: str = "speed") -> torch.Tensor:
    """``[K, 1 + T, 3]`` (row 0 = now) -> ``[K, 1 + steps, 3]``: the plan, then along its last heading either its last
    step's length repeated (``speed``: constant speed) or that length decaying linearly to 0 over the extension
    (``decel``: a constant-rate stop). ``speed`` keeps closing on a slower lead car through the unexecuted half,
    so the scorer judges that as the plan's risk; ``decel`` cannot add a collision the plan does not head into."""
    if mode not in EXTENSIONS:
        raise ValueError(f"extension {mode!r}: one of {EXTENSIONS}")
    extra = steps - (plans.shape[1] - 1)
    if extra <= 0:
        return plans[:, :steps + 1]
    step = (plans[:, -1, :2] - plans[:, -2, :2]).norm(dim=-1)                       # [K] m per step
    yaw = plans[:, -1, 2]
    i = torch.arange(1, extra + 1, device=plans.device, dtype=plans.dtype)
    dist = i if mode == "speed" else i - i * (i + 1) / (2 * extra)                 # steps travelled, in last-steps
    xy = plans[:, -1:, :2] + (step[:, None] * dist)[..., None] * torch.stack([yaw.cos(), yaw.sin()], -1)[:, None]
    return torch.cat([plans, torch.cat([xy, yaw[:, None, None].expand(-1, extra, 1)], -1)], 1)


def best_of_k(plans: torch.Tensor, score_fn: Callable[[torch.Tensor], torch.Tensor], score_steps: int,
              repair_fn: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, dict]]] = None,
              extension: str = "speed") -> tuple[int, torch.Tensor, torch.Tensor, dict]:
    """``plans [K, 1 + T, 3]`` -> (chosen index, the candidates as scored (repaired), their scores ``[K]``, stats).

    ``repair_fn(plans) -> (plans, info)`` with ``info["moved"] [K]`` (an unmoved candidate is kept bit for bit);
    ``score_fn(plans [K, 1 + score_steps, 3]) -> [K]``, higher is better."""
    t0 = time.perf_counter()
    moved = torch.zeros(len(plans), dtype=torch.bool, device=plans.device)
    charged = 0
    if repair_fn is not None:
        out, info = repair_fn(plans)
        moved = info["moved"].to(plans.device)
        charged = int((info["loss0"] > 0).sum())
        plans = torch.where(moved[:, None, None], out.to(plans.device, plans.dtype), plans)
    t1 = time.perf_counter()
    scores = score_fn(extend_plans(plans, score_steps, extension)).float()
    b = int(torch.argmax(scores))
    return b, plans, scores, {"best": float(scores[b]), "score0": float(scores[0]), "mean0": float(scores.mean()),
                              "chosen": b, "moved": bool(moved[b]), "repaired": int(moved.sum()), "charged": charged,
                              "ms_repair": round(1e3 * (t1 - t0), 1),
                              "ms_score": round(1e3 * (time.perf_counter() - t1), 1)}
