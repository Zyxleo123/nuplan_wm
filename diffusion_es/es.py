"""Diffusion-ES on Diffusion Planner (yixiz/waymax_rs ``planner/diffusion_es_planner.py``), in torch, nuPlan-free.

One replan::

    population  K samples of Diffusion Planner (one encoder pass, the decoder batched over K)
    repeat      repair every new candidate -> score the repaired plans -> keep the top E (elites) ->
    (iters)     re-noise the elites' raw samples to diffusion time ``resample_t`` and denoise them back
                (``resample_steps`` DPM-Solver steps), K children shared round-robin among the elites
    pick        the best-scoring repaired plan of the last pool (elites + children)

As in the WOMD planner, a candidate is repaired before it is scored, so elites are chosen on (and the plan
executed is) the repaired plan, while resampling restarts from the elite's raw diffusion sample (re-decoding a
repaired plan would undo the repair). Elites carry over with their scores, so the best score never drops
between iterations.

Diffusion Planner denoises the ego and its ``predicted_neighbor_num`` neighbours jointly; a candidate is that
whole joint sample, re-noised as a whole. Its prior is ``0.5 * N(0, I)`` at t = 1 (a 0.5 temperature), so the
re-noising scales its noise by ``noise_scale`` (0.5 by default) too.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler


@dataclass(frozen=True)
class ESConfig:
    """waymax_rs ``run_simulation.py`` defaults (population 64, 8 elites, 4 iterations). Its partial re-noising
    was 5 of 100 cosine DDPM steps (alpha ~ 0.99); ``resample_t`` is the continuous VP-linear time here
    (0.1: alpha 0.95, sigma 0.32 before ``noise_scale``)."""

    population: int = 64
    elites: int = 8
    iterations: int = 4
    resample_t: float = 0.1
    resample_steps: int = 3
    noise_scale: float = 0.5
    sample_steps: int = 10          # DPM-Solver steps of the initial population (Diffusion Planner's 10)

    def __post_init__(self) -> None:
        if not 1 <= self.elites <= self.population:
            raise ValueError("ES: need 1 <= elites <= population")
        if not 0.0 < self.resample_t <= 1.0:
            raise ValueError("ES: resample_t must be in (0, 1]")

    @classmethod
    def from_dict(cls, value: Optional[dict]) -> "ESConfig":
        value = dict(value or {})
        unknown = set(value) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown ES keys {sorted(unknown)}")
        return cls(**value)


class PopulationSampler:
    """Diffusion Planner's decoder (``diffusion_planner.model.module.decoder.Decoder``'s inference branch) over a
    population: ``encode`` once per replan, then ``sample`` K candidates from noise or from re-noised proposals."""

    def __init__(self, planner_model) -> None:
        self.model = planner_model                       # diffusion_planner.model.diffusion_planner.Diffusion_Planner
        self.decoder = planner_model.decoder.decoder
        if self.decoder._guidance_fn is not None:
            raise ValueError("ES samples without classifier guidance (guidance_fn must be null)")
        self.schedule = dpm.NoiseScheduleVP(schedule="linear")

    @torch.no_grad()
    def encode(self, inputs: dict) -> dict:
        """``inputs``: the observation-normalised model inputs of one scene (batch 1)."""
        dec = self.decoder
        enc = self.model.encoder(inputs)
        ego = inputs["ego_current_state"][:, None, :4]
        nb = inputs["neighbor_agents_past"][:, :dec._predicted_neighbor_num, -1, :4]
        mask = torch.sum(torch.ne(nb[..., :4], 0), dim=-1) == 0
        return {"encoding": enc["encoding"], "route_lanes": inputs["route_lanes"], "mask": mask,
                "current": torch.cat([ego, nb], dim=1)}                                     # [1, P, 4]

    @torch.no_grad()
    def sample(self, ctx: dict, k: int, proposals: Optional[torch.Tensor] = None, t_start: float = 1.0,
               steps: int = 10, noise_scale: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """K joint samples -> (raw ``[K, P, (1 + T) * 4]``, normalised as the DiT sees them; ``prediction``
        ``[K, P, T, 4]``, x / y / cos / sin in the ego frame). ``proposals`` (raw samples) are re-noised to
        ``t_start`` and denoised from there; without them the sampling starts from the prior at t = 1."""
        dec = self.decoder
        cur = ctx["current"].expand(k, -1, -1)
        P, T = cur.shape[1], dec._future_len
        if proposals is None:
            # drawn on the CPU and moved, as the decoder does, so a seed gives Diffusion Planner's own noise
            noise = torch.randn(k, P, T, 4).to(cur.device) * noise_scale
            x = torch.cat([cur[:, :, None], noise], dim=2).reshape(k, P, -1)
            sample_params = {}
        else:
            t = torch.full((1,), float(t_start), device=cur.device)
            alpha, sigma = self.schedule.marginal_alpha(t), self.schedule.marginal_std(t)
            x = alpha * proposals + sigma * noise_scale * torch.randn(proposals.shape).to(cur.device)
            x = x.reshape(k, P, -1, 4)
            x[:, :, 0] = cur
            x = x.reshape(k, P, -1)
            sample_params = {"t_start": float(t_start)}

        def initial_state_constraint(xt, t, step):
            xt = xt.reshape(k, P, -1, 4)
            xt[:, :, 0, :] = cur
            return xt.reshape(k, P, -1)

        x0 = dpm_sampler(
            dec.dit, x,
            other_model_params={"cross_c": ctx["encoding"].expand(k, -1, -1),
                                "route_lanes": ctx["route_lanes"].expand(k, -1, -1, -1),
                                "neighbor_current_mask": ctx["mask"].expand(k, -1)},
            diffusion_steps=int(steps),
            dpm_solver_params={"correcting_xt_fn": initial_state_constraint},
            sample_params=sample_params,
        )
        pred = dec._state_normalizer.inverse(x0.reshape(k, P, -1, 4))[:, :, 1:]
        return x0, pred


def plans_of(pred: torch.Tensor) -> torch.Tensor:
    """Ego predictions ``[K, P, T, 4]`` -> plans ``[K, 1 + T, 3]`` (x, y, yaw; row 0 = the current pose)."""
    ego = pred[:, 0]
    body = torch.cat([ego[..., :2], torch.atan2(ego[..., 3], ego[..., 2])[..., None]], -1)
    return torch.cat([torch.zeros_like(body[:, :1]), body], 1)


def spread_m(plans: torch.Tensor) -> tuple[float, float]:
    """How much a population differs: mean over candidates and steps of ``|plan - mean plan|`` and the largest
    endpoint deviation (m). A near-deterministic denoiser gives ~0, and then ES has nothing to select among."""
    xy = plans[:, 1:, :2]
    dev = (xy - xy.mean(0)).norm(dim=-1)
    return float(dev.mean()), float(dev[:, -1].max())


def elite_selector(elites: int, population: int) -> np.ndarray:
    """Elite slot each of the ``population`` children starts from (as the WOMD planner: contiguous blocks)."""
    base, rem = divmod(population, elites)
    return np.concatenate([np.full(base + (1 if e < rem else 0), e, np.int64) for e in range(elites)])


def diffusion_es(sampler: PopulationSampler, ctx: dict, cfg: ESConfig,
                 repair_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict]],
                 score_fn: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """The search of one replan; returns the chosen repaired plan ``[1 + T, 3]`` and per-replan stats.

    ``repair_fn(plans [K, 1 + T, 3], prediction [K, P, T, 4]) -> (plans, info)`` with ``info["moved"] [K]``;
    ``score_fn(plans [K, 1 + T, 3]) -> [K]``, higher is better."""
    k, e = int(cfg.population), int(cfg.elites)
    times = {"sample": 0.0, "repair": 0.0, "score": 0.0}

    def evaluate(raw, pred):
        t0 = time.perf_counter()
        plans, info = repair_fn(plans_of(pred), pred)
        t1 = time.perf_counter()
        scores = score_fn(plans).float()
        times["repair"] += t1 - t0
        times["score"] += time.perf_counter() - t1
        return plans, scores, info["moved"].to(scores.device)

    t0 = time.perf_counter()
    raw, pred = sampler.sample(ctx, k, steps=cfg.sample_steps, noise_scale=cfg.noise_scale)
    times["sample"] += time.perf_counter() - t0
    plans, scores, moved = evaluate(raw, pred)
    born = torch.zeros(k, dtype=torch.long, device=scores.device)
    s_mean, s_end = spread_m(plans_of(pred))
    first = {"score0": float(scores[0]), "best0": float(scores.max()), "mean0": float(scores.mean()),
             "moved0": int(moved.sum()), "spread0_m": s_mean, "spread0_end_m": s_end}
    best_by_iter = [float(scores.max())]
    pick = torch.as_tensor(elite_selector(e, k), device=scores.device)
    for it in range(1, int(cfg.iterations) + 1):
        top = torch.topk(scores, e).indices
        t0 = time.perf_counter()
        c_raw, c_pred = sampler.sample(ctx, k, proposals=raw[top][pick], t_start=cfg.resample_t,
                                       steps=cfg.resample_steps, noise_scale=cfg.noise_scale)
        times["sample"] += time.perf_counter() - t0
        c_plans, c_scores, c_moved = evaluate(c_raw, c_pred)
        raw, plans = torch.cat([raw[top], c_raw]), torch.cat([plans[top], c_plans])
        scores, moved = torch.cat([scores[top], c_scores]), torch.cat([moved[top], c_moved])
        born = torch.cat([born[top], torch.full((k,), it, dtype=torch.long, device=born.device)])
        best_by_iter.append(float(scores.max()))
    b = int(torch.argmax(scores))
    info = {**first, "best": float(scores[b]), "best_by_iter": [round(v, 4) for v in best_by_iter],
            "born": int(born[b]), "moved": bool(moved[b]),
            **{f"ms_{key}": round(1e3 * v, 1) for key, v in times.items()}}
    return plans[b], info
