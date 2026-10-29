"""Diffusion Planner + Diffusion-ES + the o_am repair + AnchorWM's learned scorer, as a nuPlan planner.

Every replan (``es.diffusion_es``): a population of Diffusion Planner samples -> each candidate repaired
(``PosthocRepair``, batched, against the drivable area and the objects' forecast boxes) -> scored by AnchorWM's
``PlanScorer`` (the total head of an ``anchor_wm.train`` checkpoint) -> elites re-noised and denoised for
``es.iterations`` rounds -> the best-scoring repaired plan goes to the simulator.

AnchorWM runs once per replan (``anchor_model.AnchorWM.prepare``) on the cache-format scene built online
(``anchor_forecast.online_scene``, with the drivable-area boundary the scorer reads, cropped as the cache crops it);
the same forward gives the repair its forecast when ``neighbor_forecast: anchor``.

``enable_repair: false`` searches and scores the raw samples (ES + scorer alone). ``seed`` seeds every replan as
``RepairedDiffusionPlanner`` does. ``stats_dir`` writes one JSON line per replan (scores per iteration, which
iteration the chosen plan was born in, how many candidates the repair moved, timings).
"""

from __future__ import annotations

import json
import os
import time
import zlib
from typing import Optional

import numpy as np
import torch

from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from diffusion_es.es import ESConfig, PopulationSampler, diffusion_es
from posthoc_repair.anchor_forecast import AnchorForecaster, online_scene
from posthoc_repair.planner import RepairedDiffusionPlanner
from posthoc_repair.scene import DrivableArea, current_objects, ego_pose, forecast_boxes

# build_cache's --da_radius / --max_da_points (what the scorer was trained on).
CACHE_DA_RADIUS_M, CACHE_DA_POINTS = 150.0, 8192


class ESRepairDiffusionPlanner(RepairedDiffusionPlanner):
    def __init__(self, *args, es: Optional[dict] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self._anchor_ckpt:
            raise ValueError("the ES planner scores with AnchorWM: set anchor_ckpt (an anchor_wm.train checkpoint)")
        self._es = ESConfig.from_dict(es)
        refresh = 25.0
        self._scorer_da = DrivableArea(radius_m=2.0 ** 0.5 * CACHE_DA_RADIUS_M + refresh + 5.0, refresh_m=refresh)
        self._sampler = None

    def name(self) -> str:
        return "diffusion_planner_es_repair" if self._enable_repair else "diffusion_planner_es"

    def initialize(self, initialization) -> None:
        super().initialize(initialization)
        if self._anchor is None:
            self._anchor = AnchorForecaster(self._anchor_ckpt, device=self._device, need_scorer=True)
        self._sampler = PopulationSampler(self._planner)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        t_all = time.perf_counter()
        ego_state, observation = current_input.history.current_state
        if self._t0_us is None:
            self._t0_us = int(ego_state.time_us)
        if self._seed is not None:
            key = f"{self._seed}/{self._map_api.map_name}/{self._t0_us}/{current_input.iteration.index}"
            torch.manual_seed(zlib.crc32(key.encode()))
        raw = self.planner_input_to_model_inputs(current_input)
        ctx = self._sampler.encode(self.observation_normalizer(raw))

        t0 = time.perf_counter()
        pose = ego_pose(ego_state)
        z = online_scene(current_input.history, list(current_input.traffic_light_data), self._map_api,
                         self._route_roadblock_ids, self._initialization.mission_goal,
                         drivable=self._scorer_da.cache_arrays(self._map_api, pose, CACHE_DA_RADIUS_M, CACHE_DA_POINTS))
        t1 = time.perf_counter()
        self._anchor.prepare(z)
        ms_scene, ms_wm = 1e3 * (t1 - t0), 1e3 * (time.perf_counter() - t1)

        dev, cfg = self._repair_device, self._repair.cfg
        state = {"agents": None, "moved": 0, "charged": 0}
        if self._enable_repair:
            edges = self._drivable(self._map_api, pose, dev)
            radius = -1.0 if self._neighbor_forecast == "none" else self._object_radius_m
            objects = current_objects(observation, pose, radius)

        def repair_fn(plans: torch.Tensor, pred: torch.Tensor):
            if not self._enable_repair:
                return plans, {"moved": torch.zeros(len(plans), dtype=torch.bool)}
            if state["agents"] is None:     # the first population's sample 0 gives the `dp` forecast
                n = int(min(cfg.window, plans.shape[1] - 1))
                predicted = None
                if self._neighbor_forecast == "dp":
                    predicted = self._dp_neighbors(raw, {"prediction": pred[:1]})
                elif self._neighbor_forecast == "anchor":
                    predicted = self._anchor.forecast()
                state["agents"] = forecast_boxes(objects, n, cfg.dt, dev, predicted=predicted)
            k = len(plans)
            batch = lambda d: {key: v.expand(k, *v.shape[1:]) for key, v in d.items()}  # noqa: E731
            out, info = self._repair(plans.to(dev, torch.float32), batch(edges), batch(state["agents"]))
            state["moved"] += int(info["moved"].sum())
            state["charged"] += int((info["loss0"] > 0).sum())
            return out.to(plans.device), info

        best, info = diffusion_es(self._sampler, ctx, self._es, repair_fn, lambda p: self._anchor.score(p)[0])
        plan = best.detach().cpu().numpy().astype(np.float64)
        self._record_es(ego_state, info, state, ms_scene, ms_wm, time.perf_counter() - t_all)
        states = transform_predictions_to_states(plan[1:], current_input.history.ego_states,
                                                 self._future_horizon, self._step_interval)
        return InterpolatedTrajectory(trajectory=states)

    def _record_es(self, ego_state, info: dict, state: dict, ms_scene: float, ms_wm: float, seconds: float) -> None:
        if not self._stats_dir:
            return
        if self._stats_file is None:            # one file per simulation (planner instance)
            os.makedirs(self._stats_dir, exist_ok=True)
            self._stats_file = os.path.join(self._stats_dir,
                                            f"{self._map_api.map_name}_{self._t0_us}_{os.getpid()}.jsonl")
        row = {"t_us": int(ego_state.time_us), "ms": round(1e3 * seconds, 1), "ms_scene": round(ms_scene, 1),
               "ms_wm": round(ms_wm, 1), "charged": state["charged"], "repaired": state["moved"]}
        row.update({k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items()})
        with open(self._stats_file, "a") as f:
            f.write(json.dumps(row) + "\n")
