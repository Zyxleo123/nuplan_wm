"""FlowDrive best-of-K under AnchorWM's scorer (+ the repair on every candidate), as a nuPlan planner
(``planner=flow_drive_select``).

Every replan (``best_of_k``): ``population`` FlowDrive samples -> each repaired (``enable_repair``, batched, as the ES
planner repairs) -> each extended to the scorer's 8 s -> AnchorWM's ``PlanScorer`` total score -> the best candidate's
4 s plan goes to the simulator. AnchorWM runs once per replan on the online scene with the drivable-area boundary the
scorer reads (as ``diffusion_es.planner`` builds it); the same forward gives the repair its ``anchor`` forecast.

Seeded as ``fd`` (``seed + iteration``), and sample 0 is ``fd``'s own sample, so ``population: 1`` without the repair
is ``fd`` bit for bit. ``stats_dir`` rows carry the ES planner's keys (``sim.readout``'s ES summary reads them).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from diffusion_es.es import spread_m
from diffusion_es.planner import CACHE_DA_POINTS, CACHE_DA_RADIUS_M
from flow_drive.best_of_k import EXTENSIONS, best_of_k, sample_population
from flow_drive.planner.planner import set_seed
from flow_drive.repair_planner import RepairedFlowDrivePlanner
from posthoc_repair.anchor_forecast import AnchorForecaster, online_scene
from posthoc_repair.scene import DrivableArea, current_objects, ego_pose, forecast_boxes


class SelectFlowDrivePlanner(RepairedFlowDrivePlanner):
    def __init__(self, *args, population: int = 32, extension: str = "speed", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self._anchor_ckpt:
            raise ValueError("best-of-K scores with AnchorWM: set anchor_ckpt (an anchor_wm.train checkpoint)")
        if extension not in EXTENSIONS:
            raise ValueError(f"extension {extension!r}: one of {EXTENSIONS}")
        self._k = int(population)
        self._extension = extension
        refresh = 25.0
        self._scorer_da = DrivableArea(radius_m=2.0 ** 0.5 * CACHE_DA_RADIUS_M + refresh + 5.0, refresh_m=refresh)

    def name(self) -> str:
        return "flow_drive_select_repair" if self._enable_repair else "flow_drive_select"

    def initialize(self, initialization) -> None:
        if self._anchor is None:                # before super(): its forecaster must be the scorer-checked one
            self._anchor = AnchorForecaster(self._anchor_ckpt, device=self._device, need_scorer=True)
        super().initialize(initialization)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        t_all = time.perf_counter()
        ego_state, observation = current_input.history.current_state
        if self._t0_us is None:
            self._t0_us = int(ego_state.time_us)
        if self._seed is not None:
            set_seed(self._seed + self._iteration)
        self._iteration += 1
        raw = self.planner_input_to_model_inputs(current_input)
        t0 = time.perf_counter()
        pred = sample_population(self._model, raw, self._k)                             # [K, T, 3]
        plans = torch.cat([torch.zeros_like(pred[:, :1]), pred], 1)                     # [K, 1 + T, 3]; row 0 = now
        t1 = time.perf_counter()

        pose = ego_pose(ego_state)
        z = online_scene(current_input.history, list(current_input.traffic_light_data), self._map_api,
                         self._route_roadblock_ids, self._initialization.mission_goal,
                         drivable=self._scorer_da.cache_arrays(self._map_api, pose, CACHE_DA_RADIUS_M, CACHE_DA_POINTS))
        t2 = time.perf_counter()
        self._anchor.prepare(z)
        t3 = time.perf_counter()

        repair_fn = None
        if self._enable_repair:
            dev, cfg = self._repair_device, self._repair.cfg
            edges = self._drivable(self._map_api, pose, dev)
            radius = -1.0 if self._neighbor_forecast == "none" else self._object_radius_m
            predicted = self._anchor.forecast() if self._neighbor_forecast == "anchor" else None
            agents = forecast_boxes(current_objects(observation, pose, radius), int(min(cfg.window, plans.shape[1] - 1)),
                                    cfg.dt, dev, predicted=predicted)
            k = len(plans)
            batch = lambda d: {key: v.expand(k, *v.shape[1:]) for key, v in d.items()}  # noqa: E731

            def repair_fn(p: torch.Tensor):
                return self._repair(p.to(dev, torch.float32), batch(edges), batch(agents))

        b, plans, _, info = best_of_k(plans, lambda p: self._anchor.score(p)[0], int(self._anchor.cfg.future_steps),
                                      repair_fn, self._extension)
        plan = plans[b].detach().cpu().numpy().astype(np.float64)
        s_mean, s_end = spread_m(torch.cat([torch.zeros_like(pred[:, :1]), pred], 1))
        self._record_select(ego_state, {**info, "best0": info["best"], "born": 0, "spread0_m": s_mean,
                                        "spread0_end_m": s_end, "ms_sample": round(1e3 * (t1 - t0), 1),
                                        "ms_scene": round(1e3 * (t2 - t1), 1), "ms_wm": round(1e3 * (t3 - t2), 1),
                                        "ms": round(1e3 * (time.perf_counter() - t_all), 1)})
        states = transform_predictions_to_states(plan[1:], current_input.history.ego_states,
                                                 self._future_horizon, self._step_interval)
        return InterpolatedTrajectory(trajectory=states)

    def _record_select(self, ego_state, row: dict) -> None:
        if not self._stats_dir:
            return
        if self._stats_file is None:            # one file per simulation (planner instance)
            os.makedirs(self._stats_dir, exist_ok=True)
            self._stats_file = os.path.join(self._stats_dir,
                                            f"{self._map_api.map_name}_{self._t0_us}_{os.getpid()}.jsonl")
        row = {"t_us": int(ego_state.time_us), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}}
        with open(self._stats_file, "a") as f:
            f.write(json.dumps(row) + "\n")
