"""Diffusion Planner + the o_am post-hoc repair, as a nuPlan planner.

Every replan: Diffusion Planner's plan (80 x 0.1 s, ego frame) -> ``PosthocRepair`` against the drivable-area
boundary and the other objects' forecast boxes over the first ``window`` steps -> the (possibly edited) 80
poses go to the simulator as Diffusion Planner's would. A plan already clear of every margin is returned
unchanged, so on clean replans this planner is Diffusion Planner.

``neighbor_forecast``: ``cv`` (constant velocity from the current tracks; static objects stay put), ``dp``
(Diffusion Planner's own joint prediction for the neighbours it predicts, constant velocity for the rest),
``anchor`` (AnchorWM's probability-weighted modes from ``anchor_ckpt``, see anchor_forecast.py; constant velocity
for the objects it does not decode) or ``none`` (drivable area only). ``stats_dir`` writes one JSON line per replan, one file per scenario.

``seed`` fixes Diffusion Planner's sampling noise per replan (from the scenario's start time and the iteration),
so a repair arm and ``enable_repair: false`` draw identical plans until the repair first moves one, and every
outcome difference between them is the repair's. ``seed: null`` samples as Diffusion Planner does.
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

from diffusion_planner.planner.planner import DiffusionPlanner
from posthoc_repair.optimizer import PosthocRepair, RepairConfig
from posthoc_repair.scene import NEIGHBOR_FORECASTS, DrivableArea, current_objects, ego_pose, forecast_boxes


class RepairedDiffusionPlanner(DiffusionPlanner):
    def __init__(self, *args, repair: Optional[dict] = None, neighbor_forecast: str = "cv",
                 repair_device: str = "cpu", object_radius_m: float = 60.0, stats_dir: Optional[str] = None,
                 enable_repair: bool = True, seed: Optional[int] = 0, anchor_ckpt: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if neighbor_forecast not in NEIGHBOR_FORECASTS:
            raise ValueError(f"neighbor_forecast {neighbor_forecast!r}: one of {NEIGHBOR_FORECASTS}")
        self._repair = PosthocRepair(RepairConfig.from_dict(repair))
        self._neighbor_forecast = neighbor_forecast
        self._repair_device = torch.device(repair_device)
        self._object_radius_m = float(object_radius_m)
        self._drivable = DrivableArea()
        self._stats_dir = stats_dir
        self._stats_file = None
        self._enable_repair = bool(enable_repair)
        self._seed = seed
        self._t0_us = None
        if neighbor_forecast == "anchor" and not anchor_ckpt:
            raise ValueError("neighbor_forecast 'anchor' needs anchor_ckpt (an anchor_wm.train checkpoint)")
        self._anchor_ckpt = anchor_ckpt
        self._anchor = None                     # loaded in initialize(), on the planner's device

    def name(self) -> str:
        return "diffusion_planner_repair" if self._enable_repair else "diffusion_planner_seeded"

    def initialize(self, initialization) -> None:
        super().initialize(initialization)
        if self._neighbor_forecast == "anchor" and self._anchor is None:
            from posthoc_repair.anchor_forecast import AnchorForecaster
            self._anchor = AnchorForecaster(self._anchor_ckpt, device=self._device)

    @staticmethod
    def _dp_neighbors(raw_inputs: dict, outputs: dict) -> dict:
        """Diffusion Planner's predicted neighbour slots (ego frame) as one-mode forecasts (scene.forecast_boxes)."""
        pred = outputs["prediction"][0, 1:].detach().cpu().numpy().astype(np.float64)       # [P, T, 4]
        cur = raw_inputs["neighbor_agents_past"][0, :pred.shape[0], -1].detach().cpu().numpy().astype(np.float64)
        ok = np.any(cur[:, :4] != 0, -1)
        pred, cur = pred[ok], cur[ok]
        return {"xy": pred[:, None, :, :2], "yaw": np.arctan2(pred[:, None, :, 3], pred[:, None, :, 2]),
                "prob": np.ones((len(pred), 1)), "cur_xy": cur[:, :2], "lw": cur[:, [7, 6]]}

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        ego_state, observation = current_input.history.current_state
        if self._t0_us is None:
            self._t0_us = int(ego_state.time_us)
        if self._seed is not None:
            key = f"{self._seed}/{self._map_api.map_name}/{self._t0_us}/{current_input.iteration.index}"
            torch.manual_seed(zlib.crc32(key.encode()))
        raw = self.planner_input_to_model_inputs(current_input)
        _, outputs = self._planner(self.observation_normalizer(raw))
        pred = outputs["prediction"][0, 0].detach().cpu().numpy().astype(np.float64)        # [T, 4]
        plan = np.concatenate([np.zeros((1, 3)), np.concatenate(
            [pred[:, :2], np.arctan2(pred[:, 3], pred[:, 2])[:, None]], -1)], 0)          # index 0 = now
        if self._enable_repair:
            plan = self._repaired(plan, current_input, raw, outputs)
        states = transform_predictions_to_states(plan[1:], current_input.history.ego_states,
                                                 self._future_horizon, self._step_interval)
        return InterpolatedTrajectory(trajectory=states)

    def _repaired(self, plan: np.ndarray, current_input: PlannerInput, raw: dict, outputs: dict) -> np.ndarray:
        t0 = time.perf_counter()
        ego_state, observation = current_input.history.current_state
        pose, dev = ego_pose(ego_state), self._repair_device
        cfg = self._repair.cfg
        n = int(min(cfg.window, len(plan) - 1))
        edges = self._drivable(self._map_api, pose, dev)
        radius = -1.0 if self._neighbor_forecast == "none" else self._object_radius_m
        objects = current_objects(observation, pose, radius)
        predicted = None
        if self._neighbor_forecast == "dp":
            predicted = self._dp_neighbors(raw, outputs)
        elif self._neighbor_forecast == "anchor":
            predicted = self._anchor(current_input, self._map_api, self._route_roadblock_ids,
                                     self._initialization.mission_goal)
        agents = forecast_boxes(objects, n, cfg.dt, dev, predicted=predicted)
        out, info = self._repair(torch.as_tensor(plan, dtype=torch.float32, device=dev)[None], edges, agents)
        repaired = out[0].cpu().numpy().astype(np.float64)
        self._record(ego_state, info, edges, agents, time.perf_counter() - t0)
        return repaired if bool(info["moved"][0]) else plan

    def _record(self, ego_state, info: dict, edges: dict, agents: dict, seconds: float) -> None:
        if not self._stats_dir:
            return
        if self._stats_file is None:            # one file per simulation (planner instance)
            os.makedirs(self._stats_dir, exist_ok=True)
            self._stats_file = os.path.join(self._stats_dir,
                                            f"{self._map_api.map_name}_{self._t0_us}_{os.getpid()}.jsonl")
        row = {"t_us": int(ego_state.time_us), "ms": round(1e3 * seconds, 2),
               "n_edges": int(edges["edge_xy"].shape[1]), "n_objects": int(agents["box"].shape[2])}
        for k in ("loss0", "loss", "shift_m", "c_edge0", "c_agent0", "c_edge", "c_agent"):
            v = float(info[k][0])
            row[k] = round(v, 5) if np.isfinite(v) else None
        row["moved"] = bool(info["moved"][0])
        row["nonfinite"] = bool(info.get("nonfinite", torch.zeros(1, dtype=torch.bool))[0])
        with open(self._stats_file, "a") as f:
            f.write(json.dumps(row) + "\n")
