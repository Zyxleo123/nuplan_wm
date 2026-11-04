"""FlowDrive + the o_am post-hoc repair, as a nuPlan planner (``planner=flow_drive_repair``).

``RepairedDiffusionPlanner`` with FlowDrive's plan in place of Diffusion Planner's: every replan, FlowDrive's
40-pose plan (0.1 s, 4 s, ego frame) -> ``PosthocRepair`` over its first ``window`` steps -> the simulator. The
repair step (``_repaired``) and its stats (``_record``) are ``RepairedDiffusionPlanner``'s own, shared as they are.

FlowDrive is seeded as ``planner=flow_drive`` is (``seed + iteration``), so ``enable_repair: false`` is FlowDrive
alone bit for bit, and a repair arm differs from ``fd`` only from the first replan the repair moves.
``neighbor_forecast``: ``cv`` | ``anchor`` | ``none``; there is no ``dp``, FlowDrive predicts no neighbours.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from flow_drive.planner.planner import FlowDrivePlannerWrapper
from posthoc_repair.optimizer import PosthocRepair, RepairConfig
from posthoc_repair.planner import RepairedDiffusionPlanner
from posthoc_repair.scene import DrivableArea

NEIGHBOR_FORECASTS = ("cv", "anchor", "none")


class RepairedFlowDrivePlanner(FlowDrivePlannerWrapper):
    _repaired = RepairedDiffusionPlanner._repaired
    _record = RepairedDiffusionPlanner._record

    def __init__(self, *args, repair: Optional[dict] = None, neighbor_forecast: str = "cv",
                 repair_device: str = "cpu", object_radius_m: float = 60.0, stats_dir: Optional[str] = None,
                 enable_repair: bool = True, anchor_ckpt: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if neighbor_forecast not in NEIGHBOR_FORECASTS:
            raise ValueError(f"neighbor_forecast {neighbor_forecast!r}: one of {NEIGHBOR_FORECASTS}")
        if neighbor_forecast == "anchor" and not anchor_ckpt:
            raise ValueError("neighbor_forecast 'anchor' needs anchor_ckpt (an anchor_wm.train checkpoint)")
        self._repair = PosthocRepair(RepairConfig.from_dict(repair))
        self._neighbor_forecast = neighbor_forecast
        self._repair_device = torch.device(repair_device)
        self._object_radius_m = float(object_radius_m)
        self._drivable = DrivableArea()
        self._stats_dir = stats_dir
        self._stats_file = None
        self._enable_repair = bool(enable_repair)
        self._t0_us = None
        self._anchor_ckpt = anchor_ckpt
        self._anchor = None                     # loaded in initialize(), on the planner's device

    def name(self) -> str:
        return "flow_drive_repair" if self._enable_repair else "flow_drive_seeded"

    def initialize(self, initialization) -> None:
        super().initialize(initialization)
        if self._neighbor_forecast == "anchor" and self._anchor is None:
            from posthoc_repair.anchor_forecast import AnchorForecaster
            self._anchor = AnchorForecaster(self._anchor_ckpt, device=self._device)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        ego_state, _ = current_input.history.current_state
        if self._t0_us is None:
            self._t0_us = int(ego_state.time_us)
        pred, raw = self._plan(current_input)
        plan = np.concatenate([np.zeros((1, 3)), pred], 0)                               # index 0 = now
        if self._enable_repair:
            plan = self._repaired(plan, current_input, raw, None)
        states = transform_predictions_to_states(plan[1:], current_input.history.ego_states,
                                                 self._future_horizon, self._step_interval)
        return InterpolatedTrajectory(trajectory=states)
