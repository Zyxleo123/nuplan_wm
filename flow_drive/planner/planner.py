"""FlowDrive as a nuPlan planner: the flow model's plan, executed as it comes (upstream ``post_mode: 0``).

Upstream ``FlowDrivePlannerWrapper`` without its FlowDrive* branch (``post_mode: 1``: moderated lateral guidance over
30 candidates, smoothing, speed bounding, PDM scoring and PDM's emergency brake), so it needs no tuplan_garage.
Each replan seeds python, numpy and torch with ``seed + the planner's call count`` and turns cudnn's autotuner
off, as upstream's ``set_seed(520 + iteration)``; ``seed: null`` leaves the generators alone. The 40 poses (0.1 s,
ego frame) go to the simulator through nuPlan's ``transform_predictions_to_states``.
"""

from __future__ import annotations

import random
import warnings
from typing import Optional, Type

import numpy as np
import torch

warnings.filterwarnings("ignore")

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from flow_drive.data_process.data_processor import DataProcessor
from flow_drive.model.flow_drive_planner import FlowDrivePlanner, load_params


def set_seed(s: int) -> None:
    """Upstream ``flow_drive.utils.train_utils.set_seed``."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FlowDrivePlannerWrapper(AbstractPlanner):
    def __init__(self, ckpt_path: str, device: str = "cuda", seed: Optional[int] = 520,
                 config_file: Optional[str] = None) -> None:
        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
        self._ckpt_path = ckpt_path
        self._device = device
        self._seed = seed
        self._params = load_params(config_file)
        self._future_horizon = self._params.data_processing.future_time_horizon  # [s]
        self._step_interval = 0.1  # [s]
        self._data_processor = DataProcessor(self._params.data_processing)
        self._model = None
        self._iteration = 0

    def name(self) -> str:
        return "flow_drive"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._iteration = 0
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        if self._model is None:
            self._model = FlowDrivePlanner(self._params, self._ckpt_path, self._device)
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> dict:
        return self._data_processor.observation_adapter(
            planner_input.history, list(planner_input.traffic_light_data), self._map_api,
            self._route_roadblock_ids, self._device)

    def _plan(self, current_input: PlannerInput) -> tuple[np.ndarray, dict]:
        """This replan's flow sample ``[T, (x, y, yaw)]`` (rear axle, ego frame) and the raw model inputs."""
        if self._seed is not None:
            set_seed(self._seed + self._iteration)
        self._iteration += 1
        inputs = self.planner_input_to_model_inputs(current_input)
        return self._model(inputs)[0].detach().cpu().numpy().astype(np.float64), inputs

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        plan, _ = self._plan(current_input)
        states = transform_predictions_to_states(plan, current_input.history.ego_states,
                                                 self._future_horizon, self._step_interval)
        return InterpolatedTrajectory(trajectory=states)
