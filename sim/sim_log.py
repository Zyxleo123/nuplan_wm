"""nuPlan's simulation-log callback, writing the log without the planner.

nuPlan's ``SimulationLog`` pickles the planner it is handed into every scenario's log, loaded weights and all, so a
log weighs what the checkpoints do: ~22 MB with FlowDrive, ~66 MB once AnchorWM rides along. What is read back
(tune/*.py, nuBoard) is the history and the scenario, so the log here keeps only the planner's name.
run_chunks.sh selects it with ``callback.simulation_log_callback._target_``.
"""

from __future__ import annotations

from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup


class PlannerName:
    """The log's stand-in for the planner: just ``name()``, which also names the log's folder."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class PlannerlessSimulationLogCallback(SimulationLogCallback):
    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        super().on_simulation_end(setup, PlannerName(planner.name()), history)
