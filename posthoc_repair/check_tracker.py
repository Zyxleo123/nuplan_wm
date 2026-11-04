"""``posthoc_repair.tracker`` against nuPlan's own controller (no data needed, CPU).

Random plans (curving, speeding up, braking to a stop, standing still) from random speeds / steering angles /
accelerations go through nuPlan's ``transform_predictions_to_states`` -> ``InterpolatedTrajectory``, then
``LQRTracker`` (``lqr_tracker.yaml``) + ``KinematicBicycleModel`` step by step on that fixed trajectory, exactly as
``TwoStageController`` steps it; the torch rollout of the same plan must give the same poses and speeds. Also: the
rollout is batch-consistent, and its gradient w.r.t. the plan is finite and non-zero.

    PYTHONPATH=. python -m posthoc_repair.check_tracker
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from posthoc_repair.tracker import track

T0 = 1_600_000_000_000_000


def nuplan_rollout(plan: np.ndarray, v0: float, steer0: float, a0: float, n: int, pose0=(3.0, -2.0, 0.4)):
    """The plan (ego frame, index 0 = now) tracked by nuPlan's controller for n steps; poses back in the ego frame."""
    vehicle = get_pacifica_parameters()
    x0, y0, h0 = pose0

    def ego(t_us, x, y, h, v, a, s):
        return EgoState.build_from_rear_axle(StateSE2(x, y, h), StateVector2D(v, 0.0), StateVector2D(a, 0.0), s,
                                             TimePoint(t_us), vehicle, True)
    prev = ego(T0 - 100_000, x0 - 0.1 * v0 * np.cos(h0), y0 - 0.1 * v0 * np.sin(h0), h0, v0, a0, steer0)
    cur = ego(T0, x0, y0, h0, v0, a0, steer0)
    states = transform_predictions_to_states(plan[1:].astype(np.float32), deque([prev, cur]), 0.1 * (len(plan) - 1), 0.1)
    traj = InterpolatedTrajectory(states)
    tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], r_lateral=[1.0],
                         discretization_time=0.1, tracking_horizon=10, jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
                         stopping_proportional_gain=0.5, stopping_velocity=0.2)
    model = KinematicBicycleModel(vehicle)
    c, s = np.cos(h0), np.sin(h0)
    out, state = [], cur
    for k in range(n):
        it, nxt = SimulationIteration(TimePoint(T0 + k * 100_000), k), SimulationIteration(TimePoint(T0 + (k + 1) * 100_000), k + 1)
        state = model.propagate_state(state, tracker.track_trajectory(it, nxt, state, traj), nxt.time_point - it.time_point)
        r = state.rear_axle
        out.append([c * (r.x - x0) + s * (r.y - y0), -s * (r.x - x0) + c * (r.y - y0), r.heading - h0,
                    state.dynamic_car_state.speed])
    out = np.array(out)
    out[:, 2] = np.arctan2(np.sin(out[:, 2]), np.cos(out[:, 2]))
    return out


def random_plan(rng, W: int):
    kind = rng.choice(["curve", "speed_up", "brake", "still", "swerve"])
    t = 0.1 * np.arange(W)
    v0 = 0.0 if kind == "still" else rng.uniform(0.5, 15.0)
    if kind == "brake":
        a = -rng.uniform(1.0, 5.0)
        s = np.where(t < v0 / -a, v0 * t + 0.5 * a * t * t, v0 * v0 / (-2 * a))
    elif kind == "speed_up":
        s = v0 * t + 0.5 * rng.uniform(0.3, 2.5) * t * t
    else:
        s = v0 * t
    kappa = rng.uniform(-0.08, 0.08) if kind in ("curve", "speed_up", "brake") else 0.0
    yaw = kappa * s
    x = np.concatenate([[0.0], np.cumsum(np.diff(s) * np.cos(yaw[:-1] + 0.5 * np.diff(yaw)))])
    y = np.concatenate([[0.0], np.cumsum(np.diff(s) * np.sin(yaw[:-1] + 0.5 * np.diff(yaw)))])
    if kind == "swerve":
        y = y + rng.uniform(-1.5, 1.5) * np.clip(t / 1.5, 0, 1) ** 2
        yaw = np.concatenate([[0.0], np.arctan2(np.diff(y), np.maximum(np.diff(x), 1e-6))])
    steer0 = rng.uniform(-0.15, 0.15)
    return kind, np.stack([x, y, yaw], -1), v0 * rng.uniform(0.8, 1.2), steer0, rng.uniform(-1.5, 1.5)


def main() -> None:
    fails = []

    def gate(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            fails.append(name)

    rng = np.random.default_rng(0)
    n = 20
    worst = {"pos": 0.0, "yaw": 0.0, "speed": 0.0}
    batch = []
    for i in range(30):
        W = int(rng.choice([41, 81]))
        kind, plan, v0, steer0, a0 = random_plan(rng, W)
        ref = nuplan_rollout(plan, v0, steer0, a0, n)
        poses, speed = track(torch.as_tensor(plan)[None], torch.tensor([v0]), torch.tensor([steer0]), torch.tensor([a0]), n)
        poses, speed = poses[0].numpy(), speed[0].numpy()
        e = {"pos": float(np.abs(poses[:, :2] - ref[:, :2]).max()),
             "yaw": float(np.abs(np.arctan2(np.sin(poses[:, 2] - ref[:, 2]), np.cos(poses[:, 2] - ref[:, 2]))).max()),
             "speed": float(np.abs(speed - ref[:, 3]).max())}
        worst = {k: max(worst[k], e[k]) for k in worst}
        if i < 8 or max(e.values()) > 1e-3:
            print(f"   {kind:8s} W={W} v0={v0:5.2f} steer0={steer0:+.2f} a0={a0:+.2f}: end ({ref[-1, 0]:6.2f}, {ref[-1, 1]:+5.2f}) "
                  f"|d pos| {e['pos']:.1e} |d yaw| {e['yaw']:.1e} |d v| {e['speed']:.1e}")
        if W == 81:
            batch.append((plan, v0, steer0, a0, poses))
    # float32 plans in (the planner's dtype): the tracker works in float64 inside
    gate("matches nuPlan's LQRTracker + KinematicBicycleModel", worst["pos"] < 1e-3 and worst["yaw"] < 1e-4 and worst["speed"] < 1e-3,
         f"30 plans x {n} steps: max |pos| {worst['pos']:.2e} m, |yaw| {worst['yaw']:.2e} rad, |v| {worst['speed']:.2e} m/s")

    P = torch.as_tensor(np.stack([b[0] for b in batch]))
    poses, _ = track(P, *(torch.tensor([b[k] for b in batch]) for k in (1, 2, 3)), n)
    d = max(float(np.abs(poses[i].numpy() - b[4]).max()) for i, b in enumerate(batch))
    gate("batched == one at a time", d < 1e-9, f"{len(batch)} plans, max |diff| {d:.1e}")

    plan = torch.as_tensor(batch[0][0], dtype=torch.float32)[None].requires_grad_(True)
    poses, speed = track(plan, torch.tensor([batch[0][1]]), torch.tensor([batch[0][2]]), torch.tensor([batch[0][3]]), n)
    (g,) = torch.autograd.grad(poses[..., 1].sum() + speed.sum(), plan)
    gate("differentiable in the plan", bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0,
         f"|grad| sum {float(g.abs().sum()):.3g}")
    print("ALL PASS" if not fails else f"FAILED: {fails}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
