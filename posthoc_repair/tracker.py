"""nuPlan's closed-loop ego controller in torch, so the repair can score what the simulator will execute.

``TwoStageController`` = ``LQRTracker`` (``simulation/controller/tracker/lqr.py``, ``lqr_tracker.yaml``'s gains) +
``KinematicBicycleModel`` (``motion_model/kinematic_bicycle.py``), tracking one fixed plan from the current state for
``n`` steps: batched over plans, differentiable in them (float64 inside). The simulator replans every step, so this is
what the next steps execute if the next plans agree with this one. On logged FlowDrive replans the fixed-plan rollout
is within ~0.1 m of the closed loop over 1 s while the plan itself is up to 0.9 m off laterally (the tracker only
follows the plan's smoothed curvature, and its speed 1 s ahead; README, "nuPlan terms").

Per plan, as ``LQRTracker`` does every step: velocity and curvature profiles least-squares fitted to the plan's poses
(``tracker_utils.get_velocity_curvature_profiles_with_derivatives_from_poses``). Per step: the longitudinal LQR on
the profile's velocity ``horizon`` steps ahead, the lateral LQR on the lateral / heading error to the plan's pose now
and the next ``horizon`` curvatures (the 3-state product written out), the P controller when stopping; then the
bicycle model with its first-order acceleration / steering lags. The lateral LQR's horizon product is written in
closed form (``_lateral_sums``), and its curvature terms are computed for all steps before the loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TrackerConfig:
    q_longitudinal: float = 10.0
    r_longitudinal: float = 1.0
    q_lateral: tuple = (1.0, 10.0, 0.0)     # lateral error, heading error, steering angle
    r_lateral: float = 1.0
    dt: float = 0.1
    horizon: int = 10
    jerk_penalty: float = 1e-4
    curvature_rate_penalty: float = 1e-2
    initial_curvature_penalty: float = 1e-10
    stopping_gain: float = 0.5
    stopping_velocity: float = 0.2
    wheel_base: float = 3.089               # get_pacifica_parameters()
    max_steering: float = math.pi / 3       # KinematicBicycleModel defaults
    accel_tau: float = 0.2
    steering_tau: float = 0.05


def wrap(a: torch.Tensor) -> torch.Tensor:
    """``principal_value`` / ``angle_diff(a, 0, 2 pi)``: into [-pi, pi)."""
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def fit_profiles(plan: torch.Tensor, cfg: TrackerConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Velocity and curvature profiles ``[B, M]`` (``M = W - 1``) of ``plan [B, W, 3]`` sampled at ``cfg.dt``."""
    B, W, _ = plan.shape
    M, dt, dev, dty = W - 1, cfg.dt, plan.device, plan.dtype
    d = plan[:, 1:] - plan[:, :-1]
    dh, h = wrap(d[..., 2]), plan[:, :-1, 2]
    # displacement_k = dt (v_0 + dt sum_{j<k} a_j) (cos h_k, sin h_k); jerk = a_{j+1} - a_j penalised
    base = torch.cat([torch.full((M, 1), dt, device=dev, dtype=dty),
                      dt * dt * torch.ones(M, M - 1, device=dev, dtype=dty).tril(-1)], 1)
    A = torch.stack([torch.cos(h)[..., None] * base, torch.sin(h)[..., None] * base], 2).reshape(B, 2 * M, M)
    D = torch.zeros(M - 2, M, device=dev, dtype=dty)
    i = torch.arange(M - 2, device=dev)
    D[i, i + 1], D[i, i + 2] = -1.0, 1.0
    x = torch.linalg.solve(A.transpose(1, 2) @ A + cfg.jerk_penalty * (D.T @ D),
                           (A.transpose(1, 2) @ d[..., :2].reshape(B, 2 * M, 1)))[..., 0]
    vel = x[:, :1] + torch.cat([torch.zeros_like(x[:, :1]), torch.cumsum(x[:, 1:] * dt, 1)], 1)        # [B, M]
    # heading displacement_i = dt v_i (k_0 + dt sum_{1<=j<=i} rate_j)
    tri = torch.ones(M, M, device=dev, dtype=dty).tril()
    tri[:, 0] = 0.0
    A2 = vel[:, :, None] * (dt * dt * tri + dt * torch.eye(M, device=dev, dtype=dty)[0])
    Q2 = torch.full((M,), cfg.curvature_rate_penalty, device=dev, dtype=dty)
    Q2[0] = cfg.initial_curvature_penalty
    k = torch.linalg.solve(A2.transpose(1, 2) @ A2 + torch.diag(Q2), A2.transpose(1, 2) @ dh[..., None])[..., 0]
    curv = k[:, :1] + torch.cat([torch.zeros_like(k[:, :1]), torch.cumsum(k[:, 1:] * dt, 1)], 1)       # [B, M]
    return vel, curv


def _lateral_sums(H: int) -> tuple[dict, np.ndarray]:
    """Constants of the lateral LQR's horizon product. Over the horizon the speed is ``v + alpha j`` (``j = 0..H-1``,
    ``alpha`` = the longitudinal command times dt), so every sum ``LQRTracker._lateral_lqr_controller`` accumulates is a
    quadratic in ``(v, alpha)`` with these coefficients, and the curvature terms are dot products of the horizon's
    curvatures with the rows of ``w``."""
    j = np.arange(H, dtype=np.float64)
    T = np.array([j[i + 1:].sum() for i in range(H)])                  # sum_{i > j} i
    U = np.array([j[m:].sum() for m in range(H)])                      # sum_{l >= m} l
    r = H - 1 - j
    q = r * j + T
    tail = lambda f: np.array([f[m:].sum() for m in range(H)])[1:].sum()  # noqa: E731  sum_{m>=1} sum_{l>=m} f_l
    c = {"S1": j.sum(), "C1": r.sum(), "C2": q.sum(), "C3": (T * j).sum(), "D1": (H - j[1:]).sum(), "D2": U[1:].sum(),
         "F1": tail(r), "F2": tail(q), "F3": tail(T * j)}
    return c, np.stack([np.ones(H), j, r, q, j * T])


def track(plan: torch.Tensor, v0: torch.Tensor, steer0: torch.Tensor, accel0: torch.Tensor, n: int,
          cfg: TrackerConfig = TrackerConfig()) -> tuple[torch.Tensor, torch.Tensor]:
    """Rear-axle poses ``[B, n, 3]`` and speeds ``[B, n]`` at steps ``1..n`` when the simulator tracks ``plan [B, W, 3]``
    (index 0 = the current pose, ``cfg.dt`` apart) from longitudinal velocity ``v0``, tire steering angle ``steer0`` and
    longitudinal acceleration ``accel0`` (``[B]``), in the plan's frame and dtype."""
    p = plan.double()
    vel, curv = fit_profiles(p, cfg)
    B, M, H, dt, L = p.shape[0], vel.shape[1], cfg.horizon, cfg.dt, cfg.wheel_base
    c, w = _lateral_sums(H)
    # everything that depends on the plan only, for all n steps at once
    idx = (torch.arange(n, device=p.device)[:, None] + torch.arange(H, device=p.device)).clamp(max=M - 1)
    K = curv[:, idx] @ torch.as_tensor(w.T, device=p.device, dtype=p.dtype)                     # [B, n, 5]
    ref = [t.unbind(1) for t in (p[:, :n, 0], p[:, :n, 1], torch.sin(p[:, :n, 2]), torch.cos(p[:, :n, 2]), p[:, :n, 2],
                                 vel[:, (torch.arange(n, device=p.device) + H).clamp(max=M - 1)])]
    K = K.unbind(1)
    x, y, h = p[:, 0, 0], p[:, 0, 1], p[:, 0, 2]
    v, steer, acc = v0.double(), steer0.double(), accel0.double()
    q0, q1, q2 = cfg.q_lateral
    b_lon = H * dt
    gain = cfg.q_longitudinal * b_lon / (cfg.q_longitudinal * b_lon ** 2 + cfg.r_longitudinal)
    Bm2 = H * dt
    poses, speeds = [], []
    for k in range(n):
        xr, yr, sr, cr, hr, ref_v = (t[k] for t in ref)
        lat = -(x - xr) * sr + (y - yr) * cr
        head = wrap(h - hr)
        a_lqr = gain * (ref_v - v)
        al = a_lqr * dt
        vv, va, aa = v * v, v * al, al * al
        # x_H = S_{H-1} .. S_0 x_0 + B u + g, S_j = I + a_j E01 + b_j E12, a_j = dt v_j, b_j = dt v_j / L, v_j = v + al j
        A01 = dt * (H * v + al * c["S1"])
        A12 = A01 / L
        A02 = dt * dt / L * (vv * c["C1"] + va * c["C2"] + aa * c["C3"])
        Bm0 = dt ** 3 / L * (vv * c["F1"] + va * c["F2"] + aa * c["F3"])
        Bm1 = dt * dt / L * (v * c["D1"] + al * c["D2"])
        Kk = K[k]
        g1 = -dt * (v * Kk[:, 0] + al * Kk[:, 1])
        g0 = -dt * dt * (vv * Kk[:, 2] + va * Kk[:, 3] + aa * Kk[:, 4])
        e0 = lat + A01 * head + A02 * steer + g0
        e1 = wrap(head + A12 * steer + g1)
        e2 = wrap(steer)
        u = -(q0 * Bm0 * e0 + q1 * Bm1 * e1 + q2 * Bm2 * e2) / (q0 * Bm0 ** 2 + q1 * Bm1 ** 2 + q2 * Bm2 ** 2
                                                                + cfg.r_lateral)
        stop = (ref_v <= cfg.stopping_velocity) & (v <= cfg.stopping_velocity)
        accel_cmd = torch.where(stop, -cfg.stopping_gain * (v - ref_v), a_lqr)
        rate_cmd = torch.where(stop, torch.zeros_like(u), u)
        # KinematicBicycleModel.propagate_state: lagged commands, then Euler with the current speed / steering
        acc = acc + dt / (dt + cfg.accel_tau) * (accel_cmd - acc)
        steer_next = steer + dt / (dt + cfg.steering_tau) * (dt * rate_cmd)
        x = x + v * torch.cos(h) * dt
        y = y + v * torch.sin(h) * dt
        h = wrap(h + v * torch.tan(steer) / L * dt)
        v = v + acc * dt
        steer = steer_next.clamp(-cfg.max_steering, cfg.max_steering)
        poses.append(torch.stack([x, y, h], -1))
        speeds.append(v.abs())
    return torch.stack(poses, 1).to(plan.dtype), torch.stack(speeds, 1).to(plan.dtype)
