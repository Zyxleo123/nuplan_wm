"""The o_am post-hoc repair (yixiz/waymax_rs ``rl/diffusion_ft/lateral_opt.py``, ``planner/posthoc_repair.py``)
in torch, on nuPlan's ego frame.

One loss per plan over a lateral offset ``d_1..d_n`` and an along-path offset ``s_1..s_n`` on the next
``window`` steps::

    L(d, s) = (1/n) sum_t [ sum_corners edge_barrier + sum_agents w * agent_barrier ](edit(plan; d, s))_t
            + 0.5 x^T H x

``H`` is the comfort quadratic: acceleration and jerk of the lateral edit (held after the window) and of the
along-path edit (its velocity held after the window), plus ``w_lon_prox`` on ``s`` itself ("stay on the
policy's timing"). ``L`` is minimized by ``iters`` steps of ``H^-1``-preconditioned descent (CHOMP) under a
geometrically decaying per-row trust region, and the lowest-loss iterate is returned. The barriers are
hinge^2, so a plan already clear of every margin has zero loss and zero gradient at the origin and comes back
unchanged: one objective decides whether and where the plan moves, with no rule on top.

What differs from the WOMD version:

* edges are the boundary of nuPlan's DRIVABLE_AREA (``anchor_wm.geometry.edge_signed_distance``), and the
  clearance is measured from where ``drivable_area_compliance`` fires (a corner ``edge_tolerance_m`` = 0.3 m
  outside), as the WOMD clearance is measured from where Waymax's ``offroad`` fires;
* the plan is nuPlan's rear axle; the ego box is the Pacifica's, centred ``EGO_REAR_TO_CENTER`` ahead;
* plans are ``[B, W, 3]`` (x, y, yaw; index 0 = the current pose) on the simulator's 0.1 s step.

The margins and the optimizer constants are o_am's (WOMD-calibrated: edge 0.3 m = expert p01, agent 0.75 m
= expert p05); they have not been re-measured on nuPlan.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch

from anchor_wm.config import EGO_LENGTH, EGO_REAR_TO_CENTER, EGO_WIDTH
from anchor_wm.geometry import box_corners, edge_signed_distance, wrap_angle


@dataclass(frozen=True)
class RepairConfig:
    """o_am: ``LateralOptConfig`` defaults + ``lon=1,w_lon_prox=0.05,lon_back_m=1000``, ``SafetyLossConfig``
    margins under ``barrier_shape=hinge2``, the ES runs' 20-step window."""

    window: int = 20                # steps (0.1 s) the loss scores
    dt: float = 0.1
    iters: int = 32
    step_m: float = 0.25            # trust region per iteration, decaying by `decay`: reach 2.9 m
    decay: float = 0.92
    damping: float = 0.5
    w_acc: float = 0.02
    w_jerk: float = 0.02
    w_prox: float = 0.0
    a_ref: float = 1.0              # m/s^2
    j_ref: float = 10.0             # m/s^3
    d_ref: float = 1.0              # m
    w_lon_acc: float = 0.02
    w_lon_jerk: float = 0.02
    w_lon_prox: float = 0.05
    lon_back_m: float = 1000.0      # how far the edit may fall behind the plan (o_am: unbounded)
    edge_margin_m: float = 0.3
    edge_tau_m: float = 0.15
    edge_weight: float = 1.0
    edge_tolerance_m: float = 0.3   # nuplan drivable_area_compliance: max_violation_threshold
    agent_margin_m: float = 0.75
    agent_tau_m: float = 0.25
    agent_weight: float = 1.0

    @classmethod
    def from_dict(cls, value: dict | None) -> "RepairConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(value or {}) - names
        if unknown:
            raise ValueError(f"unknown repair keys {sorted(unknown)} (have {sorted(names)})")
        cfg = cls(**(value or {}))
        if cfg.w_acc <= 0 and cfg.w_jerk <= 0:
            raise ValueError("need w_acc > 0 or w_jerk > 0 (the comfort Hessian is the preconditioner)")
        if cfg.w_lon_acc <= 0 and cfg.w_lon_jerk <= 0:
            raise ValueError("need w_lon_acc > 0 or w_lon_jerk > 0")
        return cfg


# ---- the comfort quadratic (numpy, verbatim from lateral_opt) -------------------------------------------

def comfort_operators(n: int, dt: float, hold: str = "value") -> tuple[np.ndarray, np.ndarray]:
    """``(Ka, Kj)``: acceleration and jerk of the edit as linear maps of ``x_1..x_n``. The profile is padded
    with three zeros before (on the plan, not drifting, not accelerating) and two steps after: the last value
    held (``value``, lateral) or its last slope continued (``velocity``, along-path)."""
    E = np.zeros((n + 5, n))                    # steps -2 .. n+2
    E[3:3 + n] = np.eye(n)
    if hold == "value":
        E[3 + n:, n - 1] = 1.0
    elif hold == "velocity":
        for k in (1, 2):                        # x_{n+k} = x_n + k (x_n - x_{n-1}); x_0 = 0 when n == 1
            E[2 + n + k, n - 1] += 1.0 + k
            if n >= 2:
                E[2 + n + k, n - 2] -= k
    else:
        raise ValueError(f"hold {hold!r}: value | velocity")
    D2 = np.zeros((n + 3, n + 5))
    for i in range(n + 3):
        D2[i, i:i + 3] = (1.0, -2.0, 1.0)
    Ka = D2 @ E / dt ** 2
    D1 = np.zeros((n + 2, n + 3))
    for i in range(n + 2):
        D1[i, i:i + 2] = (-1.0, 1.0)
    return Ka, D1 @ Ka / dt


def edit_hessian(n: int, cfg: RepairConfig) -> np.ndarray:
    """Hessian ``[2n, 2n]`` of the comfort quadratic, block-diagonal (lateral, along-path)."""
    def block(hold, w_acc, w_jerk, w_prox):
        Ka, Kj = comfort_operators(n, cfg.dt, hold)
        H = (w_acc / cfg.a_ref ** 2) * Ka.T @ Ka + (w_jerk / cfg.j_ref ** 2) * Kj.T @ Kj
        return (2.0 / n) * (H + (w_prox / cfg.d_ref ** 2) * np.eye(n))

    out = np.zeros((2 * n, 2 * n))
    out[:n, :n] = block("value", cfg.w_acc, cfg.w_jerk, cfg.w_prox)
    out[n:, n:] = block("velocity", cfg.w_lon_acc, cfg.w_lon_jerk, cfg.w_lon_prox)
    return out


# ---- the edit ---------------------------------------------------------------------------------------------

def _unwrap(yaw: torch.Tensor) -> torch.Tensor:
    d = torch.diff(yaw, dim=1)
    d = torch.atan2(torch.sin(d), torch.cos(d))
    return torch.cat([yaw[:, :1], yaw[:, :1] + torch.cumsum(d, 1)], 1)


def _interp(q: torch.Tensor, S: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Row-wise ``np.interp(q, S, f)`` for ``q`` inside ``[S_0, S_-1]``, differentiable in ``q``."""
    i = (torch.searchsorted(S, q.detach().contiguous(), right=True) - 1).clamp(0, S.shape[1] - 2)
    s0, s1 = S.gather(1, i), S.gather(1, i + 1)
    f0, f1 = f.gather(1, i), f.gather(1, i + 1)
    return f0 + (q - s0) / (s1 - s0) * (f1 - f0)


def lateral_offset(plan: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Move each pose ``d [B, W]`` metres along its left normal; the heading gains the offset path's slope."""
    yaw = plan[..., 2]
    xy = plan[..., :2] + d[..., None] * torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1)
    # |step| with a zero gradient at a zero-length step (a stopped plan, or a held tail under lon).
    step = torch.diff(plan[..., :2], dim=1)
    moved = (step != 0).any(-1)
    ds = torch.where(moved, torch.where(moved[..., None], step, torch.ones_like(step)).norm(dim=-1),
                     torch.zeros_like(step[..., 0]))
    slope = torch.atan2(torch.diff(d, dim=1), ds.clamp(min=0.05))
    dyaw = torch.cat([torch.zeros_like(slope[:, :1]), slope], 1)
    return torch.stack([xy[..., 0], xy[..., 1], wrap_angle(yaw + dyaw)], -1)


def path_retime(plan: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Move each pose ``s [B, W]`` metres along the plan's own path (clamped to the path, never reversing);
    the heading is the path's there, with its gradient stopped."""
    xy = plan[..., :2]
    seg = torch.diff(xy, dim=1).norm(dim=-1)
    # 1 mm per step keeps the knots strictly increasing through a stop.
    S = torch.cat([torch.zeros_like(seg[:, :1]), torch.cumsum(seg + 1e-3, 1)], 1)
    q = torch.cummax(torch.minimum((S + s).clamp(min=0.0), S[:, -1:]), dim=1).values
    yaw = _interp(q, S, _unwrap(plan[..., 2])).detach()
    return torch.stack([_interp(q, S, xy[..., 0]), _interp(q, S, xy[..., 1]), wrap_angle(yaw)], -1)


def expand_offset(x: torch.Tensor, W: int) -> torch.Tensor:
    """``[B, n]`` -> ``[B, W]``: 0 at the anchor, the window, then held."""
    return torch.cat([torch.zeros_like(x[:, :1]), x, x[:, -1:].expand(-1, W - 1 - x.shape[1])], 1)


def expand_retime(s: torch.Tensor, W: int) -> torch.Tensor:
    """``[B, n]`` -> ``[B, W]``: 0 at the anchor, the window, then continued at its last slope."""
    full = torch.cat([torch.zeros_like(s[:, :1]), s], 1)
    k = torch.arange(1, W - s.shape[1], device=s.device, dtype=s.dtype)[None]
    return torch.cat([full, full[:, -1:] + k * (full[:, -1:] - full[:, -2:-1])], 1)


def edit_plan(plan: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
    """The edited plan for ``x = [d_1..d_n, s_1..s_n]``: retimed first, then offset along the retimed normals."""
    W = plan.shape[1]
    return lateral_offset(path_retime(plan, expand_retime(x[:, n:], W)), expand_offset(x[:, :n], W))


# ---- the barrier and the geometry -------------------------------------------------------------------------

def barrier(c: torch.Tensor, margin: float, tau: float) -> torch.Tensor:
    """hinge^2: 0 beyond the margin, ``z^2 / 2`` on the knee, ``z - 1/2`` past it (``z = (margin - c) / tau``)."""
    z = (margin - c) / tau
    zq = z.clamp(0.0, 1.0)
    return tau * torch.where(z < 1.0, 0.5 * zq * zq, z - 0.5)


def _point_segment_distance(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = b - a
    t = (((p - a) * d).sum(-1) / (d * d).sum(-1).clamp(min=1e-12)).clamp(0.0, 1.0)
    return ((p - (a + t[..., None] * d)).square().sum(-1) + 1e-12).sqrt()


def box_signed_gap(ca: torch.Tensor, ya: torch.Tensor, cb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """Exact signed gap of oriented boxes (corners ``[..., 4, 2]``, yaws ``[...]``, broadcasting): the separation
    when disjoint, minus the penetration depth when overlapping (safety_loss.box_box_signed_gap)."""
    ca, cb = torch.broadcast_tensors(ca, cb)
    ya, yb = torch.broadcast_tensors(ya, yb)
    axes = torch.stack([
        torch.stack([torch.cos(ya), torch.sin(ya)], -1), torch.stack([-torch.sin(ya), torch.cos(ya)], -1),
        torch.stack([torch.cos(yb), torch.sin(yb)], -1), torch.stack([-torch.sin(yb), torch.cos(yb)], -1),
    ], -2)                                                                     # [..., 4 axes, 2]
    pa = (ca[..., :, None, :] * axes[..., None, :, :]).sum(-1)                 # [..., 4 corners, 4 axes]
    pb = (cb[..., :, None, :] * axes[..., None, :, :]).sum(-1)
    pen = torch.minimum(pa.amax(-2) - pb.amin(-2), pb.amax(-2) - pa.amin(-2)).amin(-1)

    def vertex_edge(v, poly):
        return _point_segment_distance(v[..., :, None, :], poly[..., None, :, :],
                                       torch.roll(poly, -1, dims=-2)[..., None, :, :]).flatten(-2).amin(-1)
    dist = torch.minimum(vertex_edge(ca, cb), vertex_edge(cb, ca))
    return torch.where(pen > 0.0, -pen, dist)


def ego_corners(window: torch.Tensor) -> torch.Tensor:
    """Ego box corners ``[B, n, 4, 2]`` from rear-axle poses ``[B, n, 3]``."""
    yaw = window[..., 2]
    centre = window[..., :2] + EGO_REAR_TO_CENTER * torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    lw = torch.tensor([EGO_LENGTH, EGO_WIDTH], device=window.device, dtype=window.dtype).expand(*yaw.shape, 2)
    return box_corners(centre, yaw, lw)


def plan_safety(cfg: RepairConfig, window: torch.Tensor, edges: dict, agents: dict) -> dict:
    """Barriers and hard clearances of the scored steps ``window [B, n, 3]``.

    ``edges``: the drivable-area boundary, ``anchor_wm.geometry.edge_signed_distance``'s dict (``[B, E, ...]``).
    ``agents``: ``box [B, n, M, 5]`` (x, y, length, width, yaw: box centre), ``valid [B, n, M]``,
    ``weight [B, n, M]`` (a mode's probability; 1 for a single forecast).
    Returns per step ``[B, n]``: ``loss``, ``edge_loss``, ``agent_loss``, ``c_edge`` (min over corners, metres
    inside the metric's limit), ``c_agent`` (min box gap over valid agents; inf without any).
    """
    B, n = window.shape[:2]
    corners = ego_corners(window)                                              # [B, n, 4, 2]
    c_edge = cfg.edge_tolerance_m - edge_signed_distance(corners.reshape(B, n * 4, 2), edges).reshape(B, n, 4)
    edge = cfg.edge_weight * barrier(c_edge, cfg.edge_margin_m, cfg.edge_tau_m).sum(-1)

    box, valid = agents["box"], agents["valid"].bool()
    if box.shape[2]:
        oc = box_corners(box[..., :2], box[..., 4], box[..., 2:4])           # [B, n, M, 4, 2]
        gap = box_signed_gap(corners[:, :, None], window[:, :, None, 2], oc, box[..., 4])   # [B, n, M]
        pair = barrier(gap, cfg.agent_margin_m, cfg.agent_tau_m) * agents["weight"]
        agent = cfg.agent_weight * torch.where(valid, pair, torch.zeros_like(pair)).sum(-1)
        c_agent = torch.where(valid, gap, torch.full_like(gap, float("inf"))).amin(-1)
    else:
        agent = torch.zeros_like(edge)
        c_agent = torch.full_like(edge, float("inf"))
    return {"loss": edge + agent, "edge_loss": edge, "agent_loss": agent,
            "c_edge": c_edge.amin(-1), "c_agent": c_agent}


# ---- the optimizer ----------------------------------------------------------------------------------------

class PosthocRepair:
    """``repair(plan [B, W, 3], edges, agents) -> (plan [B, W, 3], info)``; see the module docstring."""

    def __init__(self, cfg: RepairConfig | None = None) -> None:
        self.cfg = cfg or RepairConfig()
        self._hessians = {}

    def _hessian(self, n: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (n, str(device), dtype)
        if key not in self._hessians:
            H = edit_hessian(n, self.cfg)
            self._hessians[key] = (torch.as_tensor(H, device=device, dtype=dtype),
                                   torch.as_tensor(np.linalg.inv(H), device=device, dtype=dtype))
        return self._hessians[key]

    def reach_m(self) -> float:
        """How far any edit coordinate can move from 0: the trust region's whole schedule."""
        c = self.cfg
        return c.step_m * (1.0 - c.decay ** c.iters) / (1.0 - c.decay) if c.decay != 1.0 else c.step_m * c.iters

    def prune_agents(self, plan: torch.Tensor, n: int, agents: dict) -> dict:
        """Drop the boxes no reachable edit can bring inside the agent margin, which leaves the loss unchanged.

        Each of ``d_t``, ``s_t`` moves at most ``reach_m`` (the trust region's schedule), so the rear axle moves at
        most ``sqrt(2) reach`` and the box centre ``EGO_REAR_TO_CENTER`` ahead of it at most another
        ``2 EGO_REAR_TO_CENTER``; a box whose centre stays farther than that plus both half-diagonals plus the
        margin from the plan's ego centre at every scored step has a zero barrier for every reachable ``x``."""
        box = agents["box"]
        if not box.shape[2]:
            return agents
        w = plan[:, 1:n + 1]
        centre = w[..., :2] + EGO_REAR_TO_CENTER * torch.stack([torch.cos(w[..., 2]), torch.sin(w[..., 2])], -1)
        bound = (2.0 ** 0.5 * self.reach_m() + 2.0 * EGO_REAR_TO_CENTER + 0.5 * float(np.hypot(EGO_LENGTH, EGO_WIDTH))
                 + 0.5 * box[..., 2:4].norm(dim=-1) + self.cfg.agent_margin_m + 0.1)                 # [B, n, M]
        near = ((box[..., :2] - centre[:, :, None]).norm(dim=-1) <= bound) & agents["valid"].bool()
        keep = near.any(1).any(0)                                                                  # [M]
        return {k: v[:, :, keep] for k, v in agents.items()}

    def loss(self, plan, x, n, edges, agents, H):
        terms = plan_safety(self.cfg, edit_plan(plan, x, n)[:, 1:n + 1], edges, agents)
        return terms["loss"].sum(-1) / n + 0.5 * torch.einsum("bi,ij,bj->b", x, H, x), terms

    def __call__(self, plan: torch.Tensor, edges: dict, agents: dict) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B, W, _ = plan.shape
        n = int(min(cfg.window, W - 1))
        H, Hinv = self._hessian(n, plan.device, plan.dtype)
        agents = self.prune_agents(plan, n, {k: v[:, :n] for k, v in agents.items()})

        def project(x):          # the along-path half may not fall more than lon_back_m behind the plan
            return torch.cat([x[:, :n], x[:, n:].clamp(min=-cfg.lon_back_m)], 1)

        x0 = torch.zeros(B, 2 * n, device=plan.device, dtype=plan.dtype)
        with torch.no_grad():
            l0, t0 = self.loss(plan, x0, n, edges, agents, H)
        info = {"n": n, "loss0": l0, "c_edge0": t0["c_edge"].amin(-1), "c_agent0": t0["c_agent"].amin(-1)}
        if not bool((l0 > 0).any()):             # clear of every margin: zero loss, zero gradient, unchanged
            return plan, {**info, "loss": l0, "moved": torch.zeros(B, dtype=torch.bool, device=plan.device),
                          "shift_m": torch.zeros_like(l0), "c_edge": info["c_edge0"], "c_agent": info["c_agent0"]}

        x, best_x, best_l = x0, x0, l0
        for k in range(int(cfg.iters)):
            xr = x.detach().requires_grad_(True)
            l_b, _ = self.loss(plan, xr, n, edges, agents, H)
            (g,) = torch.autograd.grad(l_b.sum(), xr)
            l_b = l_b.detach()
            better = l_b < best_l
            best_x = torch.where(better[:, None], x, best_x)
            best_l = torch.where(better, l_b, best_l)
            step = cfg.damping * (g @ Hinv)                                    # H is symmetric
            cap = cfg.step_m * cfg.decay ** k
            step = step * (cap / step.abs().amax(1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            x = project(x - step)
        with torch.no_grad():
            l_last, _ = self.loss(plan, x, n, edges, agents, H)
            better = l_last < best_l
            best_x = torch.where(better[:, None], x, best_x)
            best_l = torch.where(better, l_last, best_l)
            # A non-finite iterate (it should not happen; the geometry is NaN-safe) keeps the plan.
            ok = torch.isfinite(best_x).all(1) & torch.isfinite(best_l)
            best_x = torch.where(ok[:, None], best_x, torch.zeros_like(best_x))
            moved = (best_x != 0).any(1)
            out = torch.where(moved[:, None, None], edit_plan(plan, best_x, n), plan)
            _, t1 = self.loss(plan, best_x, n, edges, agents, H)
        shift = (out[..., :2] - plan[..., :2]).norm(dim=-1).amax(1)
        return out, {**info, "loss": torch.where(ok, best_l, l0), "moved": moved, "nonfinite": ~ok,
                     "shift_m": shift, "c_edge": t1["c_edge"].amin(-1), "c_agent": t1["c_agent"].amin(-1)}
