"""Self-generated ego candidates and their online labels (log replay), all on the GPU.

Candidates per scene (C = K + 1 + 16), rear-axle positions in the ego frame:
  * the K decoded ego anchors (what the model would propose itself),
  * the logged ego future,
  * 16 smooth perturbations of it along its own path: lateral offset ramps (+-0.75, 1.5, 3 m, faded in
    over the first 5 m of travel),
    speed scaling x0.5 / 0.8 / 1.2, two constant-deceleration stops, a 2.5 m nudge toward the
    nearest agent ahead, lateral + speed combinations and two random combinations.

Labels (other objects follow the log; plan steps, plus midpoints when the step is coarser than 0.1 s),
following nuPlan's closed-loop metrics:
  overlap         an at-fault collision (nuplan no_ego_at_fault_collisions, see label_candidates) with any
                  cached object: agents, static objects, objects that appear later
  any_overlap     any contact (separating-axis test)
  clearance       min separating-axis gap over steps and objects, leaving out objects hit without fault
                  (m; < 0 = at-fault overlap)
  offroad         any step where an ego box corner is more than OFFROAD_TOL_M (0.3 m) outside the drivable
                  area (nuplan drivable_area_compliance)
  offroad_depth   max corner signed distance to the drivable-area boundary minus OFFROAD_TOL_M (m; > 0 offroad)
  progress        arc length of the end point projected on the logged ego path (m)
  route_progress  progress of the end point along the route centreline from the ego's projection (m;
                  the logged path when the scene has no route), as PDM's ego progress
  comfort         exceedance of lateral 4.5 / longitudinal 7 m/s^2 (m/s^2, < 0 = within limits)
  first_overlap / first_offroad   plan step (1..T, fractional on midpoints) of the first violation, T + 1 = none
  gt_progress / gt_route_progress  the same for the logged future
"""

from __future__ import annotations

import math

import torch

from anchor_wm.config import OFFROAD_TOL_M
from anchor_wm.geometry import (box_corners, comfort_excess, edge_signed_distance, ego_center, interp_angle,
                                interp_path, path_yaw, polyline_with_extension, project_progress, sat_gap, wrap_angle)

KIND_ANCHOR, KIND_GT, KIND_PERTURB = 0, 1, 2
_LATERAL = (-3.0, -1.5, -0.75, 0.75, 1.5, 3.0)
_SPEED = (0.5, 0.8, 1.2)
_STOP_DECEL = (2.5, 5.0)
NUM_PERTURB = len(_LATERAL) + len(_SPEED) + len(_STOP_DECEL) + 1 + 2 + 2
LATERAL_TRAVEL_M = 5.0
STOPPED_SPEED = 0.05                 # nuplan's stopped_speed_threshold (m/s)
BEHIND_COS = math.cos(math.radians(150.0))   # nuplan is_agent_behind: > 150 deg from the ego heading


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def perturb_gt(gt_xy: torch.Tensor, gt_yaw: torch.Tensor, ego_speed: torch.Tensor, nudge_sign: torch.Tensor,
               dt: float, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """gt_xy [B,T,2], gt_yaw [B,T] -> perturbations [B,P,T,2] that follow the logged path's geometry, and
    the logged heading at each perturbed point's arc length [B,P,T] (their heading while they barely move:
    the tangent of a stopped ego's logged path is position jitter).

    Lateral offsets ramp in over time and over the first LATERAL_TRAVEL_M of travel, so an ego that
    (nearly) stands still is not slid sideways."""
    b, t, _ = gt_xy.shape
    dev, dtype = gt_xy.device, gt_xy.dtype
    path = torch.cat([torch.zeros_like(gt_xy[:, :1]), gt_xy], 1)                         # origin + T
    ext, cum = polyline_with_extension(path)
    yaws = torch.cat([torch.zeros_like(gt_yaw[:, :1]), gt_yaw, gt_yaw[:, -1:]], 1)        # per vertex of ext
    s_gt = cum[:, 1:t + 1]                                                                 # [B,T]
    time = torch.arange(1, t + 1, device=dev, dtype=dtype) * dt                            # [T]

    lat, scale, ramp, stop = [], [], [], []

    def add(d=0.0, f=1.0, r=2.0, a=0.0):
        lat.append(d if torch.is_tensor(d) else torch.full((b,), d, device=dev, dtype=dtype))
        scale.append(f if torch.is_tensor(f) else torch.full((b,), f, device=dev, dtype=dtype))
        ramp.append(r if torch.is_tensor(r) else torch.full((b,), r, device=dev, dtype=dtype))
        stop.append(a)

    for d in _LATERAL:
        add(d=d)
    for f in _SPEED:
        add(f=f)
    for a in _STOP_DECEL:
        add(a=a)
    add(d=2.5 * nudge_sign.to(dtype), r=1.5)
    add(d=1.5, f=1.2)
    add(d=-1.5, f=1.2)
    for _ in range(2):
        u = torch.rand((3, b), generator=generator).to(device=dev, dtype=dtype)
        add(d=-3.0 + 6.0 * u[0], f=0.6 + 0.7 * u[1], r=1.0 + 2.0 * u[2])

    d = torch.stack(lat, 1)[..., None]                                                     # [B,P,1]
    f = torch.stack(scale, 1)[..., None]
    r = torch.stack(ramp, 1)[..., None]
    s = f * s_gt[:, None]                                                                  # [B,P,T]
    decel = torch.tensor(stop, device=dev, dtype=dtype)[None, :, None]                     # [1,P,1]
    v0 = ego_speed.to(dtype)[:, None, None]
    a = decel.clamp_min(1e-3)
    t_stop = v0 / a
    s_stop = torch.where(time < t_stop, v0 * time - 0.5 * a * time ** 2, v0 ** 2 / (2.0 * a))
    s = torch.where(decel > 0, s_stop, s)
    pos, _ = interp_path(ext, cum, s)                                                      # [B,P,T,2]
    yaw = interp_angle(cum, yaws, s)                                                       # [B,P,T]
    normal = torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1)
    out = pos + (d * _smoothstep(time / r) * (s / LATERAL_TRAVEL_M).clamp(0.0, 1.0))[..., None] * normal
    return out, yaw


def nearest_agent_side(f: dict) -> torch.Tensor:
    """+1 / -1: side (left / right) of the nearest valid agent within 30 m ahead (+1 if none)."""
    xy = f["current_local_xy"].float()
    ok = f["agent_valid"].bool() & ~f["is_sdc"].bool() & (xy[..., 0] > 0) & (xy.norm(dim=-1) < 30.0)
    d = torch.where(ok, xy.norm(dim=-1), torch.full_like(xy[..., 0], float("inf")))
    idx = d.argmin(1)
    y = xy[torch.arange(xy.shape[0], device=xy.device), idx, 1]
    return torch.where(torch.isfinite(d.amin(1)) & (y < 0), -torch.ones_like(y), torch.ones_like(y))


def build_candidates(f: dict, dec: dict, max_range: float, dt: float,
                     generator: torch.Generator | None = None) -> dict:
    """Ego candidates [B,C,T,2] (m, ego frame, rear axle), headings `yaw` [B,C,T] (decoded anchor yaw,
    logged yaw, travel direction of the perturbation with the logged heading while still), kinds [C] and
    validity [B,C] (perturbations need a fully valid logged future)."""
    b = f["is_sdc"].shape[0]
    ar = torch.arange(b, device=f["is_sdc"].device)
    ego = f["is_sdc"].float().argmax(1)
    ego_xy = dec["pred_xy_k"][ar, ego].detach().float() * max_range                        # [B,K,T,2]
    yv = dec["yaw_vec_k"][ar, ego].detach().float()
    ego_yaw = f["current_yaw_local"].float()[ar, ego][:, None, None] + torch.atan2(yv[..., 0], yv[..., 1])
    gt, gt_yaw = f["ego_future_xy"].float(), f["ego_future_yaw"].float()
    gt_ok = f["ego_future_valid"].bool().all(-1)                                           # [B]
    gt_safe = torch.where(gt_ok[:, None, None], gt, torch.zeros_like(gt))
    pert, pert_yaw = perturb_gt(gt_safe, gt_yaw, f["ego_vel"].float().norm(dim=-1), nearest_agent_side(f), dt, generator)
    k, p = ego_xy.shape[1], pert.shape[1]
    xy = torch.cat([ego_xy, gt_safe[:, None], pert], 1)
    yaw = torch.cat([ego_yaw, gt_yaw[:, None], path_yaw(pert, pert_yaw)], 1)
    kinds = torch.tensor([KIND_ANCHOR] * k + [KIND_GT] + [KIND_PERTURB] * p, device=xy.device)
    valid = torch.cat([torch.ones((b, k), dtype=torch.bool, device=xy.device), gt_ok[:, None].expand(b, 1 + p)], 1)
    return {"xy": xy, "yaw": yaw, "kinds": kinds, "valid": valid}


def _with_midpoints(x: torch.Tensor, angle: bool = False, axial: bool = False) -> torch.Tensor:
    """[...,T,(D)] on the step axis (-2 for vectors, -1 for scalars) -> [...,2T-1,(D)]: each step
    followed by the midpoint to the next one. Angles are interpolated on the circle; `axial` angles
    (box headings, symmetric under a half turn) modulo pi, so a tracker's heading flip between two
    frames does not rotate the box crosswise at the midpoint."""
    angle = angle or axial
    vec = not angle and x.dim() >= 2 and x.shape[-1] == 2
    ax = -2 if vec else -1
    a = x.narrow(ax, 0, x.shape[ax] - 1)
    b = x.narrow(ax, 1, x.shape[ax] - 1)
    if axial:
        mid = a + 0.5 * (torch.remainder(b - a + 0.5 * torch.pi, torch.pi) - 0.5 * torch.pi)
    else:
        mid = a + 0.5 * wrap_angle(b - a) if angle else 0.5 * (a + b)
    inter = torch.stack([a, mid], dim=ax).flatten(ax - 1, ax) if vec else torch.stack([a, mid], dim=-1).flatten(-2)
    return torch.cat([inter, x.narrow(ax, x.shape[ax] - 1, 1)], dim=ax)


def route_polyline(f: dict) -> torch.Tensor:
    """[B,R,2]: the route centreline, or the logged ego path (from the origin, padded with its end) for
    scenes without a route."""
    route = f["route_xy"].float()
    r = route.shape[1]
    gt = torch.cat([torch.zeros_like(f["ego_future_xy"][:, :1]), f["ego_future_xy"].float()], 1)
    gt = torch.cat([gt, gt[:, -1:].expand(-1, max(0, r - gt.shape[1]), 2)], 1)[:, :r]
    has = f["route_valid"].sum(-1) >= 2
    return torch.where(has[:, None, None], route, gt)


def _points_in_polygons(q: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    """Crossing-number test: points q [E,Q,2] in polygons poly [E,L,P,2] (closed implicitly) -> [E,Q,L]."""
    a = poly[:, None]                                                                      # [E,1,L,P,2]
    b = torch.roll(poly, -1, dims=2)[:, None]
    qx, qy = q[:, :, None, None, 0], q[:, :, None, None, 1]                                # [E,Q,1,1]
    crosses = (a[..., 1] > qy) != (b[..., 1] > qy)
    x_at = a[..., 0] + (qy - a[..., 1]) * (b[..., 0] - a[..., 0]) / (b[..., 1] - a[..., 1]).masked_fill(~crosses, 1.0)
    return ((crosses & (qx < x_at)).sum(-1) % 2) == 1


def in_common_or_connected_lanes(corners: torch.Tensor, lane_poly: torch.Tensor, lane_valid: torch.Tensor,
                                 chunk: int = 1024) -> torch.Tensor:
    """[E] True where all four ego corners [E,4,2] (FL, FR, RR, RL) lie in one lane / lane connector, or the
    two front and the two rear corners each share one (crossing into a connected lane): nuplan's
    `get_timestamps_in_common_or_connected_route_objs`, without the lane graph (connectivity is assumed when
    front and rear pairs each share a lane)."""
    out = []
    for e0 in range(0, corners.shape[0], chunk):
        inside = _points_in_polygons(corners[e0:e0 + chunk], lane_poly[e0:e0 + chunk]) & lane_valid[e0:e0 + chunk, None]
        common = inside.all(1).any(-1)
        front, rear = (inside[:, 0] & inside[:, 1]).any(-1), (inside[:, 2] & inside[:, 3]).any(-1)
        out.append(common | (front & rear))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.bool, device=corners.device)


def _gather_t(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B,C|1,n|1,T,...] at time idx [B,C,n] -> [B,C,n,...]."""
    b, c, n = idx.shape
    x = x.expand(b, c, n, *x.shape[3:])
    return x.gather(3, idx.view(b, c, n, 1, *([1] * (x.dim() - 4))).expand(b, c, n, 1, *x.shape[4:])).squeeze(3)


@torch.no_grad()
def label_candidates(f: dict, cand_xy: torch.Tensor, cand_yaw: torch.Tensor, dt: float,
                     obj_chunk: int = 32, midpoints: bool | None = None) -> dict:
    """Online labels for candidates [B,C,T,2] / headings [B,C,T] against the batch's logged futures,
    following nuPlan's closed-loop metrics (no_ego_at_fault_collisions, drivable_area_compliance).

    Collisions: each object's first contact is classified as nuplan's `_get_collision_type` does (ego stopped
    -> not at fault; object stopped, incl. every static object -> at fault; object behind the ego (> 150 deg
    from its heading, from the rear axle) -> not at fault; ego front bumper touching it -> at fault; else
    lateral -> at fault when the ego's corners are not in one / connected lanes). `overlap` is an at-fault
    collision, `any_overlap` any contact. Offroad: a corner more than OFFROAD_TOL_M outside the drivable area.

    With `midpoints` (default: when dt > 0.1 s), collisions and offroad are also checked halfway between
    steps (linear interpolation of both boxes), which recovers short overlaps that a 0.2 s grid skips
    over. At 0.1 s they change no label on the smoke cache and double the cost."""
    b, c, t, _ = cand_xy.shape
    if midpoints is None:
        midpoints = dt > 0.1 + 1e-6
    dev = cand_xy.device
    o_xy, o_yaw, o_lw = f["fut_xy"].float(), f["fut_yaw"].float(), f["fut_lw"].float()   # [B,N,T,...]
    o_valid = f["fut_valid"].bool()
    o_speed = f["fut_vel"].float().norm(dim=-1)                                             # [B,N,T]
    step = torch.diff(torch.cat([torch.zeros_like(cand_xy[:, :, :1]), cand_xy], 2), dim=2)
    e_speed = step.norm(dim=-1) / dt                                                       # [B,C,T]
    g_xy, g_yaw = cand_xy, cand_yaw
    if midpoints and t > 1:
        g_xy, g_yaw = _with_midpoints(cand_xy), _with_midpoints(cand_yaw, angle=True)
        o_xy, o_yaw, o_lw = _with_midpoints(o_xy), _with_midpoints(o_yaw, axial=True), _with_midpoints(o_lw)
        o_valid = _with_midpoints(o_valid.float()) > 0.75
        o_speed, e_speed = _with_midpoints(o_speed), _with_midpoints(e_speed)
    tg = g_xy.shape[2]
    ego_lw = f["ego_lw"].float()[:, None, None].expand(b, c, tg, 2)
    ego_c = box_corners(ego_center(g_xy, g_yaw), g_yaw, ego_lw)                            # [B,C,Tg,4,2] FL, FR, RR, RL
    heading = torch.stack([torch.cos(g_yaw), torch.sin(g_yaw)], -1)
    o_static = f["fut_static"].bool()
    lane_poly, lane_valid = f["lane_poly"].float(), f["lane_poly_valid"].bool()

    # Per object: first contact, its classification and the min gap, chunked over objects.
    any_hit = torch.zeros((b, c), dtype=torch.bool, device=dev)
    at_fault = torch.zeros((b, c), dtype=torch.bool, device=dev)
    first_fault = torch.full((b, c), tg, dtype=torch.long, device=dev)
    clearance = torch.full((b, c), float("inf"), device=dev)            # min gap over objects not hit without fault
    for n0 in range(0, o_xy.shape[1], obj_chunk):
        sl = slice(n0, n0 + obj_chunk)
        oc = box_corners(o_xy[:, sl], o_yaw[:, sl], o_lw[:, sl])                           # [B,n,Tg,4,2]
        gap = sat_gap(ego_c[:, :, None], g_yaw[:, :, None], oc[:, None], o_yaw[:, None, sl])  # [B,C,n,Tg]
        gap = gap.masked_fill(~o_valid[:, None, sl], float("inf"))
        grid = torch.arange(tg, device=dev).view(1, 1, 1, tg)
        first = torch.where(gap <= 0.0, grid, torch.full_like(grid, tg)).amin(-1)          # [B,C,n]
        hit = first < tg
        fi = first.clamp(max=tg - 1)
        ego_stopped = _gather_t(e_speed[:, :, None], fi) <= STOPPED_SPEED
        obj_stopped = o_static[:, None, sl] | (_gather_t(o_speed[:, None, sl], fi) <= STOPPED_SPEED)
        to_obj = _gather_t(o_xy[:, None, sl], fi) - _gather_t(g_xy[:, :, None], fi)
        behind = (to_obj * _gather_t(heading[:, :, None], fi)).sum(-1) < BEHIND_COS * to_obj.norm(dim=-1).clamp_min(1e-6)
        ec = _gather_t(ego_c[:, :, None], fi)                                               # [B,C,n,4,2]
        bumper = torch.stack([ec[..., 0, :], ec[..., 1, :], ec[..., 1, :], ec[..., 0, :]], -2)
        front = sat_gap(bumper, _gather_t(g_yaw[:, :, None], fi), _gather_t(oc[:, None], fi),
                        _gather_t(o_yaw[:, None, sl], fi)) <= 0.0
        fault = hit & ~ego_stopped & (obj_stopped | (~behind & front))
        lateral = hit & ~ego_stopped & ~obj_stopped & ~behind & ~front
        if lateral.any():
            bi, ci, ni = lateral.nonzero(as_tuple=True)
            ok = in_common_or_connected_lanes(ec[bi, ci, ni], lane_poly[bi], lane_valid[bi])
            fault[bi, ci, ni] = ~ok
        any_hit |= hit.any(-1)
        at_fault |= fault.any(-1)
        first_fault = torch.minimum(first_fault, torch.where(fault, first, torch.full_like(first, tg)).amin(-1))
        counted = gap.amin(-1).masked_fill(hit & ~fault, float("inf"))
        clearance = torch.minimum(clearance, counted.amin(-1))
    grid_t = (torch.arange(tg, device=dev, dtype=torch.float32) + 2.0) / 2.0 if (midpoints and t > 1) else (
        torch.arange(tg, device=dev, dtype=torch.float32) + 1.0)                          # plan step of each grid point
    grid_t = torch.cat([grid_t, grid_t.new_tensor([t + 1.0])])                               # index tg = none
    first_overlap = grid_t[first_fault]
    clearance = torch.where(torch.isfinite(clearance), clearance, torch.full_like(clearance, 50.0))

    # Offroad: signed distance of every box corner to the drivable-area boundary, beyond the tolerance.
    edges = {k: f[k] for k in ("edge_xy", "edge_dir", "edge_prev_dir", "edge_len", "edge_prev_len", "edge_same_prev", "edge_valid")}
    edges = {k: (v.float() if v.is_floating_point() else v) for k, v in edges.items()}
    depth = edge_signed_distance(ego_c.reshape(b, -1, 2), edges).reshape(b, c, tg, 4).amax(-1) - OFFROAD_TOL_M
    offroad_depth = depth.amax(-1)
    first_offroad = torch.where(depth > 0.0, grid_t[:tg], grid_t[tg]).amin(-1)

    # Progress along the logged ego path and along the route, comfort.
    gt = f["ego_future_xy"].float()
    gt_valid = f["ego_future_valid"].bool()
    last_valid = torch.where(gt_valid.any(1), t - 1 - gt_valid.flip(1).float().argmax(1),
                             torch.zeros_like(gt_valid[:, 0], dtype=torch.long))
    ar = torch.arange(b, device=dev)
    gt_fill = torch.where(gt_valid[..., None], gt, gt[ar, last_valid][:, None])
    gt_fill = torch.where(gt_valid.any(1)[:, None, None], gt_fill, torch.zeros_like(gt_fill))
    ext, cum = polyline_with_extension(torch.cat([torch.zeros_like(gt[:, :1]), gt_fill], 1))
    end = cand_xy[:, :, -1]
    gt_end = gt_fill[:, -1:]
    progress = project_progress(end, ext, cum)
    gt_progress = project_progress(gt_end, ext, cum)[:, 0]
    r_ext, r_cum = polyline_with_extension(route_polyline(f))
    r0 = project_progress(torch.zeros_like(gt_end), r_ext, r_cum)                          # [B,1]
    route_progress = project_progress(end, r_ext, r_cum) - r0
    gt_route_progress = (project_progress(gt_end, r_ext, r_cum) - r0)[:, 0]
    comfort = comfort_excess(cand_xy, f["ego_vel"].float()[:, None].expand(b, c, 2), dt)

    return {"overlap": at_fault, "any_overlap": any_hit, "clearance": clearance, "offroad": offroad_depth > 0.0,
            "offroad_depth": offroad_depth, "progress": progress, "gt_progress": gt_progress,
            "route_progress": route_progress, "gt_route_progress": gt_route_progress, "comfort": comfort,
            "first_overlap": first_overlap, "first_offroad": first_offroad,
            "horizon": torch.full((b,), t, device=dev, dtype=torch.long)}
