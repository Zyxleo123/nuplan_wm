"""Torch geometry for online labels and the scorer (ego frame, metres / radians)."""

from __future__ import annotations

import torch

from anchor_wm.config import EGO_REAR_TO_CENTER


def rotate_xy(xy: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Rotate xy by the angle (sin, cos). Pass -sin to rotate by the negative angle."""
    x, y = xy[..., 0], xy[..., 1]
    return torch.stack([x * cos - y * sin, x * sin + y * cos], dim=-1)


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi


def box_corners(xy: torch.Tensor, yaw: torch.Tensor, lw: torch.Tensor) -> torch.Tensor:
    """[...,2], [...], [...,2] (length, width) -> corners [...,4,2]."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    hl, hw = 0.5 * lw[..., 0], 0.5 * lw[..., 1]
    fx, fy = c * hl, s * hl          # half-length along heading
    lx, ly = -s * hw, c * hw         # half-width to the left
    signs = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]], device=xy.device, dtype=xy.dtype)
    dx = signs[:, 0] * fx[..., None] + signs[:, 1] * lx[..., None]
    dy = signs[:, 0] * fy[..., None] + signs[:, 1] * ly[..., None]
    return xy[..., None, :] + torch.stack([dx, dy], dim=-1)


def ego_center(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Ego box centre [...,2] from rear-axle positions [...,2] and headings [...] (nuPlan's ego frame)."""
    return xy + EGO_REAR_TO_CENTER * torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)


def sat_gap(ca: torch.Tensor, ya: torch.Tensor, cb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """Separating-axis gap of two oriented boxes (corners [...,4,2], yaws [...]), broadcasting.

    Returns the largest separation over the four box axes: > 0 means the boxes are disjoint
    (and the value is a lower bound on their distance), <= 0 means they overlap."""
    ya, yb = torch.broadcast_tensors(ya, yb)
    axes = torch.stack([
        torch.stack([torch.cos(ya), torch.sin(ya)], -1), torch.stack([-torch.sin(ya), torch.cos(ya)], -1),
        torch.stack([torch.cos(yb), torch.sin(yb)], -1), torch.stack([-torch.sin(yb), torch.cos(yb)], -1),
    ], dim=-2)                                                                     # [...,4 axes,2]
    pa = (ca.unsqueeze(-3) * axes.unsqueeze(-2)).sum(-1)                          # [...,4 axes,4 corners]
    pb = (cb.unsqueeze(-3) * axes.unsqueeze(-2)).sum(-1)
    gap = torch.maximum(pb.amin(-1) - pa.amax(-1), pa.amin(-1) - pb.amax(-1))     # [...,4 axes]
    return gap.amax(-1)


def edge_signed_distance(query: torch.Tensor, edges: dict, chunk: int = 1024, budget: float = 8e6) -> torch.Tensor:
    """Signed distance [B,Q] from query points [B,Q,2] to the nearest drivable-area boundary point; > 0 is
    offroad (outside the drivable area).

    The boundary rings are oriented with the drivable side on the left (build_cache), so the rule of
    waymax.metrics.roadgraph.compute_signed_distance_to_nearest_road_edge_point applies: the sign comes
    from the cross product of (query - nearest) with the boundary direction at the nearest point, or with
    its predecessor's direction when that point is on the same ring and gives the smaller cross product.
    Scenes without boundary points return -100 (on road)."""
    exy, edir, eprev = edges["edge_xy"], edges["edge_dir"], edges["edge_prev_dir"]
    esame, evalid = edges["edge_same_prev"].bool(), edges["edge_valid"].bool()
    # Keep the [B,chunk,E,2] difference tensor around `budget` elements (8e6: 64 MB) whatever the number
    # of edge points.
    chunk = max(16, min(chunk, int(budget // max(1, exy.shape[0] * exy.shape[1]))))
    out = []
    for s0 in range(0, query.shape[1], chunk):
        q = query[:, s0:s0 + chunk]
        d2 = (q.unsqueeze(2) - exy.unsqueeze(1)).square().sum(-1)                 # [B,q,E]
        d2 = d2.masked_fill(~evalid.unsqueeze(1), float("inf"))
        idx = d2.argmin(-1)                                                        # [B,q]
        take = lambda t: torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))  # noqa: E731
        near, ndir, pdir = take(exy), take(edir), take(eprev)
        same = torch.gather(esame, 1, idx)
        p2e = q - near
        cross = p2e[..., 0] * ndir[..., 1] - p2e[..., 1] * ndir[..., 0]
        cross_prev = p2e[..., 0] * pdir[..., 1] - p2e[..., 1] * pdir[..., 0]
        signed = torch.where(same & (cross_prev < cross), cross_prev, cross)
        dist = p2e.norm(dim=-1) * torch.sign(signed)
        out.append(torch.where(evalid.any(1, keepdim=True), dist, torch.full_like(dist, -100.0)))
    return torch.cat(out, 1)


def path_yaw(xy: torch.Tensor, fallback_yaw: torch.Tensor, min_step_m: float = 0.05) -> torch.Tensor:
    """Heading of travel from the origin through xy [...,T,2]; holds the last defined heading
    (or `fallback_yaw` [...,T] before any motion) while the step is below `min_step_m`.
    """
    diff = torch.diff(torch.cat([torch.zeros_like(xy[..., :1, :]), xy], dim=-2), dim=-2)
    heading = torch.atan2(diff[..., 1], diff[..., 0])
    steps = torch.arange(xy.shape[-2], device=xy.device).expand(xy.shape[:-1])
    moving_idx = torch.where(diff.norm(dim=-1) > min_step_m, steps, torch.full_like(steps, -1))
    last_idx = torch.cummax(moving_idx, dim=-1).values
    held = torch.gather(heading, -1, last_idx.clamp(min=0))
    return torch.where(last_idx >= 0, held, fallback_yaw)


def candidate_features(xy: torch.Tensor, yaw: torch.Tensor, dt: float, pos_scale: float,
                       vel_scale: float) -> torch.Tensor:
    """[...,T,2] positions + [...,T] heading -> [...,T,6] (xy, velocity, sin/cos heading)."""
    vel = torch.diff(torch.cat([torch.zeros_like(xy[..., :1, :]), xy], dim=-2), dim=-2) / dt
    return torch.cat([xy / pos_scale, vel / vel_scale, torch.sin(yaw)[..., None], torch.cos(yaw)[..., None]], -1)


def polyline_with_extension(path: torch.Tensor, extend_m: float = 60.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Path [B,P,2] starting at the origin -> (path with a straight extension [B,P+1,2], arc length
    [B,P+1]). The extension continues the last moving segment (or +x) so arc-length queries beyond the
    logged end stay defined."""
    seg = torch.diff(path, dim=1)
    seg_len = seg.norm(dim=-1)
    unit = seg / seg_len.clamp_min(1e-6)[..., None]
    steps = torch.arange(seg.shape[1], device=path.device).expand(seg_len.shape)
    last = torch.cummax(torch.where(seg_len > 1e-3, steps, torch.full_like(steps, -1)), 1).values[:, -1]
    fallback = torch.tensor([1.0, 0.0], device=path.device, dtype=path.dtype).expand(path.shape[0], 2)
    last_dir = torch.where((last >= 0)[:, None],
                           torch.gather(unit, 1, last.clamp_min(0)[:, None, None].expand(-1, 1, 2))[:, 0], fallback)
    ext = torch.cat([path, (path[:, -1] + extend_m * last_dir)[:, None]], 1)
    cum = torch.cat([torch.zeros_like(seg_len[:, :1]), torch.cumsum(torch.diff(ext, dim=1).norm(dim=-1), 1)], 1)
    return ext, cum


def interp_angle(cum: torch.Tensor, angles: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Per-vertex angles [B,P] (e.g. logged headings) interpolated on the circle at arc lengths s [B,...]
    along a polyline with cumulative lengths cum [B,P]."""
    b = s.shape[0]
    flat = s.reshape(b, -1).clamp(min=0.0).contiguous()
    hi = torch.searchsorted(cum.contiguous(), flat, right=True).clamp(1, cum.shape[1] - 1)
    lo = hi - 1
    c_lo, c_hi = torch.gather(cum, 1, lo), torch.gather(cum, 1, hi)
    t = ((flat - c_lo) / (c_hi - c_lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    a_lo, a_hi = torch.gather(angles, 1, lo), torch.gather(angles, 1, hi)
    return (a_lo + t * wrap_angle(a_hi - a_lo)).reshape(s.shape)


def interp_path(ext: torch.Tensor, cum: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions and unit tangents at arc lengths s [B,...] along (ext, cum) from `polyline_with_extension`."""
    b = s.shape[0]
    flat = s.reshape(b, -1).clamp(min=0.0).contiguous()
    hi = torch.searchsorted(cum.contiguous(), flat, right=True).clamp(1, cum.shape[1] - 1)
    lo = hi - 1
    c_lo, c_hi = torch.gather(cum, 1, lo), torch.gather(cum, 1, hi)
    p_lo = torch.gather(ext, 1, lo[..., None].expand(-1, -1, 2))
    p_hi = torch.gather(ext, 1, hi[..., None].expand(-1, -1, 2))
    seg = p_hi - p_lo
    length = (c_hi - c_lo).clamp_min(1e-6)
    t = ((flat - c_lo) / length).clamp(0.0, 1.0)
    pos = p_lo + t[..., None] * seg
    seg_n = seg.norm(dim=-1, keepdim=True)
    tangent = torch.where(seg_n > 1e-6, seg / seg_n.clamp_min(1e-6), torch.tensor([1.0, 0.0], device=s.device))
    return pos.reshape(*s.shape, 2), tangent.reshape(*s.shape, 2)


def project_progress(points: torch.Tensor, ext: torch.Tensor, cum: torch.Tensor) -> torch.Tensor:
    """Arc length [B,C] of the projection of points [B,C,2] onto the polyline (ext, cum)."""
    a, b = ext[:, :-1], ext[:, 1:]                                                  # [B,S,2]
    ab = b - a
    ab2 = ab.square().sum(-1).clamp_min(1e-9)
    ap = points.unsqueeze(2) - a.unsqueeze(1)                                       # [B,C,S,2]
    t = ((ap * ab.unsqueeze(1)).sum(-1) / ab2.unsqueeze(1)).clamp(0.0, 1.0)
    d = (ap - t.unsqueeze(-1) * ab.unsqueeze(1)).norm(dim=-1)
    seg = d.argmin(-1)                                                              # [B,C]
    t_sel = torch.gather(t, 2, seg.unsqueeze(-1)).squeeze(-1)
    return torch.gather(cum[:, :-1], 1, seg) + t_sel * torch.gather(ab2.sqrt(), 1, seg)


def comfort_excess(xy: torch.Tensor, v0: torch.Tensor, dt: float, lat_limit: float = 4.5,
                   lon_limit: float = 7.0) -> torch.Tensor:
    """Largest exceedance [...] (m/s^2) of lateral / longitudinal acceleration limits along xy [...,T,2]
    starting at velocity v0 [...,2]. Velocities are 3-tap smoothed first: raw second differences of
    logged positions at 0.1-0.2 s are dominated by position noise."""
    vel = torch.diff(torch.cat([torch.zeros_like(xy[..., :1, :]), xy], dim=-2), dim=-2) / dt
    vel = torch.cat([v0.unsqueeze(-2), vel], dim=-2)
    pad = torch.cat([vel[..., :1, :], vel, vel[..., -1:, :]], dim=-2)
    vel = (pad[..., :-2, :] + pad[..., 1:-1, :] + pad[..., 2:, :]) / 3.0
    acc = torch.diff(vel, dim=-2) / dt                                               # [...,T,2]
    v_mid = 0.5 * (vel[..., 1:, :] + vel[..., :-1, :])
    speed = v_mid.norm(dim=-1, keepdim=True)
    u = torch.where(speed > 0.5, v_mid / speed.clamp_min(1e-6), torch.zeros_like(v_mid))
    lon = (acc * u).sum(-1)
    lat = acc[..., 0] * u[..., 1] - acc[..., 1] * u[..., 0]
    return torch.maximum(lat.abs() - lat_limit, lon.abs() - lon_limit).amax(-1)
