"""What the repair scores a plan against, built online from a nuPlan ``PlannerInput`` in the ego frame
(rear axle at the origin, x forward; the frame Diffusion Planner plans in):

* the DRIVABLE_AREA boundary (``anchor_wm.build_cache.drivable_area_arrays``, the cache's own extraction, so
  the repair's edge is the labels' edge), rebuilt only when the ego has moved ``refresh_m`` from where it was;
* the other objects' boxes over the scored window: constant velocity from their current state (static objects
  stay put), optionally replaced by a model's forecast for the agents it predicts (Diffusion Planner's own joint
  prediction, or AnchorWM's modes).
"""

from __future__ import annotations

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from anchor_wm.build_cache import drivable_area_arrays

NEIGHBOR_FORECASTS = ("cv", "dp", "anchor", "none")


def to_ego(xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """World xy ``[..., 2]`` -> the frame of ``pose`` (x, y, yaw); float64 in, float64 out."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = xy - pose[:2]
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def rotate(vec: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]], -1)


def ego_pose(ego_state: EgoState) -> np.ndarray:
    r = ego_state.rear_axle
    return np.array([r.x, r.y, r.heading], dtype=np.float64)


class DrivableArea:
    """The drivable-area boundary within ``radius_m`` of where it was last built (world frame, float64)."""

    def __init__(self, radius_m: float = 100.0, refresh_m: float = 25.0, query_m: float = 60.0,
                 max_points: int = 16384) -> None:
        self.radius_m, self.refresh_m, self.query_m, self.max_points = radius_m, refresh_m, query_m, max_points
        self._centre = None
        self._map = None
        self._arr = None

    def _build(self, map_api, centre: np.ndarray) -> None:
        # Built in a north-up frame at `centre`, so the stored points are world coordinates minus `centre`.
        arr = drivable_area_arrays(map_api, np.array([centre[0], centre[1], 0.0]), self.radius_m, self.max_points)
        self._arr = {k: np.asarray(v, np.float64) for k, v in arr.items()}
        self._centre, self._map = centre.copy(), map_api.map_name

    def __call__(self, map_api, pose: np.ndarray, device) -> dict:
        """``anchor_wm.geometry.edge_signed_distance``'s edge dict (batch 1) in the frame of ``pose``."""
        if (self._centre is None or self._map != map_api.map_name
                or np.hypot(*(pose[:2] - self._centre)) > self.refresh_m):
            self._build(map_api, pose[:2])
        a = self._arr
        xy = to_ego(a["da_xy"] + self._centre, pose)
        keep = np.flatnonzero(np.hypot(xy[:, 0], xy[:, 1]) <= self.query_m)
        out = {"edge_xy": xy[keep], "edge_dir": rotate(a["da_dir"][keep], pose[2]),
               "edge_prev_dir": rotate(a["da_prev_dir"][keep], pose[2]),
               "edge_len": a["da_len"][keep], "edge_prev_len": a["da_prev_len"][keep]}
        valid = np.ones(len(keep), bool)
        if not len(keep):                       # one invalid point: edge_signed_distance reads "on road"
            out = {k: np.zeros((1,) + v.shape[1:]) for k, v in out.items()}
            valid = np.zeros(1, bool)
        out = {k: torch.as_tensor(v, dtype=torch.float32, device=device)[None] for k, v in out.items()}
        # Closed rings: every point's predecessor is on its own ring.
        out["edge_same_prev"] = torch.as_tensor(valid, device=device)[None]
        out["edge_valid"] = torch.as_tensor(valid, device=device)[None]
        return out

    def cache_arrays(self, map_api, pose: np.ndarray, crop_m: float = 150.0, max_points: int = 8192) -> dict:
        """The cache's ``da_*`` (``drivable_area_arrays(map_api, pose, crop_m, max_points)``, build_cache's
        ``--da_radius`` / ``--max_da_points``): points in the frame of ``pose`` inside its ``+-crop_m`` square,
        float32, capped to the ``max_points`` nearest in ring order. Every point of that square must be within
        ``radius_m - refresh_m`` of the build centre, so build this instance with ``radius_m >= sqrt(2) crop_m +
        refresh_m``."""
        if (self._centre is None or self._map != map_api.map_name
                or np.hypot(*(pose[:2] - self._centre)) > self.refresh_m):
            self._build(map_api, pose[:2])
        a = self._arr
        xy = to_ego(a["da_xy"] + self._centre, pose)
        keep = np.flatnonzero((np.abs(xy) <= crop_m).all(-1))
        if len(keep) > max_points:
            keep = np.sort(keep[np.argsort(np.hypot(xy[keep, 0], xy[keep, 1]), kind="stable")[:max_points]])
        out = {"da_xy": xy[keep], "da_dir": rotate(a["da_dir"][keep], pose[2]),
               "da_prev_dir": rotate(a["da_prev_dir"][keep], pose[2]),
               "da_len": a["da_len"][keep], "da_prev_len": a["da_prev_len"][keep]}
        return {k: v.astype(np.float32) for k, v in out.items()}


def speed_limit(map_api, ego_state: EgoState) -> float:
    """The limit nuPlan's ``speed_limit_compliance`` holds the ego to where it is now: the speed limit of the lane under
    the ego's centre, or on lane connectors the largest of their incoming / outgoing lanes' (``route_extractor``'s
    lookup); inf where the map gives none (the metric then counts no violation)."""
    point = ego_state.center.point
    lane = map_api.get_one_map_object(point, SemanticMapLayer.LANE)
    if lane is not None:
        limits = [lane.speed_limit_mps]
    else:
        limits = [e.speed_limit_mps for c in map_api.get_all_map_objects(point, SemanticMapLayer.LANE_CONNECTOR)
                  for e in c.outgoing_edges + c.incoming_edges]
    return float(max(limits)) if limits and all(limits) else float("inf")


def current_objects(observation, pose: np.ndarray, radius_m: float) -> dict:
    """Every tracked object within ``radius_m`` of the ego, ego frame: box centre ``xy [M, 2]``, ``yaw [M]``,
    ``lw [M, 2]`` (length, width), ``vel [M, 2]`` (0 for static objects), ``is_agent [M]``."""
    rows = []
    for obj in observation.tracked_objects.tracked_objects:
        c = obj.center
        is_agent = obj.tracked_object_type in AGENT_TYPES
        v = obj.velocity if is_agent else None
        rows.append((c.x, c.y, c.heading, obj.box.length, obj.box.width,
                     v.x if v is not None else 0.0, v.y if v is not None else 0.0, float(is_agent)))
    a = np.asarray(rows, np.float64).reshape(-1, 8)
    xy = to_ego(a[:, :2], pose)
    keep = np.hypot(xy[:, 0], xy[:, 1]) <= radius_m
    a, xy = a[keep], xy[keep]
    return {"xy": xy, "yaw": a[:, 2] - pose[2], "lw": a[:, 3:5], "vel": rotate(a[:, 5:7], pose[2]),
            "is_agent": a[:, 7] > 0.5}


def forecast_boxes(objects: dict, n: int, dt: float, device, predicted: dict | None = None,
                   match_m: float = 0.5) -> dict:
    """The repair's ``agents`` dict (batch 1) over steps ``1..n``: constant velocity with the heading held; ``vel``
    is each box's velocity (the forecast's finite differences for predicted agents).

    ``predicted``: a model's forecast for some agents, ``xy [P, K, T, 2]`` and ``yaw [P, K, T]`` at 0.1 s,
    ``prob [P, K]`` (summing to 1 per agent), and each agent's current box centre ``cur_xy [P, 2]`` and size
    ``lw [P, 2]``. Each replaces the forecast of the tracked agent at its current position (within ``match_m``)
    by K pseudo-agents weighted by their probabilities, so the agent barrier is the expectation over modes (the
    WOMD repair's ``wm_mf``); one that matches no tracked agent is added.
    """
    k = np.arange(1, n + 1, dtype=np.float64)[:, None] * dt
    m = len(objects["xy"])
    xy = objects["xy"][None] + objects["vel"][None] * k[..., None]                  # [n, M, 2]
    yaw = np.broadcast_to(objects["yaw"][None], (n, m))
    vel = np.broadcast_to(objects["vel"][None], (n, m, 2))
    lw, weight = objects["lw"], np.ones(m)
    if predicted is not None and len(predicted["cur_xy"]):
        keep = np.ones(m, bool)
        agents = np.flatnonzero(objects["is_agent"])
        for p in range(len(predicted["cur_xy"])):
            if len(agents):
                dist = np.hypot(*(objects["xy"][agents] - predicted["cur_xy"][p]).T)
                j = int(np.argmin(dist))
                if dist[j] <= match_m:
                    keep[agents[j]] = False
        P, K = predicted["prob"].shape
        pxy = predicted["xy"][:, :, :n].reshape(P * K, n, 2).transpose(1, 0, 2)    # [n, P*K, 2]
        pyaw = predicted["yaw"][:, :, :n].reshape(P * K, n).T
        prev = np.concatenate([np.repeat(predicted["cur_xy"], K, 0)[None], pxy[:-1]], 0)
        xy = np.concatenate([xy[:, keep], pxy], 1)
        yaw = np.concatenate([yaw[:, keep], pyaw], 1)
        vel = np.concatenate([vel[:, keep], (pxy - prev) / dt], 1)
        lw = np.concatenate([lw[keep], np.repeat(predicted["lw"], K, 0)], 0)
        weight = np.concatenate([weight[keep], predicted["prob"].reshape(-1)])
    box = np.concatenate([xy, np.broadcast_to(lw[None], (n, len(lw), 2)), yaw[..., None]], -1)
    box = torch.as_tensor(box, dtype=torch.float32, device=device)[None]           # [1, n, M, 5]
    w = torch.as_tensor(np.broadcast_to(weight[None], (n, len(weight))).copy(), dtype=torch.float32, device=device)
    return {"box": box, "valid": torch.ones(box.shape[:3], dtype=torch.bool, device=device), "weight": w[None],
            "vel": torch.as_tensor(np.array(vel), dtype=torch.float32, device=device)[None]}
