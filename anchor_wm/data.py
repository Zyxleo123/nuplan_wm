"""AnchorWM cache files (anchor_wm.build_cache) -> fixed-size model inputs, forecasting targets and
label geometry. Everything is in the ego frame at t = 0 (rear axle), metres / radians.

Model inputs
  agent_history [N,H,12], agent_history_valid [N,H], agent_valid [N]
                              slot 0 is the ego, then the nearest dynamic agents valid at t = 0. Per frame:
                              x, y (/max_range), vx, vy (/max_velocity), sin / cos yaw, length, width
                              (/max_width), is-ego, type one-hot (vehicle, pedestrian, bicycle)
  object_types [N], is_sdc [N], current_local_xy [N,2], current_yaw_local [N]
  static_xy [S,2], static_yaw [S], static_lw [S,2], static_type [S], static_valid [S]
                              nearest static objects at t = 0 (encoder tokens, exact gaps in the scorer)
  map_features [M,P,F], map_valid [M,P]
                              lanes / lane connectors then crosswalks / stop lines (config.map_feature_dim)
  route_pts [Rs,2], route_pts_valid [Rs]   route centreline every route_spacing_m ahead of the ego
  goal_xy [2], goal_valid                  mission goal (used only with cfg.use_goal)
  ego_vel [2], ego_lw [2]

Targets / labels (read from the logged future)
  target_delta_xy [N,T,2] (/max_range), target_yaw_vec [N,T,2], target_valid_all [N,T]
  ego_future_xy [T,2], ego_future_yaw [T], ego_future_valid [T]   (rear axle)
  fut_xy [L,T,2], fut_yaw [L,T], fut_lw [L,T,2], fut_vel [L,T,2], fut_valid [L,T], fut_static [L]
                                           log-replay boxes of every cached object
  lane_poly [Lp,2P,2], lane_poly_valid [Lp], lane_succ [Lp,S]
                                           lane / lane-connector polygons (left boundary + reversed right) and
                                           successors, for nuplan's "ego in one or connected lanes" test
  edge_xy / edge_dir / edge_prev_dir [E,2], edge_len / edge_prev_len / edge_same_prev / edge_valid [E]
                                           drivable-area boundary points with their outgoing / incoming segments
  tok_edge_xy / tok_edge_dir [Et,2], tok_edge_valid [Et]                       scorer boundary tokens
  route_xy [R,2], route_valid [R]          route centreline, 1 m apart (progress labels, route features)
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from anchor_wm.config import (ANCHOR_GROUPS, EGO_LENGTH, EGO_WIDTH, MAP_TYPES, PAST_FRAMES, TL_STATES,
                              AnchorWMConfig)

NOW = PAST_FRAMES - 1


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _pad(x: np.ndarray, n: int) -> np.ndarray:
    x = x[:n]
    return np.concatenate([x, np.zeros((n - len(x),) + x.shape[1:], x.dtype)], 0) if len(x) < n else x


def _polyline_dirs(xy: np.ndarray) -> np.ndarray:
    """Unit direction to the next point [...,P,2] (the last point repeats the previous direction)."""
    d = np.diff(xy, axis=-2)
    d = np.concatenate([d, d[..., -1:, :]], -2)
    return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)


def agent_arrays(z, n: int) -> dict:
    """Ego + the n - 1 nearest dynamic agents valid at t = 0: poses / velocities / validity over all
    cached frames, sizes, types (padded to n)."""
    otype, ovalid, opose = z["obj_type"].astype(np.int64), z["obj_valid"], z["obj_pose"]
    dynamic = (otype < len(ANCHOR_GROUPS)) & ovalid[:, NOW]
    dist = np.where(dynamic, np.linalg.norm(opose[:, NOW, :2], axis=-1), np.inf)
    order = np.argsort(dist, kind="stable")[:n - 1]
    order = order[dynamic[order]]
    frames = z["ego_pose"].shape[0]
    return {
        "pose": _pad(np.concatenate([z["ego_pose"][None], opose[order]]), n),
        "vel": _pad(np.concatenate([z["ego_vel"][None], z["obj_vel"][order]]), n),
        "valid": _pad(np.concatenate([np.ones((1, frames), bool), ovalid[order]]), n),
        "lw": _pad(np.concatenate([np.array([[EGO_LENGTH, EGO_WIDTH]], np.float32), z["obj_lw"][order]]), n),
        "type": _pad(np.concatenate([[0], otype[order]]), n),
    }


def static_arrays(z, cfg: AnchorWMConfig) -> dict:
    """The max_static nearest static objects (czone sign / barrier / cone / generic) present at t = 0."""
    m = int(cfg.max_static)
    otype = z["obj_type"].astype(np.int64)
    ok = (otype >= len(ANCHOR_GROUPS)) & z["obj_valid"][:, NOW]
    pose = z["obj_pose"][:, NOW]
    order = np.argsort(np.where(ok, np.linalg.norm(pose[:, :2], axis=-1), np.inf), kind="stable")[:m]
    order = order[ok[order]]
    return {"static_xy": _pad(pose[order, :2], m), "static_yaw": _pad(pose[order, 2], m),
            "static_lw": _pad(z["obj_lw"][order], m), "static_type": _pad(otype[order] - len(ANCHOR_GROUPS), m),
            "static_valid": np.arange(m) < len(order)}


def map_arrays(z, cfg: AnchorWMConfig) -> tuple[np.ndarray, np.ndarray]:
    nl, npoly, p = int(cfg.max_lanes), int(cfg.max_polygons), int(cfg.map_points)
    rng = float(cfg.max_range)
    nt, ntl = len(MAP_TYPES), len(TL_STATES)
    feat = np.zeros((nl + npoly, p, cfg.map_feature_dim), np.float32)
    valid = np.zeros((nl + npoly, p), bool)

    lane = z["lane_xy"][:nl]
    k = len(lane)
    if k:
        f = feat[:k]
        f[..., 0:2] = lane / rng
        f[..., 2:4] = _polyline_dirs(lane)
        f[..., 4:6] = (z["lane_left"][:k] - lane) / 5.0
        f[..., 6:8] = (z["lane_right"][:k] - lane) / 5.0
        f[..., 8:8 + nt] = np.eye(nt, dtype=np.float32)[z["lane_type"][:k].astype(np.int64)][:, None]
        f[..., 8 + nt] = z["lane_on_route"][:k, None]
        f[..., 9 + nt:9 + nt + ntl] = np.eye(ntl, dtype=np.float32)[z["lane_tl"][:k].astype(np.int64)][:, None]
        f[..., 9 + nt + ntl] = z["lane_speed_limit"][:k, None] / 20.0
        f[..., 10 + nt + ntl] = z["lane_has_speed_limit"][:k, None]
        valid[:k] = True

    poly = z["poly_xy"][:npoly]
    k = len(poly)
    if k:
        f = feat[nl:nl + k]
        f[..., 0:2] = poly / rng
        f[..., 2:4] = _polyline_dirs(poly)
        f[..., 8:8 + nt] = np.eye(nt, dtype=np.float32)[2 + z["poly_type"][:k].astype(np.int64)][:, None]
        valid[nl:nl + k] = True
    return feat, valid


def lane_polygons(z, cfg: AnchorWMConfig) -> dict:
    """Polygons of the first max_lanes lanes and their successors among them (-1 = none)."""
    nl = int(cfg.max_lanes)
    poly = np.concatenate([z["lane_left"][:nl], z["lane_right"][:nl, ::-1]], 1)
    succ = z["lane_succ"][:nl].astype(np.int64)
    lane_succ = np.full((nl, z["lane_succ"].shape[1]), -1, np.int64)
    lane_succ[:len(succ)] = np.where(succ < nl, succ, -1)
    return {"lane_poly": _pad(poly, nl).astype(np.float32), "lane_poly_valid": np.arange(nl) < len(poly),
            "lane_succ": lane_succ}


def edge_arrays(z, cfg: AnchorWMConfig) -> dict:
    """Drivable-area boundary points inside edge_box (label edges, strided to num_label_edges) and the
    scorer's boundary tokens (a further subset inside edge_token_box)."""
    e, et = int(cfg.num_label_edges), int(cfg.num_edge_tokens)
    xy, d, pd, ln, pln = z["da_xy"], z["da_dir"], z["da_prev_dir"], z["da_len"], z["da_prev_len"]
    back, ahead, half = cfg.edge_box
    sel = np.flatnonzero((xy[:, 0] >= back) & (xy[:, 0] <= ahead) & (np.abs(xy[:, 1]) <= half))
    if len(sel) > e:
        sel = sel[::math.ceil(len(sel) / e)][:e]
    k = len(sel)
    out = {"edge_xy": _pad(xy[sel], e), "edge_dir": _pad(d[sel], e), "edge_prev_dir": _pad(pd[sel], e),
           "edge_len": _pad(ln[sel], e), "edge_prev_len": _pad(pln[sel], e),
           "edge_same_prev": np.arange(e) < k, "edge_valid": np.arange(e) < k}   # closed rings: always same ring
    tback, tahead, thalf = cfg.edge_token_box
    loc = xy[sel]
    tsel = np.flatnonzero((loc[:, 0] >= tback) & (loc[:, 0] <= tahead) & (np.abs(loc[:, 1]) <= thalf))
    if len(tsel) > et:
        tsel = tsel[np.linspace(0, len(tsel) - 1, et).round().astype(np.int64)]
    out.update(tok_edge_xy=_pad(loc[tsel], et), tok_edge_dir=_pad(d[sel][tsel], et),
               tok_edge_valid=np.arange(et) < len(tsel))
    return out


def route_arrays(z, cfg: AnchorWMConfig) -> dict:
    r = int(cfg.route_label_points)
    xy = z["route_xy"][:r]
    k = len(xy)
    out = {"route_xy": np.zeros((r, 2), np.float32), "route_valid": np.arange(r) < k,
           "route_pts": np.zeros((cfg.route_points, 2), np.float32), "route_pts_valid": np.zeros(cfg.route_points, bool)}
    if k < 2:
        return out
    out["route_xy"][:k] = xy
    out["route_xy"][k:] = xy[-1]                   # pad with the last point (zero-length segments)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))])
    s = cum[int(np.argmin(np.linalg.norm(xy, axis=-1)))] + cfg.route_spacing_m * np.arange(1, cfg.route_points + 1)
    out["route_pts"] = np.stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])], -1).astype(np.float32)
    out["route_pts_valid"] = s <= cum[-1]
    return out


def build_sample(z, cfg: AnchorWMConfig, with_targets: bool = True) -> dict[str, np.ndarray]:
    rng, vmax, wmax = float(cfg.max_range), float(cfg.max_velocity), float(cfg.max_width)
    h, t = int(cfg.history_steps), int(cfg.future_steps)
    hist_idx = NOW - np.arange(h - 1, -1, -1) * int(cfg.history_stride)
    fut_idx = NOW + np.arange(1, t + 1) * int(cfg.future_stride)

    a = agent_arrays(z, int(cfg.max_agents))
    n = len(a["type"])
    pose, hvalid = a["pose"][:, hist_idx], a["valid"][:, hist_idx]
    is_sdc = np.arange(n) == 0
    type_oh = np.eye(len(ANCHOR_GROUPS), dtype=np.float32)[a["type"]]
    hist = np.concatenate([
        pose[..., :2] / rng, a["vel"][:, hist_idx] / vmax, np.sin(pose[..., 2:3]), np.cos(pose[..., 2:3]),
        np.broadcast_to(a["lw"][:, None] / wmax, (n, h, 2)),
        np.broadcast_to(is_sdc[:, None, None], (n, h, 1)), np.broadcast_to(type_oh[:, None], (n, h, len(ANCHOR_GROUPS))),
    ], -1).astype(np.float32)
    agent_valid = a["valid"][:, NOW]
    cur_xy, cur_yaw = a["pose"][:, NOW, :2], a["pose"][:, NOW, 2]
    map_features, map_valid = map_arrays(z, cfg)

    out = {
        "agent_history": np.where(hvalid[..., None], hist, 0.0).astype(np.float32),
        "agent_history_valid": hvalid,
        "agent_valid": agent_valid,
        "object_types": a["type"].astype(np.int64),
        "is_sdc": is_sdc,
        "current_local_xy": cur_xy.astype(np.float32),
        "current_yaw_local": cur_yaw.astype(np.float32),
        "map_features": map_features,
        "map_valid": map_valid,
        "ego_vel": z["ego_vel"][NOW].astype(np.float32),
        "ego_lw": np.array([EGO_LENGTH, EGO_WIDTH], np.float32),
        "goal_xy": z["goal_pose"][:2].astype(np.float32),
        "goal_valid": np.bool_(z["goal_valid"]),
    }
    out.update(static_arrays(z, cfg))
    out.update(route_arrays(z, cfg))
    out.update(edge_arrays(z, cfg))
    if not with_targets:
        return out

    fut = a["pose"][:, fut_idx]
    target_valid = a["valid"][:, fut_idx] & agent_valid[:, None]
    yaw_delta = wrap_to_pi(fut[..., 2] - cur_yaw[:, None])
    nl = int(cfg.max_label_objects)
    lab = z["obj_pose"][:nl, fut_idx]
    out.update(
        target_delta_xy=((fut[..., :2] - cur_xy[:, None]) / rng).astype(np.float32),
        target_yaw_vec=np.stack([np.sin(yaw_delta), np.cos(yaw_delta)], -1).astype(np.float32),
        target_valid_all=target_valid,
        ego_future_xy=fut[0, :, :2].astype(np.float32),
        ego_future_yaw=fut[0, :, 2].astype(np.float32),
        ego_future_valid=a["valid"][0, fut_idx],
        fut_xy=_pad(lab[..., :2], nl).astype(np.float32),
        fut_yaw=_pad(lab[..., 2], nl).astype(np.float32),
        fut_lw=_pad(np.broadcast_to(z["obj_lw"][:nl, None], (len(lab), t, 2)), nl).astype(np.float32),
        fut_vel=_pad(z["obj_vel"][:nl, fut_idx], nl).astype(np.float32),
        fut_valid=_pad(z["obj_valid"][:nl, fut_idx], nl),
        fut_static=_pad(z["obj_type"][:nl] >= len(ANCHOR_GROUPS), nl),
        **lane_polygons(z, cfg),
    )
    return out


class AnchorWMDataset(Dataset):
    def __init__(self, files: list[str], cfg: AnchorWMConfig) -> None:
        self.files, self.cfg = list(files), cfg

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        with np.load(self.files[i]) as z:
            return build_sample(z, self.cfg)


def make_loader(files: list[str], cfg: AnchorWMConfig, *, batch_size: int, shuffle: bool, seed: int,
                num_workers: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(AnchorWMDataset(files, cfg), batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                      generator=g, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      prefetch_factor=4 if num_workers > 0 else None, persistent_workers=num_workers > 0)


def resolve_files(cache_dir: str, file_list: str = "") -> list[str]:
    """Cache files: the names in `file_list` (JSON list, relative to cache_dir) or every npz in cache_dir."""
    root = Path(cache_dir)
    if file_list:
        names = json.loads(Path(file_list).read_text())
        files = [str(root / n) for n in names]
    else:
        files = sorted(str(p) for p in root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no cache files in {cache_dir} {file_list}")
    return files


def log_of(path: str) -> str:
    """Cache files are named <log_name>__<token>.npz."""
    return Path(path).name.split("__")[0]


def split_files(files: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    """Split by log (scenarios of one log overlap in time), with a stable hash of the log name."""
    logs = sorted({log_of(f) for f in files})
    val_logs = {g for g in logs if zlib.crc32(g.encode()) % 10000 < val_fraction * 10000}
    if len(logs) == 1:                        # one log (e.g. a small subset): nothing to hold out
        val_logs = set()
    elif val_fraction > 0 and (not val_logs or len(val_logs) == len(logs)):   # tiny caches: hold out the last log
        val_logs = {logs[-1]}
    return [f for f in files if log_of(f) not in val_logs], [f for f in files if log_of(f) in val_logs]
