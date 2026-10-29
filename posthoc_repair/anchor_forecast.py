"""AnchorWM's modes as the repair's forecast, online in nuPlan's simulation.

A ``PlannerInput`` becomes the AnchorWM cache's per-scenario dict (``anchor_wm.build_cache.process_scenario``'s keys)
built by the cache's own functions: ``ego_arrays`` / ``object_arrays`` read the simulation history buffer through
:class:`HistoryLog` (the 21 past frames; the 80 future frames are empty), and ``lane_arrays``, ``polygon_arrays``,
``route_centerline`` read the map with the cache's defaults. ``anchor_wm.data.build_sample`` then makes the model
inputs exactly as in training, and the decoder's ``num_out_modes`` NMS-selected modes per agent
(``anchor_wm.model.select_modes``) go to the repair as pseudo-agents weighted by their renormalized probabilities.

What the online dict cannot hold: objects that only appear after t = 0 (the cache keeps them for its labels) and
future traffic lights (UNKNOWN; labels only). Neither reaches the model's inputs, which are the agents and static
objects present at t = 0, except through the 192-object cap. The drivable area is left empty: only the scorer reads
it. ``check_anchor_inputs.py`` compares online and cached inputs on Val14 scenarios at t = 0.
"""

from __future__ import annotations

import numpy as np
import torch

from nuplan.common.actor_state.state_representation import Point2D

from anchor_wm.build_cache import (CATEGORY_INDEX, NUM_FRAMES, OBJ_TYPE_INDEX, TL_INDEX, TL_OFFSETS, ego_arrays,
                                   lane_arrays, object_arrays, polygon_arrays, route_centerline, to_local, wrap)
from anchor_wm.config import PAST_FRAMES, AnchorWMConfig
from anchor_wm.data import NOW, agent_arrays, build_sample
from anchor_wm.model import AnchorWorldModel, select_modes
from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction

# anchor_wm.build_cache's argument defaults (what the training cache was built with).
CACHE_ARGS = {"object_radius": 120.0, "max_objects": 192, "map_radius": 120.0, "max_lanes": 256,
              "max_polygons": 64, "route_length_m": 400.0}
# A DB category name per cache object type, so object_arrays' name -> type lookup reads tracked objects.
_NAME_OF_INDEX = {}
for _name, _idx in CATEGORY_INDEX.items():
    _NAME_OF_INDEX.setdefault(_idx, _name)
_EMPTY_DA = {"da_xy": np.zeros((0, 2), np.float32), "da_dir": np.zeros((0, 2), np.float32),
             "da_prev_dir": np.zeros((0, 2), np.float32), "da_len": np.zeros(0, np.float32),
             "da_prev_len": np.zeros(0, np.float32)}


class HistoryLog:
    """The part of ``build_cache.LogData`` that ``ego_arrays`` / ``object_arrays`` read, from a simulation history
    buffer: rows 0..20 are its last 21 frames (t = -2 .. 0 s), rows 21..100 (the future) are empty."""

    def __init__(self, history) -> None:
        egos = list(history.ego_states)[-PAST_FRAMES:]
        observations = list(history.observations)[-PAST_FRAMES:]
        pad = PAST_FRAMES - len(egos)           # a buffer shorter than 2 s: the earliest rows stay empty
        self.pose, self.vel, self.acc = np.zeros((NUM_FRAMES, 3)), np.zeros((NUM_FRAMES, 2)), np.zeros((NUM_FRAMES, 2))
        self.boxes = {i: [] for i in range(NUM_FRAMES)}
        for k, (ego, obs) in enumerate(zip(egos, observations)):
            i, r, car = pad + k, ego.rear_axle, ego.dynamic_car_state
            self.pose[i] = (r.x, r.y, r.heading)
            self.vel[i] = (car.rear_axle_velocity_2d.x, car.rear_axle_velocity_2d.y)
            self.acc[i] = (car.rear_axle_acceleration_2d.x, car.rear_axle_acceleration_2d.y)
            for obj in obs.tracked_objects.tracked_objects:
                name = _NAME_OF_INDEX.get(OBJ_TYPE_INDEX.get(obj.tracked_object_type, -1))
                if name is None:
                    continue
                c, v = obj.center, getattr(obj, "velocity", None)
                self.boxes[i].append((name, c.x, c.y, c.heading, obj.box.length, obj.box.width,
                                      v.x if v is not None else None, v.y if v is not None else None,
                                      obj.track_token))


def online_scene(history, traffic_light_data, map_api, route_roadblock_ids, mission_goal) -> dict:
    """The cache dict of the current replan (``process_scenario``'s keys, its t = 0 = now)."""
    log = HistoryLog(history)
    idx = np.arange(NUM_FRAMES)
    origin = log.pose[NOW].copy()
    point = Point2D(origin[0], origin[1])
    route_ids = list(route_roadblock_ids)
    if route_ids != [""]:
        route_ids = route_roadblock_correction(history.current_state[0], map_api, route_ids)
    route_ids = [r for r in route_ids if r]
    goal_pose = np.zeros(3)
    if mission_goal is not None:
        g = np.array([mission_goal.x, mission_goal.y, mission_goal.heading])
        goal_pose = np.concatenate([to_local(g[:2], origin), [wrap(g[2] - origin[2])]])
    now_tl = {str(t.lane_connector_id): TL_INDEX.get(t.status.name, TL_INDEX["UNKNOWN"]) for t in traffic_light_data}
    tls = [now_tl] + [{} for _ in TL_OFFSETS[1:]]
    a = CACHE_ARGS
    data = {"map_name": map_api.map_name, "origin_pose": origin.copy(), "goal_pose": goal_pose,
            "goal_valid": np.bool_(mission_goal is not None)}
    data.update(ego_arrays(log, idx, origin))
    data.update(object_arrays(log, idx, origin, a["object_radius"], a["max_objects"]))
    data.update(lane_arrays(map_api, point, origin, a["map_radius"], a["max_lanes"], set(route_ids), tls))
    data.update(polygon_arrays(map_api, point, origin, a["map_radius"], a["max_polygons"]))
    data["route_xy"] = route_centerline(map_api, point, origin, route_ids, a["route_length_m"])
    data["route_roadblock_ids"] = np.array(route_ids, dtype=str)
    data.update(_EMPTY_DA)
    # As the cache stores them: float32, object velocities float16.
    data = {k: v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 and k != "origin_pose"
            else v for k, v in data.items()}
    data["obj_vel"] = data["obj_vel"].astype(np.float16)
    return data


def load_model(ckpt_path: str, device) -> AnchorWorldModel:
    """An ``anchor_wm.train`` checkpoint (its anchors are a buffer of the state dict, so no anchor file is needed)."""
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = AnchorWMConfig.from_dict(ck["config"])
    anchors = ck["model"]["anchors"].float() * float(cfg.max_range)
    model = AnchorWorldModel(cfg, anchors)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


class AnchorForecaster:
    """``forecaster(planner_input, map_api, route_ids, mission_goal)`` -> ``scene.forecast_boxes``'s ``predicted``:
    per agent valid at t = 0 (the ego excluded), its ``num_out_modes`` modes in the ego frame (box centres)."""

    def __init__(self, ckpt_path: str | None = None, device="cpu", model: AnchorWorldModel | None = None) -> None:
        self.device = torch.device(device)
        self.model = model if model is not None else load_model(ckpt_path, self.device)
        self.cfg = self.model.config

    @torch.no_grad()
    def predict(self, z: dict) -> dict:
        cfg = self.cfg
        sample = build_sample(z, cfg, with_targets=False)
        f = {k: torch.as_tensor(np.asarray(v))[None].to(self.device) for k, v in sample.items()}
        _, dec = self.model(f)
        picks = select_modes(dec, int(cfg.num_out_modes), float(cfg.nms_dist_m), float(cfg.max_range))   # [1,N,m]
        gather = lambda x: x.gather(2, picks.view(*picks.shape, *([1] * (x.dim() - 3))).expand(  # noqa: E731
            *picks.shape, *x.shape[3:]))
        rng = float(cfg.max_range)
        cur_xy, cur_yaw = f["current_local_xy"].float(), f["current_yaw_local"].float()
        xy = cur_xy[:, :, None, None] + gather(dec["pred_xy_k"]).float() * rng                  # [1,N,m,T,2]
        yv = gather(dec["yaw_vec_k"]).float()
        yaw = cur_yaw[:, :, None, None] + torch.atan2(yv[..., 0], yv[..., 1])                  # [1,N,m,T]
        prob = gather(dec["probs"]).float()
        prob = prob / prob.sum(-1, keepdim=True).clamp(min=1e-9)
        keep = f["agent_valid"][0].bool() & ~f["is_sdc"][0].bool()
        lw = agent_arrays(z, int(cfg.max_agents))["lw"]                                         # build_sample's order
        out = {"xy": xy[0, keep], "yaw": yaw[0, keep], "prob": prob[0, keep], "cur_xy": cur_xy[0, keep]}
        out = {k: v.cpu().numpy().astype(np.float64) for k, v in out.items()}
        out["lw"] = np.asarray(lw, np.float64)[keep.cpu().numpy()]
        return out

    def __call__(self, current_input, map_api, route_roadblock_ids, mission_goal) -> dict:
        z = online_scene(current_input.history, list(current_input.traffic_light_data), map_api,
                         route_roadblock_ids, mission_goal)
        return self.predict(z)
