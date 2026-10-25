"""nuPlan scenarios -> AnchorWM cache: one npz per (single-sample) scenario, everything in the ego frame
at t = 0 (origin at the rear axle, x forward, y left; metres / radians, float32).

The time grid is 0.1 s: PAST_FRAMES = 21 frames up to and including t = 0, then FUTURE_FRAMES = 80
(t = 0.1 ... 8.0 s), so frame 20 is "now". Per file:

  meta        log_name, token, map_name, scenario_type, timestamp_us, origin_pose (world x, y, yaw of the frame;
              float64)
  ego         ego_pose [101,3] (x, y, yaw of the rear axle), ego_vel / ego_acc [101,2] (rear-axle velocity /
              acceleration rotated into the t = 0 frame)
  objects     every tracked object (vehicle / pedestrian / bicycle / czone sign / barrier / traffic cone /
              generic object) seen within --object_radius of the ego at t = 0 in any of the 101 frames,
              nearest first (at most --max_objects): obj_pose [N,101,3] (box centre), obj_vel [N,101,2] (float16),
              obj_valid [N,101], obj_lw [N,2] (length, width), obj_type [N] (config.OBJ_TYPES index).
              Missing frames are invalid (no padding), objects that only appear later are included.
  lanes       lanes + lane connectors within --map_radius, nearest first: lane_xy / lane_left / lane_right
              [L,20,2] (centreline and boundaries, each resampled to 20 points), lane_type [L] (0 lane,
              1 connector), lane_on_route [L], lane_tl [L] (config.TL_STATES index at t = 0), lane_tl_future
              [L,17] (t = 0, 0.5, ..., 8 s), lane_speed_limit [L] (m/s), lane_has_speed_limit [L], lane_id /
              lane_roadblock_id [L] (map ids), lane topology as indices into these lanes (-1 = none / not cached):
              lane_succ / lane_pred [L,6], lane_left_nbr / lane_right_nbr [L] (the devkit's outgoing / incoming
              / adjacent edges)
  polygons    crosswalks, stop lines, intersections: poly_xy [P,20,2] (exterior ring), poly_type [P]
              (0 crosswalk, 1 stop line, 2 intersection)
  route       route_roadblock_ids [K] (corrected route), route_xy [R,2]: centreline of the lane path along the
              route roadblocks, 1 m apart, from 30 m behind the ego's projection on it; empty when no route lane
              is found near the ego
  goal        goal_pose [3], goal_valid (mission goal: the scene's goal ego pose, as scenario.get_mission_goal)
  drivable    da_xy / da_dir / da_prev_dir [E,2], da_len / da_prev_len [E]: boundary of the union of the
              DRIVABLE_AREA layer (what nuplan's drivable_area_compliance uses) within --da_radius: the ring
              vertices plus points at most 1 m apart, each with its outgoing / incoming segment. Rings are
              oriented so the drivable side is on the left of `da_dir`, which gives the Waymax-style sign
              (> 0 offroad) with every point's predecessor on the same ring.

Small caches list and build in one go:

  python -m anchor_wm.build_cache --data_path .../nuplan-v1.1/splits/trainval --map_path .../maps \
      --save_path .../cache_smoke --max_db_files 12 --total_scenarios 300

Large caches plan once, then build in shards (slurm/cache.sh): listing scenarios reads every log, so it
is done once (`--make_plan` writes {log_name: [tokens]}); each shard then opens only its own logs and
skips scenarios whose file exists, so a preempted shard restarts where it stopped.

  python -m anchor_wm.build_cache ... --total_scenarios 1000000 --plan_file .../plan.json --make_plan
  python -m anchor_wm.build_cache ... --plan_file .../plan.json --num_shards 8 --shard_id 3
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import sqlite3  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402
from shapely.ops import orient, unary_union  # noqa: E402

from nuplan.common.actor_state.ego_state import EgoState  # noqa: E402
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint  # noqa: E402
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType  # noqa: E402
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters  # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder  # noqa: E402
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter  # noqa: E402
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_sensor_token  # noqa: E402
from nuplan.database.utils.label.utils import local2agent_type, raw_mapping  # noqa: E402
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon  # noqa: E402
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from anchor_wm.config import CACHE_DT, FUTURE_FRAMES, PAST_FRAMES  # noqa: E402
from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction  # noqa: E402

OBJ_TYPE_INDEX = {
    TrackedObjectType.VEHICLE: 0, TrackedObjectType.PEDESTRIAN: 1, TrackedObjectType.BICYCLE: 2,
    TrackedObjectType.CZONE_SIGN: 3, TrackedObjectType.BARRIER: 4, TrackedObjectType.TRAFFIC_CONE: 5,
    TrackedObjectType.GENERIC_OBJECT: 6,
}
TL_INDEX = {"GREEN": 0, "YELLOW": 1, "RED": 2, "UNKNOWN": 3}
MAP_POINTS = 20
NUM_FRAMES = PAST_FRAMES + FUTURE_FRAMES
ROUTE_BEHIND_M = 30.0
DA_SMOOTH_M = 0.1
DA_SPACING_M = 1.0
# Frame offsets (in 20 Hz lidar rows) of the cached frames around t = 0, as the devkit samples them
# (NuPlanScenario.get_past / get_future_tracked_objects with these horizons and sample counts).
DB_ROW_INTERVAL = 0.05
FRAME_OFFSETS = np.array(
    [-i for i in sample_indices_with_time_horizon(PAST_FRAMES - 1, (PAST_FRAMES - 1) * CACHE_DT, DB_ROW_INTERVAL)][::-1]
    + [0] + sample_indices_with_time_horizon(FUTURE_FRAMES, FUTURE_FRAMES * CACHE_DT, DB_ROW_INTERVAL))
# Traffic-light states are kept at t = 0 and every 0.5 s of the future (lidar-row offsets).
TL_OFFSETS = np.arange(0, FUTURE_FRAMES * 2 + 1, 10)
MAX_LANE_EDGES = 6                      # successors / predecessors kept per lane
CATEGORY_INDEX = {name: OBJ_TYPE_INDEX.get(TrackedObjectType[local2agent_type[raw_mapping["global2local"][name]]], -1)
                  for name in raw_mapping["global2local"]}
AGENT_INDICES = {OBJ_TYPE_INDEX[t] for t in AGENT_TYPES if t in OBJ_TYPE_INDEX}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", required=True, help="directory of nuPlan .db files")
    p.add_argument("--map_path", required=True)
    p.add_argument("--map_version", default="nuplan-maps-v1.0")
    p.add_argument("--save_path", required=True)
    p.add_argument("--log_names_json", default="nuplan_train.json", help="logs to use (empty = every log)")
    p.add_argument("--max_db_files", type=int, default=0, help="use only the first N matching .db files (small caches)")
    p.add_argument("--total_scenarios", type=int, default=0, help="0 = no limit")
    p.add_argument("--scenarios_per_type", type=int, default=0, help="0 = no limit")
    p.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--plan_file", default="", help="scenario plan: written with --make_plan, read by the shards")
    p.add_argument("--make_plan", action="store_true", help="list the scenarios, write --plan_file and exit")
    p.add_argument("--max_tasks_per_child", type=int, default=20,
                   help="restart workers after this many logs (bounds the devkit's map-object cache)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--object_radius", type=float, default=120.0)
    p.add_argument("--max_objects", type=int, default=192)
    p.add_argument("--map_radius", type=float, default=120.0)
    p.add_argument("--max_lanes", type=int, default=256)
    p.add_argument("--max_polygons", type=int, default=64)
    p.add_argument("--da_radius", type=float, default=150.0)
    p.add_argument("--max_da_points", type=int, default=8192)
    p.add_argument("--route_length_m", type=float, default=400.0)
    return p.parse_args()


# ---- geometry helpers ------------------------------------------------------------------------

def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def to_local(xy: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """World xy [...,2] -> frame of origin (x, y, yaw)."""
    c, s = np.cos(origin[2]), np.sin(origin[2])
    d = xy - origin[:2]
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def rotate_to_local(vec: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate vectors [...,2] by -yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]], -1)


def resample(xy: np.ndarray, n: int, closed: bool = False) -> np.ndarray:
    """Polyline [P,2] -> n points evenly spaced in arc length (degenerate lines repeat their first point)."""
    if closed:
        xy = np.concatenate([xy, xy[:1]], 0)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return np.repeat(xy[:1], n, 0)
    s = np.linspace(0.0, cum[-1], n, endpoint=not closed)
    return np.stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])], -1)


def path_xy(path) -> np.ndarray:
    return np.array([[p.x, p.y] for p in path], dtype=np.float64)


# ---- per-scenario extraction -----------------------------------------------------------------

def _yaw(qw, qx, qy, qz) -> float:
    return Quaternion(qw, qx, qy, qz).yaw_pitch_roll[0]          # as the devkit's queries convert ego poses


class LogData:
    """One log read with a few queries instead of the devkit's ~105 per scenario: every lidar frame of the
    merged point cloud in timestamp order (the order get_sampled_lidarpcs_from_db samples in) with its ego
    pose and the scene's mission goal, then the traffic lights at the scenarios' t = 0 frames and the boxes of
    every frame the scenarios need. Rows and conversions match the devkit's (nuplan_scenario_queries)."""

    def __init__(self, db_file: str, tokens: list) -> None:
        sensor = bytearray.fromhex(get_sensor_token(db_file, "lidar", "MergedPointCloud"))
        con = sqlite3.connect(db_file)
        try:
            rows = con.execute("""
                SELECT lp.token, lp.timestamp, ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz, ep.vx, ep.vy,
                       ep.acceleration_x, ep.acceleration_y, gp.x, gp.y, gp.qw, gp.qx, gp.qy, gp.qz
                FROM lidar_pc AS lp
                LEFT JOIN ego_pose AS ep ON ep.token = lp.ego_pose_token
                LEFT JOIN scene AS s ON s.token = lp.scene_token
                LEFT JOIN ego_pose AS gp ON gp.token = s.goal_ego_pose_token
                WHERE lp.lidar_token = ?
                ORDER BY lp.timestamp ASC""", (sensor,)).fetchall()
            self.index = {r[0].hex(): i for i, r in enumerate(rows)}
            self.timestamp = np.array([r[1] for r in rows], np.int64)
            self.pose = np.array([(r[2], r[3], _yaw(*r[4:8])) for r in rows], np.float64)
            self.vel = np.array([r[8:10] for r in rows], np.float64)
            self.acc = np.array([r[10:12] for r in rows], np.float64)
            self.goal = [None if r[12] is None else (r[12], r[13], _yaw(*r[14:18])) for r in rows]
            starts = [self.index[t] for t in tokens if t in self.index]
            frames = sorted({int(i) for i0 in starts for i in i0 + FRAME_OFFSETS if 0 <= i < len(rows)})
            self.tl = {int(i): {} for i0 in starts for i in i0 + TL_OFFSETS if i < len(rows)}
            frames = sorted(set(frames) | set(self.tl))
            self.boxes = {i: [] for i in frames}
            for chunk in range(0, len(frames), 500):
                ids = frames[chunk:chunk + 500]
                tok = [bytearray(rows[i][0]) for i in ids]
                marks = ",".join("?" * len(ids))
                at = {rows[i][0]: i for i in ids}
                for r in con.execute(f"""
                        SELECT lb.lidar_pc_token, c.name, lb.x, lb.y, lb.yaw, lb.length, lb.width, lb.vx, lb.vy, lb.track_token
                        FROM lidar_box AS lb
                        INNER JOIN track AS t ON t.token = lb.track_token
                        INNER JOIN category AS c ON c.token = t.category_token
                        WHERE lb.lidar_pc_token IN ({marks})""", tok):
                    self.boxes[at[r[0]]].append(r[1:])
                for r in con.execute(f"""
                        SELECT lidar_pc_token, status, lane_connector_id FROM traffic_light_status
                        WHERE lidar_pc_token IN ({marks})""", tok):
                    i = at[r[0]]
                    if i in self.tl:
                        self.tl[i][str(r[2])] = TL_INDEX.get(str(r[1]).upper(), TL_INDEX["UNKNOWN"])
        finally:
            con.close()

    def ego_state(self, i: int) -> EgoState:
        return EgoState.build_from_rear_axle(
            StateSE2(*self.pose[i]), tire_steering_angle=0.0, vehicle_parameters=get_pacifica_parameters(),
            time_point=TimePoint(int(self.timestamp[i])), rear_axle_velocity_2d=StateVector2D(*self.vel[i]),
            rear_axle_acceleration_2d=StateVector2D(*self.acc[i]))


def ego_arrays(log: LogData, idx: np.ndarray, origin: np.ndarray) -> dict:
    pose = log.pose[idx]
    rel = wrap(pose[:, 2] - origin[2])
    return {"ego_pose": np.concatenate([to_local(pose[:, :2], origin), rel[:, None]], -1),
            "ego_vel": rotate_to_local(log.vel[idx], -rel), "ego_acc": rotate_to_local(log.acc[idx], -rel)}


def object_arrays(log: LogData, idx: np.ndarray, origin: np.ndarray, radius: float, max_objects: int) -> dict:
    tracks: dict[bytes, dict] = {}
    for t, i in enumerate(idx):
        for name, x, y, yaw, length, width, vx, vy, track in log.boxes[int(i)]:
            ty = CATEGORY_INDEX.get(name, -1)
            if ty < 0:
                continue
            tr = tracks.get(track)
            if tr is None:
                tr = tracks[track] = {"type": ty, "pose": np.zeros((NUM_FRAMES, 3)), "vel": np.zeros((NUM_FRAMES, 2)),
                                      "lw": np.zeros((NUM_FRAMES, 2)), "valid": np.zeros(NUM_FRAMES, bool)}
            tr["pose"][t] = (x, y, yaw)
            if ty in AGENT_INDICES and vx is not None and vy is not None and np.isfinite(vx) and np.isfinite(vy):
                tr["vel"][t] = (vx, vy)
            tr["lw"][t] = (length, width)
            tr["valid"][t] = True
    if not tracks:
        return {"obj_pose": np.zeros((0, NUM_FRAMES, 3)), "obj_vel": np.zeros((0, NUM_FRAMES, 2)),
                "obj_valid": np.zeros((0, NUM_FRAMES), bool), "obj_lw": np.zeros((0, 2)), "obj_type": np.zeros(0, np.int8)}
    trs = list(tracks.values())
    pose = np.stack([t["pose"] for t in trs])
    valid = np.stack([t["valid"] for t in trs])
    local = to_local(pose[..., :2], origin)
    dist = np.where(valid, np.linalg.norm(local, axis=-1), np.inf).min(1)
    order = np.argsort(dist, kind="stable")
    order = order[dist[order] <= radius][:max_objects]
    now = PAST_FRAMES - 1
    out = {"obj_pose": [], "obj_vel": [], "obj_valid": [], "obj_lw": [], "obj_type": []}
    for i in order:
        tr = trs[i]
        # Box size: at t = 0 when the object is there, else its first valid frame.
        ref = now if tr["valid"][now] else int(np.argmax(tr["valid"]))
        rel = wrap(tr["pose"][:, 2] - origin[2])
        out["obj_pose"].append(np.concatenate([local[i], rel[:, None]], -1) * tr["valid"][:, None])
        out["obj_vel"].append(rotate_to_local(tr["vel"], np.full(NUM_FRAMES, origin[2])) * tr["valid"][:, None])
        out["obj_valid"].append(tr["valid"])
        out["obj_lw"].append(tr["lw"][ref])
        out["obj_type"].append(tr["type"])
    return {k: np.stack(v) if k != "obj_type" else np.array(v, np.int8) for k, v in out.items()}


_GRAPHS: dict = {}


def lane_graph(map_api) -> dict:
    """Lane-graph edges of a whole map, from its lane / lane-connector tables (once per worker and map), with
    the devkit's rules: a lane's outgoing edges are the connectors whose exit_lane_fid is the lane, a
    connector leads into its entry_lane_fid; neighbours are lanes of the same lane group with lane_index -+ 1
    (smaller index on the left). Keys are (kind, id) with kind 0 lane / 1 connector."""
    name = map_api.map_name
    if name not in _GRAPHS:
        lanes = map_api._get_vector_map_layer(SemanticMapLayer.LANE)
        conns = map_api._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR)
        succ: dict = {}
        pred: dict = {}
        for fid, exit_lane, entry_lane in zip(conns["fid"].astype(str), conns["exit_lane_fid"].astype(str),
                                              conns["entry_lane_fid"].astype(str)):
            succ.setdefault((0, exit_lane), []).append((1, fid))
            pred.setdefault((0, entry_lane), []).append((1, fid))
            succ[(1, fid)] = [(0, entry_lane)]
            pred[(1, fid)] = [(0, exit_lane)]
        slot = {(str(g), int(i)): str(f) for f, g, i in zip(lanes["fid"], lanes["lane_group_fid"], lanes["lane_index"])}
        left = {(0, f): (0, slot[(str(g), int(i) - 1)]) for f, g, i in zip(lanes["fid"].astype(str), lanes["lane_group_fid"],
                                                                         lanes["lane_index"]) if (str(g), int(i) - 1) in slot}
        right = {(0, f): (0, slot[(str(g), int(i) + 1)]) for f, g, i in zip(lanes["fid"].astype(str), lanes["lane_group_fid"],
                                                                          lanes["lane_index"]) if (str(g), int(i) + 1) in slot}
        _GRAPHS[name] = {"succ": succ, "pred": pred, "left": left, "right": right}
    return _GRAPHS[name]


def lane_arrays(map_api, point: Point2D, origin: np.ndarray, radius: float, max_lanes: int,
                route_ids: set, traffic_lights: list) -> dict:
    layers = map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    rows = []
    for layer, kind in ((SemanticMapLayer.LANE, 0), (SemanticMapLayer.LANE_CONNECTOR, 1)):
        for obj in layers[layer]:
            center = path_xy(obj.baseline_path.discrete_path)
            if len(center) < 2:
                continue
            dist = float(np.linalg.norm(center - origin[:2], axis=-1).min())
            rows.append((dist, kind, obj, center))
    rows.sort(key=lambda r: r[0])
    rows = rows[:max_lanes]
    n = len(rows)
    out = {"lane_xy": np.zeros((n, MAP_POINTS, 2)), "lane_left": np.zeros((n, MAP_POINTS, 2)),
           "lane_right": np.zeros((n, MAP_POINTS, 2)), "lane_type": np.zeros(n, np.int8),
           "lane_on_route": np.zeros(n, bool), "lane_tl_future": np.full((n, len(TL_OFFSETS)), TL_INDEX["UNKNOWN"], np.int8),
           "lane_speed_limit": np.zeros(n), "lane_has_speed_limit": np.zeros(n, bool),
           "lane_id": np.array([str(r[2].id) for r in rows], dtype=str),
           "lane_roadblock_id": np.array([str(r[2].get_roadblock_id()) for r in rows], dtype=str),
           "lane_succ": np.full((n, MAX_LANE_EDGES), -1, np.int16), "lane_pred": np.full((n, MAX_LANE_EDGES), -1, np.int16),
           "lane_left_nbr": np.full(n, -1, np.int16), "lane_right_nbr": np.full(n, -1, np.int16)}
    graph = lane_graph(map_api)
    slot = {(r[1], str(r[2].id)): i for i, r in enumerate(rows)}
    for i, (_, kind, obj, center) in enumerate(rows):
        c = resample(center, MAP_POINTS)
        out["lane_xy"][i] = to_local(c, origin)
        for key, bound in (("lane_left", obj.left_boundary), ("lane_right", obj.right_boundary)):
            b = resample(path_xy(bound.discrete_path), MAP_POINTS)
            if np.linalg.norm(b[-1] - c[0]) < np.linalg.norm(b[0] - c[0]):
                b = b[::-1]
            out[key][i] = to_local(b, origin)
        out["lane_type"][i] = kind
        out["lane_on_route"][i] = obj.get_roadblock_id() in route_ids
        out["lane_tl_future"][i] = [tl.get(str(obj.id), TL_INDEX["UNKNOWN"]) for tl in traffic_lights]
        key = (kind, str(obj.id))
        for name in ("succ", "pred"):
            nbrs = [slot[e] for e in graph[name].get(key, []) if e in slot][:MAX_LANE_EDGES]
            out[f"lane_{name}"][i, :len(nbrs)] = nbrs
        for name in ("left", "right"):
            out[f"lane_{name}_nbr"][i] = slot.get(graph[name].get(key), -1)
        if obj.speed_limit_mps is not None:
            out["lane_speed_limit"][i], out["lane_has_speed_limit"][i] = obj.speed_limit_mps, True
    out["lane_tl"] = out["lane_tl_future"][:, 0].copy()
    return out


def polygon_arrays(map_api, point: Point2D, origin: np.ndarray, radius: float, max_polygons: int) -> dict:
    kinds = ((SemanticMapLayer.CROSSWALK, 0), (SemanticMapLayer.STOP_LINE, 1), (SemanticMapLayer.INTERSECTION, 2))
    layers = map_api.get_proximal_map_objects(point, radius, [k for k, _ in kinds])
    ego = shapely.Point(origin[0], origin[1])
    rows = []
    for layer, kind in kinds:
        for obj in layers[layer]:
            poly = obj.polygon
            rows.append((poly.distance(ego), kind, np.asarray(poly.exterior.coords)[:-1]))
    rows.sort(key=lambda r: r[0])
    rows = rows[:max_polygons]
    xy = np.zeros((len(rows), MAP_POINTS, 2))
    for i, (_, _, ring) in enumerate(rows):
        xy[i] = to_local(resample(ring, MAP_POINTS, closed=True), origin)
    return {"poly_xy": xy, "poly_type": np.array([r[1] for r in rows], np.int8)}


def route_centerline(map_api, point: Point2D, origin: np.ndarray, route_ids: list, length_m: float) -> np.ndarray:
    """Lane path through the route roadblocks starting at the route lane best matching the ego pose,
    following lane-graph successors that stay on the route (the one reaching furthest along the route),
    as a 1 m-spaced centreline from ROUTE_BEHIND_M behind the ego's projection, in the ego frame."""
    rank = {rid: i for i, rid in enumerate(route_ids)}
    near = map_api.get_proximal_map_objects(point, 15.0, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    cands = [o for layer in near.values() for o in layer if o.get_roadblock_id() in rank]
    if not cands:
        return np.zeros((0, 2))

    def start_cost(o) -> float:
        pose = o.baseline_path.get_nearest_pose_from_position(point)
        return float(np.hypot(pose.x - point.x, pose.y - point.y) + 5.0 * abs(wrap(pose.heading - origin[2])))

    start = min(cands, key=start_cost)
    memo: dict[str, tuple] = {}

    def best(lane, budget: float, stack: frozenset) -> tuple:
        # (furthest route rank reached, total length, lanes); memoised per lane (first budget wins).
        if lane.id in memo:
            return memo[lane.id]
        length = float(lane.baseline_path.length)
        res = (rank[lane.get_roadblock_id()], length, [lane])
        if budget - length > 0:
            nxt = [e for e in lane.outgoing_edges if e.get_roadblock_id() in rank and e.id not in stack]
            subs = [best(e, budget - length, stack | {lane.id}) for e in nxt]
            if subs:
                r, ln, lanes = max(subs, key=lambda s: (s[0], s[1]))
                res = (r, length + ln, [lane] + lanes)
        memo[lane.id] = res
        return res

    lanes = best(start, length_m + 100.0, frozenset())[2]
    xy = np.concatenate([path_xy(l.baseline_path.discrete_path) for l in lanes], 0)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=-1) > 1e-3])
    xy = xy[keep]
    if len(xy) < 2:
        return np.zeros((0, 2))
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s_ego = cum[int(np.argmin(np.linalg.norm(xy - origin[:2], axis=-1)))]
    s = np.arange(max(0.0, s_ego - ROUTE_BEHIND_M), min(cum[-1], s_ego + length_m), 1.0)
    pts = np.stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])], -1)
    return to_local(pts, origin)


def drivable_area_arrays(map_api, origin: np.ndarray, radius: float, max_points: int) -> dict:
    layer = map_api._get_vector_map_layer(SemanticMapLayer.DRIVABLE_AREA)   # union of the drivable layers
    patch = box(origin[0] - radius, origin[1] - radius, origin[0] + radius, origin[1] + radius)
    geoms = [g for g in layer.geometry.iloc[layer.sindex.query(patch, predicate="intersects")] if g is not None]
    empty = {"da_xy": np.zeros((0, 2)), "da_dir": np.zeros((0, 2)), "da_prev_dir": np.zeros((0, 2)),
             "da_len": np.zeros(0), "da_prev_len": np.zeros(0)}
    if not geoms:
        return empty
    # Closing then opening (grow, union, shrink twice, grow): adjacent map polygons leave hairline cracks and
    # zero-width spikes that a plain union keeps; their opposite-facing sides / ambiguous tips would flip the
    # offroad sign of points metres away. Features narrower than 2 * DA_SMOOTH_M are removed.
    union = unary_union([g.buffer(DA_SMOOTH_M) for g in geoms]).buffer(-2.0 * DA_SMOOTH_M).buffer(DA_SMOOTH_M)
    polys = [union] if isinstance(union, Polygon) else [g for g in getattr(union, "geoms", []) if isinstance(g, Polygon)]
    out = {k: [] for k in empty}
    for poly in polys:
        poly = orient(poly, sign=1.0)                       # exterior CCW, holes CW: drivable side on the left
        for ring in [poly.exterior, *poly.interiors]:
            # The ring's own vertices plus points so that no segment is longer than DA_SPACING_M; every point
            # keeps its outgoing / incoming segment (direction, length) for exact point-to-segment distances.
            p = shapely.get_coordinates(shapely.segmentize(ring, DA_SPACING_M))[:-1]
            # Drop repeated vertices: a zero-length segment has no direction, which would flip the sign.
            p = p[np.linalg.norm(np.roll(p, -1, 0) - p, axis=-1) > 1e-3]
            if len(p) < 3:
                continue
            seg = np.roll(p, -1, 0) - p
            length = np.linalg.norm(seg, axis=-1)
            d = rotate_to_local(seg / np.maximum(length, 1e-6)[:, None], np.full(len(p), origin[2]))
            local = to_local(p, origin)
            inside = (np.abs(local) <= radius).all(-1)
            for key, v in (("da_xy", local), ("da_dir", d), ("da_prev_dir", np.roll(d, 1, 0)), ("da_len", length),
                           ("da_prev_len", np.roll(length, 1))):
                out[key].append(v[inside])
    out = {k: np.concatenate(v) for k, v in out.items()}
    if len(out["da_xy"]) > max_points:
        keep = np.sort(np.argsort(np.linalg.norm(out["da_xy"], axis=-1), kind="stable")[:max_points])
        out = {k: v[keep] for k, v in out.items()}
    return out


def process_scenario(scenario, log: LogData, args) -> dict | None:
    """None when the log has no full 2 s past / 8 s future around the scenario."""
    i0 = log.index[scenario.token]
    idx = i0 + FRAME_OFFSETS
    if idx[0] < 0 or idx[-1] >= len(log.timestamp):
        return None
    origin = log.pose[i0].copy()
    point = Point2D(origin[0], origin[1])
    map_api = scenario.map_api
    route_ids = scenario.get_route_roadblock_ids()
    if route_ids != [""]:
        route_ids = route_roadblock_correction(log.ego_state(i0), map_api, route_ids)
    route_ids = [r for r in route_ids if r]
    goal = log.goal[i0]
    goal_pose = np.zeros(3)
    if goal is not None:
        goal_pose = np.concatenate([to_local(np.array(goal[:2]), origin), [wrap(goal[2] - origin[2])]])

    data = {"log_name": scenario.log_name, "token": scenario.token, "map_name": map_api.map_name,
            "scenario_type": scenario.scenario_type, "timestamp_us": np.int64(log.timestamp[i0]),
            "origin_pose": origin.copy(),
            "goal_pose": goal_pose, "goal_valid": np.bool_(goal is not None)}
    data.update(ego_arrays(log, idx, origin))
    data.update(object_arrays(log, idx, origin, args.object_radius, args.max_objects))
    tls = [log.tl[int(i0 + o)] for o in TL_OFFSETS]
    data.update(lane_arrays(map_api, point, origin, args.map_radius, args.max_lanes, set(route_ids), tls))
    data.update(polygon_arrays(map_api, point, origin, args.map_radius, args.max_polygons))
    data["route_xy"] = route_centerline(map_api, point, origin, route_ids, args.route_length_m)
    data["route_roadblock_ids"] = np.array(route_ids, dtype=str)
    data.update(drivable_area_arrays(map_api, origin, args.da_radius, args.max_da_points))
    # float32 everywhere (origin_pose float64); object velocities float16 (0.01 m/s), the largest array after poses.
    data = {k: v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 and k != "origin_pose" else v
            for k, v in data.items()}
    data["obj_vel"] = data["obj_vel"].astype(np.float16)
    return data


def cache_file(save_path: str, scenario) -> str:
    return os.path.join(save_path, f"{scenario.log_name}__{scenario.token}.npz")


_ARGS = None


def _init(args) -> None:
    global _ARGS
    _ARGS = args


def _work_log(scenarios: list) -> dict:
    """All scenarios of one log: read the log once, then write each scenario's file."""
    counts: dict[str, int] = {}
    todo = [s for s in scenarios if not os.path.exists(cache_file(_ARGS.save_path, s))]
    counts["exists"] = len(scenarios) - len(todo)
    if not todo:
        return counts
    # The DBs live on NFS: one sequential copy to node-local disk is several times faster than the random
    # page reads of the queries (cold 100 MB log: 4.7 s copy + 0.9 s queries). LogData reads all it needs, so
    # the copy goes right after.
    local = os.path.join(os.environ.get("TMPDIR") or "/tmp", f"awm_{socket.gethostname()}_{os.getpid()}.db")
    try:
        shutil.copyfile(os.path.join(_ARGS.data_path, todo[0].log_name + ".db"), local)
        log = LogData(local, [s.token for s in todo])
    except Exception:  # one bad log must not stop a long caching run
        print(f"FAILED log {todo[0].log_name}\n{traceback.format_exc()}", flush=True)
        counts["failed"] = len(todo)
        return counts
    finally:
        if os.path.exists(local):
            os.remove(local)
    for scenario in todo:
        try:
            data = process_scenario(scenario, log, _ARGS)
        except Exception:
            print(f"FAILED {scenario.log_name} {scenario.token}\n{traceback.format_exc()}", flush=True)
            counts["failed"] = counts.get("failed", 0) + 1
            continue
        if data is None:
            counts["short"] = counts.get("short", 0) + 1
            continue
        path = cache_file(_ARGS.save_path, scenario)
        tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
        with open(tmp, "wb") as fh:
            np.savez(fh, **data)
        os.replace(tmp, path)
        counts["written"] = counts.get("written", 0) + 1
    return counts


def scenario_filter(args, log_names: list | None, tokens: list | None = None) -> ScenarioFilter:
    # Same filter as data_process.py (Diffusion-Planner's training set) when listing; exact tokens when building.
    planned = tokens is not None
    return ScenarioFilter(
        scenario_types=None, scenario_tokens=tokens, log_names=log_names, map_names=None,
        num_scenarios_per_type=None if planned else args.scenarios_per_type or None,
        limit_total_scenarios=None if planned else args.total_scenarios or None,
        timestamp_threshold_s=None, ego_displacement_minimum_m=None, expand_scenarios=True,
        remove_invalid_goals=False, shuffle=not planned, ego_start_speed_threshold=None, ego_stop_speed_threshold=None,
        speed_noise_tolerance=None)


def list_scenarios(args, db_files: list | None, log_names: list | None, tokens: list | None = None) -> list:
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, db_files, args.map_version)
    random.seed(args.seed)
    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=args.num_workers)
    return sorted(builder.get_scenarios(scenario_filter(args, log_names, tokens), worker), key=lambda s: (s.log_name, s.token))


def shard_logs(plan: dict, num_shards: int, shard_id: int) -> list:
    """Logs of one shard: greedy balance of the scenario counts, largest logs first (deterministic)."""
    loads, owner = [0] * num_shards, {}
    for log in sorted(plan, key=lambda g: (-len(plan[g]), g)):
        i = loads.index(min(loads))
        owner[log], loads[i] = i, loads[i] + len(plan[log])
    return sorted(g for g, i in owner.items() if i == shard_id)


def main() -> None:
    args = parse_args()
    assert 0 <= args.shard_id < args.num_shards
    os.makedirs(args.save_path, exist_ok=True)
    log_names = None
    if args.log_names_json:
        with open(args.log_names_json) as fh:
            log_names = json.load(fh)
    t0 = time.time()
    if args.plan_file and not args.make_plan:
        with open(args.plan_file) as fh:
            plan = json.load(fh)["logs"]
        logs = shard_logs(plan, args.num_shards, args.shard_id)
        todo = {g: [t for t in plan[g] if not os.path.exists(os.path.join(args.save_path, f"{g}__{t}.npz"))] for g in logs}
        todo = {g: t for g, t in todo.items() if t}
        print(json.dumps({"shard": args.shard_id, "of": args.num_shards, "logs": len(logs),
                          "scenarios": sum(len(plan[g]) for g in logs), "todo": sum(map(len, todo.values()))}), flush=True)
        if not todo:
            return
        scenarios = list_scenarios(args, [os.path.join(args.data_path, g + ".db") for g in todo], list(todo),
                                   [t for ts in todo.values() for t in ts])
    else:
        db_files = None
        if args.max_db_files:
            wanted = set(log_names) if log_names else None
            db_files = [f for f in sorted(glob.glob(os.path.join(args.data_path, "*.db")))
                        if wanted is None or os.path.basename(f)[:-3] in wanted][:args.max_db_files]
        scenarios = list_scenarios(args, db_files, log_names)
        if args.make_plan:
            plan = {}
            for s in scenarios:
                plan.setdefault(s.log_name, []).append(s.token)
            with open(args.plan_file, "w") as fh:
                json.dump({"args": vars(args), "num_scenarios": len(scenarios), "logs": plan}, fh)
            print(json.dumps({"plan": args.plan_file, "logs": len(plan), "scenarios": len(scenarios),
                              "list_s": round(time.time() - t0, 1)}), flush=True)
            return
        shard = -(-len(scenarios) // args.num_shards)
        scenarios = scenarios[args.shard_id * shard:(args.shard_id + 1) * shard]
    print(json.dumps({"scenarios": len(scenarios), "list_s": round(time.time() - t0, 1)}), flush=True)

    # One task per log (LogData reads it once); largest logs first so the pool's tail is short.
    groups: dict[str, list] = {}
    for sc in scenarios:
        groups.setdefault(sc.log_name, []).append(sc)
    tasks = sorted(groups.values(), key=len, reverse=True)
    t0, counts, done = time.time(), {}, 0
    with Pool(args.num_workers, initializer=_init, initargs=(args,), maxtasksperchild=args.max_tasks_per_child) as pool:
        for i, c in enumerate(pool.imap_unordered(_work_log, tasks, chunksize=1)):
            for k, v in c.items():
                counts[k] = counts.get(k, 0) + v
            done += sum(c.values())
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                print(json.dumps({"logs": i + 1, "of": len(tasks), "scenarios": done, **counts,
                                  "s_per_scenario": round((time.time() - t0) / max(done, 1), 4)}), flush=True)


if __name__ == "__main__":
    main()
