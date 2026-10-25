"""nuPlan scenarios -> AnchorWM cache: one npz per (single-sample) scenario, everything in the ego frame
at t = 0 (origin at the rear axle, x forward, y left; metres / radians, float32).

The time grid is 0.1 s: PAST_FRAMES = 21 frames up to and including t = 0, then FUTURE_FRAMES = 80
(t = 0.1 ... 8.0 s), so frame 20 is "now". Per file:

  meta        log_name, token, map_name, scenario_type, timestamp_us
  ego         ego_pose [101,3] (x, y, yaw of the rear axle), ego_vel / ego_acc [101,2] (rear-axle velocity /
              acceleration rotated into the t = 0 frame)
  objects     every tracked object (vehicle / pedestrian / bicycle / czone sign / barrier / traffic cone /
              generic object) seen within --object_radius of the ego at t = 0 in any of the 101 frames,
              nearest first (at most --max_objects): obj_pose [N,101,3] (box centre), obj_vel [N,101,2],
              obj_valid [N,101], obj_lw [N,2] (length, width), obj_type [N] (config.OBJ_TYPES index).
              Missing frames are invalid (no padding), objects that only appear later are included.
  lanes       lanes + lane connectors within --map_radius, nearest first: lane_xy / lane_left / lane_right
              [L,20,2] (centreline and boundaries, each resampled to 20 points), lane_type [L] (0 lane,
              1 connector), lane_on_route [L], lane_tl [L] (config.TL_STATES index), lane_speed_limit [L]
              (m/s), lane_has_speed_limit [L]
  polygons    crosswalks and stop lines: poly_xy [P,20,2] (exterior ring), poly_type [P] (0 crosswalk, 1 stop line)
  route       route_xy [R,2]: centreline of the lane path along the (corrected) route roadblocks, 1 m apart,
              from 30 m behind the ego's projection on it; empty when no route lane is found near the ego
  goal        goal_pose [3], goal_valid (mission goal, scenario.get_mission_goal)
  drivable    da_xy / da_dir / da_prev_dir [E,2]: boundary of the union of the DRIVABLE_AREA layer (what
              nuplan's drivable_area_compliance uses), 1 m apart within --da_radius. Rings are oriented
              so the drivable side is on the left of `da_dir`, which gives the Waymax-style signed distance
              (> 0 offroad) with every point's predecessor on the same ring.

  python -m anchor_wm.build_cache --data_path .../nuplan-v1.1/splits/trainval --map_path .../maps \
      --save_path .../anchor_wm_cache --total_scenarios 1000000 --num_workers 32
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402
from shapely.ops import orient, unary_union  # noqa: E402

from nuplan.common.actor_state.state_representation import Point2D  # noqa: E402
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType  # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder  # noqa: E402
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter  # noqa: E402
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor  # noqa: E402

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
    p.add_argument("--max_tasks_per_child", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--object_radius", type=float, default=120.0)
    p.add_argument("--max_objects", type=int, default=256)
    p.add_argument("--map_radius", type=float, default=100.0)
    p.add_argument("--max_lanes", type=int, default=128)
    p.add_argument("--max_polygons", type=int, default=48)
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

def ego_arrays(scenario, origin: np.ndarray) -> dict | None:
    past = list(scenario.get_ego_past_trajectory(iteration=0, num_samples=PAST_FRAMES - 1,
                                                 time_horizon=(PAST_FRAMES - 1) * CACHE_DT))
    future = list(scenario.get_ego_future_trajectory(iteration=0, num_samples=FUTURE_FRAMES,
                                                     time_horizon=FUTURE_FRAMES * CACHE_DT))
    if len(past) != PAST_FRAMES - 1 or len(future) != FUTURE_FRAMES:
        return None
    states = past + [scenario.initial_ego_state] + future
    pose = np.array([[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading] for s in states])
    vel_body = np.array([[s.dynamic_car_state.rear_axle_velocity_2d.x, s.dynamic_car_state.rear_axle_velocity_2d.y]
                         for s in states])
    acc_body = np.array([[s.dynamic_car_state.rear_axle_acceleration_2d.x,
                          s.dynamic_car_state.rear_axle_acceleration_2d.y] for s in states])
    rel = wrap(pose[:, 2] - origin[2])
    return {"ego_pose": np.concatenate([to_local(pose[:, :2], origin), rel[:, None]], -1),
            "ego_vel": rotate_to_local(vel_body, -rel), "ego_acc": rotate_to_local(acc_body, -rel)}


def object_arrays(scenario, origin: np.ndarray, radius: float, max_objects: int) -> dict:
    past = [d.tracked_objects for d in scenario.get_past_tracked_objects(
        iteration=0, time_horizon=(PAST_FRAMES - 1) * CACHE_DT, num_samples=PAST_FRAMES - 1)]
    future = [d.tracked_objects for d in scenario.get_future_tracked_objects(
        iteration=0, time_horizon=FUTURE_FRAMES * CACHE_DT, num_samples=FUTURE_FRAMES)]
    frames = past + [scenario.initial_tracked_objects.tracked_objects] + future
    if len(frames) != NUM_FRAMES:
        raise ValueError(f"expected {NUM_FRAMES} frames of tracked objects, got {len(frames)}")
    tracks: dict[str, dict] = {}
    for t, objs in enumerate(frames):
        for o in objs.tracked_objects:
            ty = OBJ_TYPE_INDEX.get(o.tracked_object_type)
            if ty is None:
                continue
            tr = tracks.get(o.track_token)
            if tr is None:
                tr = tracks[o.track_token] = {"type": ty, "pose": np.zeros((NUM_FRAMES, 3)), "vel": np.zeros((NUM_FRAMES, 2)),
                                              "lw": np.zeros((NUM_FRAMES, 2)), "valid": np.zeros(NUM_FRAMES, bool)}
            tr["pose"][t] = (o.center.x, o.center.y, o.center.heading)
            v = getattr(o, "velocity", None)
            if v is not None and np.isfinite(v.x) and np.isfinite(v.y):
                tr["vel"][t] = (v.x, v.y)
            tr["lw"][t] = (o.box.length, o.box.width)
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


def lane_arrays(map_api, point: Point2D, origin: np.ndarray, radius: float, max_lanes: int,
                route_ids: set, traffic_lights: dict) -> dict:
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
           "lane_on_route": np.zeros(n, bool), "lane_tl": np.full(n, TL_INDEX["UNKNOWN"], np.int8),
           "lane_speed_limit": np.zeros(n), "lane_has_speed_limit": np.zeros(n, bool)}
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
        out["lane_tl"][i] = traffic_lights.get(str(obj.id), TL_INDEX["UNKNOWN"])
        if obj.speed_limit_mps is not None:
            out["lane_speed_limit"][i], out["lane_has_speed_limit"][i] = obj.speed_limit_mps, True
    return out


def polygon_arrays(map_api, point: Point2D, origin: np.ndarray, radius: float, max_polygons: int) -> dict:
    layers = map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.CROSSWALK, SemanticMapLayer.STOP_LINE])
    ego = shapely.Point(origin[0], origin[1])
    rows = []
    for layer, kind in ((SemanticMapLayer.CROSSWALK, 0), (SemanticMapLayer.STOP_LINE, 1)):
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
    empty = {"da_xy": np.zeros((0, 2)), "da_dir": np.zeros((0, 2)), "da_prev_dir": np.zeros((0, 2))}
    if not geoms:
        return empty
    # Closing then opening (grow, union, shrink twice, grow): adjacent map polygons leave hairline cracks and
    # zero-width spikes that a plain union keeps; their opposite-facing sides / ambiguous tips would flip the
    # offroad sign of points metres away. Features narrower than 2 * DA_SMOOTH_M are removed.
    union = unary_union([g.buffer(DA_SMOOTH_M) for g in geoms]).buffer(-2.0 * DA_SMOOTH_M).buffer(DA_SMOOTH_M)
    polys = [union] if isinstance(union, Polygon) else [g for g in getattr(union, "geoms", []) if isinstance(g, Polygon)]
    xy, dirs, prev = [], [], []
    for poly in polys:
        poly = orient(poly, sign=1.0)                       # exterior CCW, holes CW: drivable side on the left
        for ring in [poly.exterior, *poly.interiors]:
            n = max(int(ring.length), 3)
            p = shapely.get_coordinates(shapely.line_interpolate_point(ring, np.linspace(0.0, ring.length, n, endpoint=False)))
            d = np.roll(p, -1, 0) - p
            d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
            local = to_local(p, origin)
            inside = (np.abs(local) <= radius).all(-1)
            xy.append(local[inside])
            dirs.append(rotate_to_local(d, np.full(n, origin[2]))[inside])
            prev.append(rotate_to_local(np.roll(d, 1, 0), np.full(n, origin[2]))[inside])
    xy, dirs, prev = np.concatenate(xy), np.concatenate(dirs), np.concatenate(prev)
    if len(xy) > max_points:
        keep = np.sort(np.argsort(np.linalg.norm(xy, axis=-1), kind="stable")[:max_points])
        xy, dirs, prev = xy[keep], dirs[keep], prev[keep]
    return {"da_xy": xy, "da_dir": dirs, "da_prev_dir": prev}


def process_scenario(scenario, args) -> dict:
    ego0 = scenario.initial_ego_state
    origin = np.array([ego0.rear_axle.x, ego0.rear_axle.y, ego0.rear_axle.heading], dtype=np.float64)
    point = Point2D(origin[0], origin[1])
    ego = ego_arrays(scenario, origin)
    if ego is None:
        return None
    map_api = scenario.map_api
    route_ids = scenario.get_route_roadblock_ids()
    if route_ids != [""]:
        route_ids = route_roadblock_correction(ego0, map_api, route_ids)
    route_ids = [r for r in route_ids if r]
    tls = {str(t.lane_connector_id): TL_INDEX.get(t.status.name, TL_INDEX["UNKNOWN"])
           for t in scenario.get_traffic_light_status_at_iteration(0)}
    goal = scenario.get_mission_goal()
    goal_pose = np.zeros(3)
    if goal is not None:
        goal_pose = np.concatenate([to_local(np.array([goal.x, goal.y]), origin), [wrap(goal.heading - origin[2])]])

    data = {"log_name": scenario.log_name, "token": scenario.token, "map_name": scenario.map_api.map_name,
            "scenario_type": scenario.scenario_type, "timestamp_us": np.int64(scenario.start_time.time_us),
            "goal_pose": goal_pose, "goal_valid": np.bool_(goal is not None)}
    data.update(ego)
    data.update(object_arrays(scenario, origin, args.object_radius, args.max_objects))
    data.update(lane_arrays(map_api, point, origin, args.map_radius, args.max_lanes, set(route_ids), tls))
    data.update(polygon_arrays(map_api, point, origin, args.map_radius, args.max_polygons))
    data["route_xy"] = route_centerline(map_api, point, origin, route_ids, args.route_length_m)
    data.update(drivable_area_arrays(map_api, origin, args.da_radius, args.max_da_points))
    return {k: v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v for k, v in data.items()}


def cache_file(save_path: str, scenario) -> str:
    return os.path.join(save_path, f"{scenario.log_name}__{scenario.token}.npz")


_ARGS = None


def _init(args) -> None:
    global _ARGS
    _ARGS = args


def _work(scenario) -> str:
    path = cache_file(_ARGS.save_path, scenario)
    if os.path.exists(path):
        return "exists"
    try:
        data = process_scenario(scenario, _ARGS)
    except Exception:  # one bad scenario must not stop a long caching run
        print(f"FAILED {scenario.log_name} {scenario.token}\n{traceback.format_exc()}", flush=True)
        return "failed"
    if data is None:
        return "short"
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, **data)
    os.replace(tmp, path)
    return "written"


def main() -> None:
    args = parse_args()
    assert 0 <= args.shard_id < args.num_shards
    os.makedirs(args.save_path, exist_ok=True)
    log_names = None
    if args.log_names_json:
        with open(args.log_names_json) as fh:
            log_names = json.load(fh)
    db_files = None
    if args.max_db_files:
        wanted = set(log_names) if log_names else None
        db_files = [f for f in sorted(glob.glob(os.path.join(args.data_path, "*.db")))
                    if wanted is None or os.path.basename(f)[:-3] in wanted][:args.max_db_files]
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, db_files, args.map_version)
    scenario_filter = ScenarioFilter(
        scenario_types=None, scenario_tokens=None, log_names=log_names, map_names=None,
        num_scenarios_per_type=args.scenarios_per_type or None, limit_total_scenarios=args.total_scenarios or None,
        timestamp_threshold_s=None, ego_displacement_minimum_m=None, expand_scenarios=True,
        remove_invalid_goals=False, shuffle=True, ego_start_speed_threshold=None, ego_stop_speed_threshold=None,
        speed_noise_tolerance=None)
    random.seed(args.seed)
    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=args.num_workers)
    scenarios = sorted(builder.get_scenarios(scenario_filter, worker), key=lambda s: (s.log_name, s.token))
    shard = -(-len(scenarios) // args.num_shards)
    scenarios = scenarios[args.shard_id * shard:(args.shard_id + 1) * shard]
    print(f"{len(scenarios)} scenarios (shard {args.shard_id}/{args.num_shards})", flush=True)
    del worker

    t0, counts = time.time(), {}
    with Pool(args.num_workers, initializer=_init, initargs=(args,), maxtasksperchild=args.max_tasks_per_child) as pool:
        for i, status in enumerate(pool.imap_unordered(_work, scenarios, chunksize=1)):
            counts[status] = counts.get(status, 0) + 1
            if (i + 1) % 500 == 0 or i + 1 == len(scenarios):
                print(json.dumps({"done": i + 1, "of": len(scenarios), **counts,
                                  "s_per_scenario": round((time.time() - t0) / (i + 1), 3)}), flush=True)


if __name__ == "__main__":
    main()
