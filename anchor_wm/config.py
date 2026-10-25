from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Object types in the cache (`obj_type`): 0-2 dynamic agents, 3-6 static objects.
OBJ_TYPES = ("vehicle", "pedestrian", "bicycle", "czone_sign", "barrier", "traffic_cone", "generic_object")
DYNAMIC_TYPES = (0, 1, 2)
# Anchor sets are per group; the ego uses the vehicle set.
ANCHOR_GROUPS = ("vehicle", "pedestrian", "bicycle")
# Map polyline types (`map_type` one-hot): lanes / lane connectors from the lane layers, then polygons.
MAP_TYPES = ("lane", "lane_connector", "crosswalk", "stop_line")
TL_STATES = ("green", "yellow", "red", "unknown")
SUB_SCORES = ("overlap", "offroad", "progress", "comfort")

# Pacifica (nuplan get_pacifica_parameters): the ego frame origin is the rear axle, the box centre is
# 1.461 m ahead of it.
EGO_LENGTH, EGO_WIDTH, EGO_REAR_TO_CENTER = 5.176, 2.297, 1.461

# Cache time grid (0.1 s): PAST_FRAMES frames up to and including t = 0, then FUTURE_FRAMES frames.
CACHE_DT = 0.1
PAST_FRAMES = 21
FUTURE_FRAMES = 80


@dataclass
class AnchorWMConfig:
    # ---- data / encoder ----
    history_steps: int = 21          # model history: 21 frames 0.1 s apart (2 s) ending at t = 0
    history_stride: int = 1          # in cache frames (0.1 s)
    future_steps: int = 80           # 80 steps 0.1 s apart (8 s), nuPlan's usual planning output
    future_stride: int = 1
    max_agents: int = 64             # ego (slot 0) + nearest dynamic agents valid at t = 0
    max_range: float = 50.0          # position normalisation (m)
    max_velocity: float = 25.0
    max_width: float = 10.0
    max_static: int = 32             # static objects (cones, barriers, ...) present at t = 0, nearest first
    max_lanes: int = 96              # lanes + lane connectors, nearest first
    max_polygons: int = 32           # crosswalks + stop lines, nearest first
    map_points: int = 20             # points per map polyline (as cached)
    max_label_objects: int = 192     # log-replay boxes for overlap labels (every object type)
    hidden_dim: int = 256
    num_heads: int = 4
    num_scene_layers: int = 3
    dropout: float = 0.1
    route_points: int = 10           # route summary for the ego query / scorer: centreline every 10 m
    route_spacing_m: float = 10.0
    use_goal: bool = False           # also feed the mission goal (off by default: the route is the subgoal)

    # ---- anchor decoder ----
    num_anchors: int = 64            # per anchor group; must match the anchor file
    num_out_modes: int = 6
    nms_dist_m: float = 2.0
    sigma_eps: float = 0.002         # sigma floor, normalised by max_range (0.1 m)
    sigma_raw_bias: float = -4.0
    yaw_loss_weight: float = 0.5
    cls_weight: float = 1.0
    cls_soft_temp_m: float = 1.0     # soft anchor-classification target exp(-ade / temp); 0 = one-hot
    collision_focus_weight: float = 2.0
    interaction_tau_m: float = 5.0   # per-agent weight 1 + w * exp(-d / tau), d = closest approach to the ego
    ego_loss_weight: float = 1.0

    # ---- label geometry (online, log replay) ----
    num_label_edges: int = 3072      # drivable-area boundary points kept for offroad labels
    edge_box: tuple = (-30.0, 180.0, 70.0)    # behind, ahead, half-width (m) in the ego frame
    num_edge_tokens: int = 256       # scorer boundary tokens (strided subset of the label edges)
    edge_token_box: tuple = (-25.0, 120.0, 40.0)
    route_label_points: int = 400    # route centreline (1 m spacing) for progress labels / route features

    # ---- scorer ----
    scorer_hidden: int = 128
    scorer_heads: int = 4
    scorer_agents: int = 32
    scorer_modes: int = 6
    # The `scorer_near_agents` nearest selected agents get `scorer_near_modes` modes, the rest
    # `scorer_modes` (0 = every agent gets `scorer_modes`).
    scorer_near_agents: int = 8
    scorer_near_modes: int = 12
    scorer_pos_scale: float = 20.0
    scorer_vel_scale: float = 5.0
    scorer_route_features: bool = True    # per-step distance / heading error to the route + endpoint progress
    scorer_rule_features: bool = True     # exact box gaps to the forecast modes + signed drivable-area distance
    scorer_aux_heads: bool = True         # time-to-violation and route-progress heads

    def __post_init__(self) -> None:
        self.edge_box = tuple(float(v) for v in self.edge_box)
        self.edge_token_box = tuple(float(v) for v in self.edge_token_box)
        if self.scorer_hidden % self.scorer_heads:
            raise ValueError("scorer_hidden must be divisible by scorer_heads")
        if self.num_out_modes > self.num_anchors or self.scorer_modes > self.num_anchors:
            raise ValueError("num_out_modes and scorer_modes must be <= num_anchors")
        if self.scorer_near_agents > 0 and not self.scorer_modes <= self.scorer_near_modes <= self.num_anchors:
            raise ValueError("scorer_near_modes must be in [scorer_modes, num_anchors]")
        if (self.history_steps - 1) * self.history_stride >= PAST_FRAMES:
            raise ValueError("history does not fit the cached past")
        if self.future_steps * self.future_stride > FUTURE_FRAMES:
            raise ValueError("future does not fit the cached future")

    @property
    def scorer_modes_max(self) -> int:
        return max(int(self.scorer_modes), int(self.scorer_near_modes) if self.scorer_near_agents > 0 else 0)

    @property
    def agent_feature_dim(self) -> int:
        return 8 + 1 + len(ANCHOR_GROUPS)       # pose / velocity / size, is-ego, type one-hot

    @property
    def static_feature_dim(self) -> int:
        return 6 + len(OBJ_TYPES) - len(ANCHOR_GROUPS)   # xy, sin / cos, size, type one-hot

    @property
    def map_feature_dim(self) -> int:
        # xy, direction, left / right boundary offsets, type, on-route, traffic light, speed limit (+ has)
        return 2 + 2 + 4 + len(MAP_TYPES) + 1 + len(TL_STATES) + 2

    @property
    def future_dt(self) -> float:
        return float(self.future_stride) * CACHE_DT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnchorWMConfig":
        return cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})
