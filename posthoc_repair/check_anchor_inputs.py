"""Online AnchorWM inputs (anchor_forecast.online_scene) vs the cache's (build_cache.process_scenario).

For a few scenarios of a chunk filter (split_filter's tokens_logs.json), build both dicts at t = 0 -- the cache
from the DB as the training cache was built, the online one from the history buffer exactly as
``nuplan.planning.simulation.simulation.Simulation.initialize`` builds it (0.1 s scenarios, 2.1 s of past, then the
current state appended) -- and compare every model input of ``build_sample``. Then
run a random-init AnchorWM through ``AnchorForecaster`` -> ``forecast_boxes`` -> the repair (plumbing, shapes).

    PYTHONPATH=. python -m posthoc_repair.check_anchor_inputs --filters /data/user_data/$USER/nuplan_wm/filters/val14
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

from anchor_wm.build_cache import LogData, process_scenario
from anchor_wm.config import AnchorWMConfig
from anchor_wm.data import NOW, build_sample
from anchor_wm.model import AnchorWorldModel
from posthoc_repair.anchor_forecast import CACHE_ARGS, AnchorForecaster, online_scene
from posthoc_repair.optimizer import PosthocRepair
from posthoc_repair.scene import forecast_boxes

AGENT_INPUTS = {"agent_history", "agent_history_valid", "agent_valid", "object_types", "current_local_xy",
                "current_yaw_local"}
MODEL_INPUTS = ("agent_history", "agent_history_valid", "agent_valid", "object_types", "is_sdc", "current_local_xy",
                "current_yaw_local", "map_features", "map_valid", "ego_vel", "goal_xy", "goal_valid", "static_xy",
                "static_yaw", "static_lw", "static_type", "static_valid", "route_pts", "route_pts_valid")


def scenarios_of(tokens: list, log: str, data_root: str, map_root: str):
    # subsample 0.5 (10 Hz), as run_simulation's nuplan_scenario_mapping
    builder = NuPlanScenarioBuilder(data_root=data_root, map_root=map_root, sensor_root=data_root,
                                    db_files=[os.path.join(data_root, f"{log}.db")], map_version="nuplan-maps-v1.0",
                                    scenario_mapping=ScenarioMapping(scenario_map={}, subsample_ratio_override=0.5))
    f = ScenarioFilter(scenario_types=None, scenario_tokens=tokens, log_names=[log], map_names=None,
                       num_scenarios_per_type=None, limit_total_scenarios=None, timestamp_threshold_s=None,
                       ego_displacement_minimum_m=None, expand_scenarios=False, remove_invalid_goals=False,
                       shuffle=False)
    return builder.get_scenarios(f, Sequential())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--filters", required=True)
    p.add_argument("--chunk", default="chunk_000")
    p.add_argument("--scenarios", type=int, default=6)
    p.add_argument("--data_root", default="/data/datasets/nuPlan-v1.1/nuplan-v1.1/splits/trainval")
    p.add_argument("--map_root", default="/data/datasets/nuPlan-v1.1/maps")
    args = p.parse_args()

    chunk = yaml.safe_load(open(os.path.join(args.filters, "scenario_filter", f"{args.chunk}.yaml")))
    log_of = json.load(open(os.path.join(args.filters, "tokens_logs.json")))["log_of"]
    tokens = chunk["scenario_tokens"][:args.scenarios]
    by_log = {}
    for t in tokens:
        by_log.setdefault(log_of[t], []).append(t)

    cfg = AnchorWMConfig()
    cache_args = argparse.Namespace(**CACHE_ARGS, da_radius=150.0, max_da_points=8192)
    torch.manual_seed(0)
    model = AnchorWorldModel(cfg, torch.randn(3, cfg.num_anchors, cfg.future_steps, 2).cumsum(2)).eval()
    fc = AnchorForecaster(model=model)
    worst, seconds, ok = {}, [], True
    for log, toks in by_log.items():
        lg = LogData(os.path.join(args.data_root, f"{log}.db"), toks)
        for sc in scenarios_of(toks, log, args.data_root, args.map_root):
            cached = process_scenario(sc, lg, cache_args)
            if cached is None:
                print(f"{sc.token}: no full past / future in the log (the cache skips it)")
                continue
            # Simulation.initialize: buffer of int((2 + dt) / dt) + 1 samples, then the current state appended
            history = SimulationHistoryBuffer.initialize_from_scenario(22, sc, DetectionsTracks)
            history.append(sc.initial_ego_state, DetectionsTracks(sc.initial_tracked_objects))
            t0 = time.perf_counter()
            z = online_scene(history, list(sc.get_traffic_light_status_at_iteration(0)), sc.map_api,
                             sc.get_route_roadblock_ids(), sc.get_mission_goal())
            seconds.append(time.perf_counter() - t0)
            a, b = build_sample(cached, cfg, with_targets=False), build_sample(z, cfg, with_targets=False)
            row = []
            for k in MODEL_INPUTS:
                x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
                d = float(np.abs(x - y).max()) if x.shape == y.shape else float("inf")
                worst[k] = max(worst.get(k, 0.0), d)
                if d > 1e-4:
                    row.append(f"{k} {d:.3g}")
            # The cache keeps objects that appear later too; at its object cap they can displace agents present at
            # t = 0, which the online dict (past only) keeps. Explained if only agent inputs differ and online has more.
            def present(d):
                return int(((d["obj_type"] < 3) & d["obj_valid"][:, NOW]).sum())
            capped = (len(cached["obj_type"]) >= CACHE_ARGS["max_objects"] and present(z) > present(cached)
                      and all(r.split()[0] in AGENT_INPUTS for r in row))
            status = "match" if not row else (f"cap: cache {present(cached)} / online {present(z)} agents at t = 0"
                                              if capped else "DIFF " + ", ".join(row))
            print(f"{sc.token} ({sc.scenario_type}): {status}; "
                  f"{int(np.asarray(b['agent_valid']).sum())} agents, {int(np.asarray(b['static_valid']).sum())} "
                  f"statics, {len(z['lane_xy'])} lanes, online {1e3 * seconds[-1]:.0f} ms")
            ok &= not row or capped

            pred = fc.predict(z)
            agents = forecast_boxes({"xy": np.zeros((0, 2)), "yaw": np.zeros(0), "lw": np.zeros((0, 2)),
                                     "vel": np.zeros((0, 2)), "is_agent": np.zeros(0, bool)}, 20, 0.1, "cpu", pred)
            plan = torch.zeros(1, 81, 3)
            plan[0, :, 0] = torch.arange(81) * 0.5
            edges = {k: torch.zeros(1, 1, 2) for k in ("edge_xy", "edge_dir", "edge_prev_dir")}
            edges.update(edge_len=torch.zeros(1, 1), edge_prev_len=torch.zeros(1, 1),
                         edge_same_prev=torch.zeros(1, 1, dtype=torch.bool), edge_valid=torch.zeros(1, 1, dtype=torch.bool))
            out, info = PosthocRepair()(plan, edges, agents)
            w = agents["weight"][0, 0].numpy()
            print(f"   forecast: {pred['xy'].shape} modes, prob sums {np.unique(pred['prob'].sum(1).round(4))}, "
                  f"repair boxes {tuple(agents['box'].shape)} weight sum {w.sum():.2f}, repair finite "
                  f"{bool(torch.isfinite(out).all())}")
    print("worst |online - cache| per input:", {k: round(v, 6) for k, v in worst.items()})
    print(f"online_scene {1e3 * np.median(seconds):.0f} ms median")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
