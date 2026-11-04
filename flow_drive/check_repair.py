"""FlowDrive and the repair on diffusion_es.check's synthetic road (no nuPlan data or devkit; CPU, ~a minute).

The road: 3 lanes, the ego in the middle one at 8 m/s, a car 25 m ahead in the same lane at 6 m/s. FlowDrive is
the released checkpoint (``--ckpt``). Its lanes are 25 m map elements (20 points, ~1.3 m apart, the middle lane on
route, as ``map_process`` marks it), not ``dp_inputs``'s 100 m polylines with 5 m spacing: on those FlowDrive's
samples scatter (x -20..26 m, y 2..14 m at 4 s over 8 seeds) while Diffusion Planner stays on its lane. Gates:

1. FlowDrive's plans on the road are sane: all finite with 40 poses, and at least 90% of the seeds make 15-45 m
   of progress in 4 s, stay on the road and head along it (real inputs, so a wrong input convention shows here;
   ``check.py`` only uses random ones). Not every seed: plain FlowDrive's samples spread (2.2 m on average over
   32 seeds), and on this bare road (12 lane elements) one in 32 ended 6.8 m left, past the edge (job 92424);
   [info] how much the plan varies with the seed (the flow model's own spread);
2. the repair takes FlowDrive's 41-row plans, batched over seeds: a plan clear of every margin comes back
   unchanged, and the clean plans pushed 6 m left over the repair's window (through the road edge) are all
   moved back inward.

    PYTHONPATH=. python -m flow_drive.check_repair --ckpt flow_drive/checkpoint/flow_drive_model.pth
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from diffusion_es.check import HALF_ROAD, LANES_Y, dp_inputs, edges_of, lead_boxes, road_scene
from diffusion_es.es import spread_m
from flow_drive.model.flow_drive_planner import FlowDrivePlanner, load_params
from posthoc_repair.optimizer import PosthocRepair


def fd_inputs(params, device, segment_m: float = 25.0) -> dict:
    """``diffusion_es.check.dp_inputs`` (FlowDrive reads the same features), its lanes cut into ``segment_m`` map
    elements of ``lane_len`` points (``map_process``'s layout: x, y, step to the next point (0 at the last),
    offsets to the left / right boundary, traffic light one-hot) + ``lanes_is_route`` (the middle lane)."""
    dp, enc = params.data_processing, params.encoder
    cfg = SimpleNamespace(lane_len=dp.lane_len, lane_num=dp.lane_num, lane_state_dim=12, route_num=dp.route_num,
                          route_len=dp.route_len, route_state_dim=12, agent_num=dp.agent_num, time_len=enc.time_len,
                          agent_state_dim=11, static_objects_num=dp.static_objects_num,
                          static_objects_state_dim=enc.static_objects_state_dim)
    x = dp_inputs(cfg, device)
    lanes = torch.zeros_like(x["lanes"])
    speed, has_speed = torch.zeros_like(x["lanes_speed_limit"]), torch.zeros_like(x["lanes_has_speed_limit"])
    route = torch.zeros(1, dp.lane_num, 1, dtype=torch.int64, device=device)
    starts = torch.arange(-10.0, 90.0, segment_m).tolist()
    for i, (y, x0) in enumerate((y, x0) for y in LANES_Y for x0 in starts):
        xs = torch.linspace(x0, x0 + segment_m, dp.lane_len, device=device)
        lanes[0, i, :, 0], lanes[0, i, :, 1] = xs, y
        lanes[0, i, :-1, 2] = xs[1:] - xs[:-1]
        lanes[0, i, :, 5], lanes[0, i, :, 7] = 1.75, -1.75
        lanes[0, i, :, 11] = 1.0                         # traffic light unknown
        speed[0, i], has_speed[0, i], route[0, i] = 15.0, True, int(y == 0.0)
    x.update(lanes=lanes, lanes_speed_limit=speed, lanes_has_speed_limit=has_speed, lanes_is_route=route)
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, default=32)
    args = p.parse_args()
    dev = torch.device(args.device)
    ok = True

    def gate(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    params = load_params()
    model = FlowDrivePlanner(params, args.ckpt, device=str(dev))
    x = fd_inputs(params, dev)

    # 1. FlowDrive on the road
    plans = []
    for s in range(args.seeds):
        torch.manual_seed(520 + s)
        pred = model(x)[0]                                                               # [T, 3]
        plans.append(torch.cat([torch.zeros(1, 3, device=dev), pred], 0))
    plans = torch.stack(plans).clone()                                                   # [S, 1 + T, 3]
    end = plans[:, -1]
    prog, y_end, yaw_end = end[:, 0], end[:, 1], end[:, 2]
    sane = (prog > 15) & (prog < 45) & (y_end.abs() < HALF_ROAD - 1.0) & (yaw_end.abs() < 0.3)
    gate("FlowDrive's plans on the road are sane",
         bool(torch.isfinite(plans).all()) and plans.shape[1] == params.diffuser.pred_horizon + 1
         and float(sane.float().mean()) >= 0.9,
         f"{int(sane.sum())} of {args.seeds} seeds; at 4 s: x {prog.min():.1f}..{prog.max():.1f} m (8 m/s holds "
         f"32 m, the lead car is at {25 + 6 * 4:.0f} m), y {y_end.min():.2f}..{y_end.max():.2f} m, yaw "
         f"{yaw_end.abs().max():.3f} rad; outside: "
         + (", ".join(f"seed {520 + i} ({prog[i]:.1f}, {y_end[i]:.2f}) m" for i in (~sane).nonzero()[:, 0].tolist())
            or "none"))
    s_mean, s_end = spread_m(plans)
    print(f"[info] spread over {args.seeds} seeds: mean |plan - mean plan| {s_mean:.3f} m, "
          f"max endpoint deviation {s_end:.3f} m")

    # 2. the repair on FlowDrive's plans
    repair = PosthocRepair()
    n = repair.cfg.window
    edges, agents = edges_of(road_scene(), dev), lead_boxes(n, dev)
    k = len(plans)
    batch = lambda d: {key: v.expand(k, *v.shape[1:]) for key, v in d.items()}  # noqa: E731
    out, info = repair(plans, batch(edges), batch(agents))
    clean = info["loss0"] == 0
    gate("a clean plan comes back unchanged",
         bool(clean.any()) and torch.equal(out[clean], plans[clean]) and not bool(info["moved"][clean].any()),
         f"{int(clean.sum())} of {k} plans clear of every margin, all returned bit-identical")
    ramp = (torch.arange(1, plans.shape[1], device=dev, dtype=plans.dtype) / n).clamp(max=1.0)
    drift = plans[clean].clone()
    drift[:, 1:, 1] += 6.0 * ramp
    k = len(drift)
    out, info = repair(drift, batch(edges), batch(agents))
    gate("clean plans pushed through the road edge are moved back",
         bool(info["moved"].all()) and bool((info["c_edge"] > info["c_edge0"]).all())
         and bool((out[:, n, 1] < drift[:, n, 1]).all()),
         f"{k} plans; edge clearance (m) {info['c_edge0'].min():.2f}..{info['c_edge0'].max():.2f} -> "
         f"{info['c_edge'].min():.2f}..{info['c_edge'].max():.2f}; y at the window's end "
         f"{drift[:, n, 1].mean():.2f} -> {out[:, n, 1].mean():.2f} m (mean)")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
