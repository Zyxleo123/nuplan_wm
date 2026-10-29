"""Behavioural gates for posthoc_repair on a synthetic straight road (no data needed, CPU, seconds).

The drivable area goes through the cache's own extraction (``drivable_area_arrays`` on a stub map API), so
the edge sign / tolerance convention is the one the planner sees.

    PYTHONPATH=. python -m posthoc_repair.check
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import box

from posthoc_repair.optimizer import PosthocRepair, RepairConfig
from posthoc_repair.scene import DrivableArea, forecast_boxes

HALF_ROAD = 5.5


class _StubMap:
    map_name = "stub"

    def __init__(self) -> None:
        self._layer = gpd.GeoDataFrame(geometry=[box(-100.0, -HALF_ROAD, 400.0, HALF_ROAD)])

    def _get_vector_map_layer(self, layer):
        return self._layer


def _plan(speed: float, lateral=None, W: int = 81) -> np.ndarray:
    x = speed * 0.1 * np.arange(W)
    y = np.zeros(W) if lateral is None else lateral(np.arange(W))
    yaw = np.concatenate([[0.0], np.arctan2(np.diff(y), np.maximum(np.diff(x), 1e-6))])
    return np.stack([x, y, yaw], -1)


def _objects(rows) -> dict:
    a = np.asarray(rows, np.float64).reshape(-1, 7)             # x, y, yaw, length, width, vx, vy
    return {"xy": a[:, :2], "yaw": a[:, 2], "lw": a[:, 3:5], "vel": a[:, 5:7], "is_agent": np.ones(len(a), bool)}


def run(plan: np.ndarray, rows=(), cfg: RepairConfig | None = None):
    rep = PosthocRepair(cfg)
    edges = DrivableArea()(_StubMap(), np.zeros(3), "cpu")
    agents = forecast_boxes(_objects(rows), rep.cfg.window, rep.cfg.dt, "cpu")
    out, info = rep(torch.as_tensor(plan, dtype=torch.float32)[None], edges, agents)
    return out[0].numpy(), {k: (v[0].item() if torch.is_tensor(v) else v) for k, v in info.items()}


def main() -> None:
    fails = []

    def gate(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            fails.append(name)

    # 1. A clean plan (lane centre, nothing near) is returned bit-identical.
    plan = _plan(10.0)
    out, info = run(plan, [(60.0, -3.5, 0.0, 4.5, 1.9, 10.0, 0.0)])
    gate("clean plan untouched", info["loss0"] == 0.0 and not info["moved"]
         and np.array_equal(out, plan.astype(np.float32)), f"loss0 {info['loss0']:.3g} moved {info['moved']}")

    # 2. A plan drifting off the road (a corner ~0.65 m outside by step 20; within o_am's 2.9 m reach) is
    # pulled back inside.
    plan = _plan(10.0, lambda t: np.minimum(t, 20) / 20 * (HALF_ROAD - 0.5))
    out, info = run(plan)
    gate("off-road plan repaired", info["c_edge0"] < 0 <= info["c_edge"] and info["moved"],
         f"c_edge {info['c_edge0']:+.2f} -> {info['c_edge']:+.2f} m, shift {info['shift_m']:.2f} m")

    # 3. A parked car ahead, overlapping the ego lane by 1.3 m and hit inside the window: no overlap after.
    plan = _plan(8.0)
    out, info = run(plan, [(18.0, 0.8, 0.0, 4.6, 1.9, 0.0, 0.0)])
    gate("parked car avoided", info["c_agent0"] < 0 < info["c_agent"] and info["c_edge"] > 0,
         f"c_agent {info['c_agent0']:+.2f} -> {info['c_agent']:+.2f} m, c_edge {info['c_edge']:+.2f}, "
         f"max lateral {np.abs(out[:21, 1]).max():.2f} m")
    turn = np.abs(np.diff(out[:22, 2])).max() / 0.1
    gate("repair is smooth", turn < 1.0, f"max yaw rate over the window {turn:.2f} rad/s")

    # 4. A stopped plan next to a box: finite, and the ego is not pushed through anything.
    plan = _plan(0.0)
    out, info = run(plan, [(4.5, 2.4, 0.0, 4.6, 1.9, 0.0, 0.0)])
    gate("stopped plan finite", np.isfinite(out).all() and not info["nonfinite"],
         f"loss {info['loss0']:.3g} -> {info['loss']:.3g}, shift {info['shift_m']:.2f} m")

    # 5. The repaired tail keeps the window's lateral offset and continues the along-path speed change.
    plan = _plan(8.0)
    out, info = run(plan, [(18.0, 0.8, 0.0, 4.6, 1.9, 0.0, 0.0)])
    dy = out[21:, 1] - out[20, 1]
    gate("tail holds the offset", np.abs(dy).max() < 0.05, f"tail drift {np.abs(dy).max():.3f} m")

    print("ALL PASS" if not fails else f"FAILED: {fails}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
