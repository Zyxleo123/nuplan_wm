"""Gates for the repair's TTC term and ``agent_behind_deg`` (no data needed, CPU, seconds).

* the defaults (``ttc_weight: 0``, ``agent_behind_deg: 180``) repair random scenes bit-identically to ``--reference``,
  a copy of the optimizer from before these terms (e.g. a code snapshot's ``posthoc_repair/optimizer.py``);
* on check.py's straight road, nuPlan's TTC rule (the ego at its speed along its heading and every agent ahead at
  its velocity, shapely boxes, 0.1 s steps; a failure is a hit within 0.95 s) computed independently of the repair's
  own geometry: a plan braking late for a stopped car fails it before the repair and passes after; a lead car at
  the plan's speed, a stopped car in the next lane and oncoming traffic in the next lane are left alone; a car
  closing from behind charges the agent barrier by default and not with ``agent_behind_deg: 150``.

    PYTHONPATH=. python -m posthoc_repair.check_ttc --reference <snapshot>/posthoc_repair/optimizer.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import numpy as np
import torch
from shapely.affinity import rotate, translate
from shapely.geometry import box as sbox

from anchor_wm.config import EGO_LENGTH, EGO_REAR_TO_CENTER, EGO_WIDTH
from posthoc_repair.check import _StubMap, _objects, _plan
from posthoc_repair.optimizer import PosthocRepair, RepairConfig
from posthoc_repair.scene import DrivableArea, forecast_boxes


def _poly(x, y, yaw, length, width):
    return translate(rotate(sbox(-length / 2, -width / 2, length / 2, width / 2), yaw, use_radians=True), x, y)


def nuplan_ttc(plan: np.ndarray, rows, steps: int = 20) -> float:
    """Min TTC [s] over plan steps 1..steps under nuPlan's rule (inf when nothing is hit within 3 s)."""
    a = np.asarray(rows, np.float64).reshape(-1, 7)
    best = np.inf
    for t in range(1, steps + 1):
        x, y, yaw = plan[t]
        v = np.hypot(*(plan[t + 1, :2] - plan[t - 1, :2])) / 0.2
        if v <= 5e-3:
            continue
        cx, cy = x + EGO_REAR_TO_CENTER * np.cos(yaw), y + EGO_REAR_TO_CENTER * np.sin(yaw)
        for r in a:
            ox, oy = r[0] + r[5] * 0.1 * t, r[1] + r[6] * 0.1 * t
            rel = np.arctan2(oy - y, ox - x) - yaw
            if abs(np.arctan2(np.sin(rel), np.cos(rel))) >= np.deg2rad(30):
                continue
            for tau in np.arange(0.1, 3.0, 0.1):
                e = _poly(cx + v * tau * np.cos(yaw), cy + v * tau * np.sin(yaw), yaw, EGO_LENGTH, EGO_WIDTH)
                o = _poly(ox + r[5] * tau, oy + r[6] * tau, r[2], r[3], r[4])
                if e.intersects(o):
                    best = min(best, tau)
                    break
    return best


def run(plan, rows=(), cfg=None, optimizer=PosthocRepair):
    rep = optimizer(cfg)
    edges = DrivableArea()(_StubMap(), np.zeros(3), "cpu")
    agents = forecast_boxes(_objects(rows), rep.cfg.window, rep.cfg.dt, "cpu")
    out, info = rep(torch.as_tensor(plan, dtype=torch.float32)[None], edges, agents)
    return out[0].numpy().astype(np.float64), {k: (v[0].item() if torch.is_tensor(v) and v.dim() else v) for k, v in info.items()}


def braking(v0: float, decel: float, W: int = 81) -> np.ndarray:
    t = 0.1 * np.arange(W)
    t_stop = v0 / decel
    x = np.where(t < t_stop, v0 * t - 0.5 * decel * t * t, v0 * t_stop - 0.5 * decel * t_stop ** 2)
    return np.stack([x, np.zeros(W), np.zeros(W)], -1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="", help="optimizer.py from before the TTC term (bit-identity gate)")
    args = p.parse_args()
    fails = []

    def gate(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            fails.append(name)

    if args.reference:
        spec = importlib.util.spec_from_file_location("reference_optimizer", args.reference)
        ref = importlib.util.module_from_spec(spec)
        sys.modules["reference_optimizer"] = ref          # dataclasses look their module up
        spec.loader.exec_module(ref)
        rng = np.random.default_rng(0)
        worst, moved = 0.0, 0
        for i in range(40):
            v = rng.uniform(0, 12)
            plan = _plan(v, lambda t, a=rng.uniform(-3, 3): a * np.minimum(t, 30) / 30)
            rows = [(rng.uniform(-10, 40), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi), 4.6, 1.9,
                     rng.uniform(-8, 8), rng.uniform(-2, 2)) for _ in range(int(rng.integers(0, 8)))]
            out_new, info_new = run(plan, rows)
            rep = ref.PosthocRepair(None)
            edges = DrivableArea()(_StubMap(), np.zeros(3), "cpu")
            agents = forecast_boxes(_objects(rows), rep.cfg.window, rep.cfg.dt, "cpu")
            agents = {k: v for k, v in agents.items() if k != "vel"}
            out_ref, _ = rep(torch.as_tensor(plan, dtype=torch.float32)[None], edges, agents)
            worst = max(worst, float(np.abs(out_new - out_ref[0].numpy().astype(np.float64)).max()))
            moved += int(info_new["moved"])
        gate("defaults == reference", worst == 0.0, f"40 random scenes ({moved} repaired), max |diff| {worst:.3g} m")

    on = RepairConfig(ttc_weight=1.0)
    # 1. 5 m/s, a stopped car 6 m ahead of the bumper, the plan braking at 2.5 m/s^2 to stop 1 m short: clear of the
    # agent margin, but the TTC rule fails it mid-brake. The repair brakes earlier.
    car = [(4.049 + 6.0 + 2.3, 0.0, 0.0, 4.6, 1.9, 0.0, 0.0)]
    plan = braking(5.0, 2.5)
    out0, info0 = run(plan, car)
    out1, info1 = run(plan, car, on)
    t0, t1 = nuplan_ttc(plan, car), nuplan_ttc(out1, car)
    gate("late brake: TTC fails before", info0["c_agent0"] > 0.75 and t0 < 0.95,
         f"min gap {info0['c_agent0']:.2f} m, default repair moved {info0['moved']}, nuPlan TTC {t0:.1f} s")
    gate("late brake: TTC passes after", t1 > 0.95 and info1["moved"] and info1["c_ttc"] > 0,
         f"TTC {t0:.1f} -> {t1:.1f} s, c_ttc {info1['c_ttc0']:+.2f} -> {info1['c_ttc']:+.2f} m, "
         f"speed at 1 s {np.hypot(*(plan[11, :2] - plan[9, :2])) / 0.2:.2f} -> {np.hypot(*(out1[11, :2] - out1[9, :2])) / 0.2:.2f} m/s")
    lat = np.abs(out1[:21, 1]).max()
    gate("late brake: fixed by braking", lat < 0.3 and out1[20, 0] < plan[20, 0], f"max lateral {lat:.2f} m, "
         f"x at 2 s {plan[20, 0]:.2f} -> {out1[20, 0]:.2f} m")

    # 2. Cases the TTC rule passes, so the term must leave them alone (bit-identical, nothing charged).
    for name, plan, rows in [
        ("lead car at the plan's speed, 3 m gap", _plan(8.0), [(4.049 + 3.0 + 2.3, 0.0, 0.0, 4.6, 1.9, 8.0, 0.0)]),
        ("stopped car in the next lane", _plan(10.0), [(20.0, 3.5, 0.0, 4.6, 1.9, 0.0, 0.0)]),
        ("oncoming car in the next lane", _plan(10.0), [(30.0, 3.5, np.pi, 4.6, 1.9, -10.0, 0.0)]),
    ]:
        out, info = run(plan, rows, on)
        gate(f"untouched: {name}", info["loss0"] == 0.0 and np.array_equal(out, plan.astype(np.float32).astype(np.float64)),
             f"loss0 {info['loss0']:.3g}, nuPlan TTC {nuplan_ttc(plan, rows):.1f} s")

    # 3. A car closing fast from behind: charged by default, not with agent_behind_deg 150 (not at fault in nuPlan).
    plan, rows = _plan(5.0), [(-6.0, 0.0, 0.0, 4.6, 1.9, 12.0, 0.0)]
    _, a = run(plan, rows)
    _, b = run(plan, rows, RepairConfig(agent_behind_deg=150.0))
    gate("rear car: charged by default", a["loss0"] > 0, f"loss0 {a['loss0']:.3g}")
    gate("rear car: ignored behind 150 deg", b["loss0"] == 0.0 and not b["moved"], f"loss0 {b['loss0']:.3g}")

    # 4. A stopped plan behind a stopped car, inside the TTC margin: finite.
    out, info = run(_plan(0.0), [(4.049 + 0.2 + 2.3, 0.0, 0.0, 4.6, 1.9, 0.0, 0.0)], on)
    gate("stopped plan finite", np.isfinite(out).all() and not info["nonfinite"],
         f"loss {info['loss0']:.3g} -> {info['loss']:.3g}, shift {info['shift_m']:.2f} m")

    # 5. speed_weight: a plan at 12 m/s under a 10 m/s limit is slowed to it (plan-space and tracked); one at 8 m/s is
    # left alone.
    for track in (False, True):
        cfg = RepairConfig(speed_weight=1.0, track=track)
        rep = PosthocRepair(cfg)
        edges = DrivableArea()(_StubMap(), np.zeros(3), "cpu")
        agents = forecast_boxes(_objects(()), 20, 0.1, "cpu")
        for v0, want_moved in ((12.0, True), (8.0, False)):
            ego = {"v": torch.tensor([v0]), "steer": torch.tensor([0.0]), "accel": torch.tensor([0.0]), "limit": torch.tensor([10.0])}
            plan = torch.as_tensor(_plan(v0), dtype=torch.float32)[None]
            out, info = rep(plan, edges, agents, ego)
            window, speed = rep.scored(out, 20, ego)
            ok = bool(info["moved"][0]) == want_moved and (not want_moved or float(speed[0, -5:].mean()) < v0 - 0.5)
            gate(f"speed limit 10 m/s, plan {v0:.0f} m/s, track={track}", ok,
                 f"moved {bool(info['moved'][0])}, scored speed at 1.5-2 s {float(speed[0, -5:].mean()):.2f} m/s")

    print("ALL PASS" if not fails else f"FAILED: {fails}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
