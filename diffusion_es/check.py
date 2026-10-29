"""Gates for diffusion_es on a synthetic straight road (no nuPlan data or devkit needed; CPU or GPU, ~a minute).

Diffusion Planner is the released checkpoint (``--dp_dir`` with ``args.json`` / ``model.pth``); AnchorWM is a
random init (its scores are noise: these gates check plumbing and invariances, not quality).

1. population sampling == Diffusion Planner's own decoder for the same seed (K = 1, bit-identical);
   [info] the released model's population spread on this road, per ``noise_scale``;
2. re-noising: the children's distance from their proposal grows with ``resample_t``;
3. the ES climbs a known objective (prefer the left lane): the best score never drops over the rounds and ends
   above the first population's best;
   2 and 3 test the search mechanics, so their denoiser is an echo (x0 = x_t, the released encoder kept): the
   released DiT is near-deterministic on this road (the [info] line), and a population that does not vary gives
   a search nothing to climb;
4. the AnchorWM scorer is batch-consistent: permuting / chunking the candidates permutes / keeps the scores;
5. the batched repair (edges / boxes expanded over the population, as the planner calls it) equals the repair of
   each plan alone, in float64 (in float32 the 32 iterations amplify rounding through the piecewise loss to
   decimetres: reported, not gated);
6. one full replan (population -> repair -> AnchorWM score -> ES) runs finite, with the per-stage times.

    PYTHONPATH=. python -m diffusion_es.check --dp_dir /data/user_data/$USER/nuplan_wm/checkpoints/diffusion_planner_release

``--write_random_anchor_ckpt PATH`` writes a random-init AnchorWM checkpoint (``anchor_wm.train``'s format) and exits:
closed-loop plumbing runs only, until a trained scorer is readable (its scores are noise).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from anchor_wm.config import AnchorWMConfig
from anchor_wm.model import AnchorWorldModel
from diffusion_es.es import ESConfig, PopulationSampler, diffusion_es, plans_of, spread_m
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from posthoc_repair.anchor_model import AnchorWM
from posthoc_repair.optimizer import PosthocRepair

LANES_Y = (-3.5, 0.0, 3.5)
HALF_ROAD = 5.25
EGO_SPEED, LEAD_X, LEAD_SPEED = 8.0, 25.0, 6.0


class _EchoDiT(torch.nn.Module):
    """An ``x_start`` denoiser that predicts x0 = x_t."""
    model_type = "x_start"

    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask):
        return x


def load_dp(dp_dir: str, device) -> tuple[Diffusion_Planner, Config]:
    cfg = Config(os.path.join(dp_dir, "args.json"), None)
    model = Diffusion_Planner(cfg)
    sd = torch.load(os.path.join(dp_dir, "model.pth"), map_location="cpu")
    sd = sd["ema_state_dict"] if "ema_state_dict" in sd else sd.get("model", sd)
    model.load_state_dict({k[len("module."):]: v for k, v in sd.items() if k.startswith("module.")})
    return model.to(device).eval(), cfg


def dp_inputs(cfg: Config, device) -> dict:
    """Diffusion Planner's raw inputs (data_process conventions) for a 3-lane road, ego in the middle lane at
    8 m/s, one car 25 m ahead in the same lane at 6 m/s."""
    x = np.linspace(-10.0, 90.0, cfg.lane_len)
    lane = lambda y: np.stack([x, np.full_like(x, y), np.r_[np.diff(x), np.diff(x)[-1:]], np.zeros_like(x),  # noqa: E731
                               np.zeros_like(x), np.full_like(x, 1.75), np.zeros_like(x), np.full_like(x, -1.75),
                               np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.ones_like(x)], -1)
    lanes = np.zeros((cfg.lane_num, cfg.lane_len, cfg.lane_state_dim))
    for i, y in enumerate(LANES_Y):
        lanes[i] = lane(y)
    route = np.zeros((cfg.route_num, cfg.route_len, cfg.route_state_dim))
    route[0] = lane(0.0)
    nb = np.zeros((cfg.agent_num, cfg.time_len, cfg.agent_state_dim))
    t = np.arange(-(cfg.time_len - 1), 1) * 0.1
    nb[0] = np.stack([LEAD_X + LEAD_SPEED * t, np.zeros_like(t), np.ones_like(t), np.zeros_like(t),
                      np.full_like(t, LEAD_SPEED), np.zeros_like(t), np.full_like(t, 2.0), np.full_like(t, 4.8),
                      np.ones_like(t), np.zeros_like(t), np.zeros_like(t)], -1)
    sl = np.zeros((cfg.lane_num, 1))
    sl[:3] = 15.0
    raw = {"ego_current_state": np.array([[0, 0, 1, 0, EGO_SPEED, 0, 0, 0, 0, 0]], float)[0],
           "neighbor_agents_past": nb, "static_objects": np.zeros((cfg.static_objects_num, cfg.static_objects_state_dim)),
           "lanes": lanes, "lanes_speed_limit": sl, "lanes_has_speed_limit": sl > 0,
           "route_lanes": route, "route_lanes_speed_limit": np.full((cfg.route_num, 1), 15.0),
           "route_lanes_has_speed_limit": np.ones((cfg.route_num, 1), bool)}
    return {k: torch.as_tensor(v, dtype=torch.bool if v.dtype == bool else torch.float32)[None].to(device)
            for k, v in raw.items()}


def ring(pts: np.ndarray) -> dict:
    """A closed CCW ring (drivable side on the left) as cache ``da_*`` arrays."""
    seg = np.roll(pts, -1, 0) - pts
    ln = np.linalg.norm(seg, axis=-1)
    d = seg / ln[:, None]
    return {"da_xy": pts, "da_dir": d, "da_prev_dir": np.roll(d, 1, 0), "da_len": ln, "da_prev_len": np.roll(ln, 1)}


def road_scene() -> dict:
    """The same road as an AnchorWM cache dict (anchor_forecast.online_scene's keys; past frames only)."""
    frames, now = 101, 20
    t = (np.arange(frames) - now) * 0.1
    past = np.arange(frames) <= now
    ego_pose = np.stack([EGO_SPEED * t, np.zeros(frames), np.zeros(frames)], -1)
    obj_pose = np.stack([LEAD_X + LEAD_SPEED * t, np.zeros(frames), np.zeros(frames)], -1)[None]
    xs = np.linspace(-60.0, 200.0, 20)
    lanes = np.stack([np.stack([xs, np.full_like(xs, y)], -1) for y in LANES_Y])
    xb = np.arange(-60.0, 200.0, 1.0)
    pts = np.concatenate([np.stack([xb, np.full_like(xb, -HALF_ROAD)], -1),
                          np.stack([xb[::-1], np.full_like(xb, HALF_ROAD)], -1)])
    z = {"ego_pose": ego_pose, "ego_vel": np.tile([EGO_SPEED, 0.0], (frames, 1)), "ego_acc": np.zeros((frames, 2)),
         "obj_type": np.array([0]), "obj_valid": past[None], "obj_pose": obj_pose,
         "obj_vel": np.tile([LEAD_SPEED, 0.0], (1, frames, 1)), "obj_lw": np.array([[4.8, 2.0]]),
         "lane_xy": lanes, "lane_left": lanes + [0, 1.75], "lane_right": lanes - [0, 1.75],
         "lane_type": np.zeros(3), "lane_on_route": np.array([0.0, 1.0, 0.0]), "lane_tl": np.full(3, 3),
         "lane_speed_limit": np.full(3, 15.0), "lane_has_speed_limit": np.ones(3),
         "poly_xy": np.zeros((0, 20, 2)), "poly_type": np.zeros(0),
         "route_xy": np.stack([np.arange(-20.0, 380.0), np.zeros(400)], -1),
         "goal_pose": np.zeros(3), "goal_valid": np.bool_(False), **ring(pts)}
    return {k: v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v for k, v in z.items()}


def edges_of(z: dict, device) -> dict:
    e = {k.replace("da_", "edge_"): torch.as_tensor(v, dtype=torch.float32, device=device)[None] for k, v in z.items()
         if k.startswith("da_")}
    n = e["edge_xy"].shape[1]
    e["edge_same_prev"] = e["edge_valid"] = torch.ones(1, n, dtype=torch.bool, device=device)
    return e


def lead_boxes(n: int, device) -> dict:
    k = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    centre_x = LEAD_X + LEAD_SPEED * 0.1 * k
    box = torch.stack([centre_x, torch.zeros_like(k), torch.full_like(k, 4.8), torch.full_like(k, 2.0),
                       torch.zeros_like(k)], -1)[None, :, None]                               # [1, n, 1, 5]
    return {"box": box, "valid": torch.ones(1, n, 1, dtype=torch.bool, device=device),
            "weight": torch.ones(1, n, 1, device=device)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dp_dir", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--write_random_anchor_ckpt", default="")
    args = p.parse_args()
    if args.write_random_anchor_ckpt:
        acfg = AnchorWMConfig()
        torch.manual_seed(0)
        awm = AnchorWorldModel(acfg, torch.randn(3, acfg.num_anchors, acfg.future_steps, 2).cumsum(2))
        torch.save({"model": awm.state_dict(), "config": acfg.to_dict(), "args": {"random_init": True}, "step": 0},
                   args.write_random_anchor_ckpt)
        print(f"wrote a random-init AnchorWM checkpoint (plumbing only) to {args.write_random_anchor_ckpt}")
        return
    if not args.dp_dir:
        p.error("--dp_dir is required (the released Diffusion Planner: args.json, model.pth)")
    dev = torch.device(args.device)
    ok = True

    def gate(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    model, cfg = load_dp(args.dp_dir, dev)
    inputs = cfg.observation_normalizer(dp_inputs(cfg, dev))
    sampler = PopulationSampler(model)
    ctx = sampler.encode(inputs)

    # 1. parity with the decoder
    torch.manual_seed(7)
    with torch.no_grad():
        _, out = model(dict(inputs))
    torch.manual_seed(7)
    _, pred = sampler.sample(ctx, 1)
    d = float((out["prediction"] - pred).abs().max())
    gate("parity with Diffusion Planner's decoder", d == 0.0, f"max |diff| {d:.3g} over {tuple(pred.shape)}")

    # diagnostic: how much the released model's population spreads at all
    spread = {}
    for ns in (0.5, 1.0, 2.0):
        torch.manual_seed(4)
        _, sp = sampler.sample(ctx, 32, noise_scale=ns)
        spread[ns] = spread_m(plans_of(sp))
    print("[info] released model, population spread vs noise_scale (mean |plan - mean plan|, max endpoint "
          "deviation): " + "; ".join(f"{ns}: {a:.3f} m, {b:.3f} m" for ns, (a, b) in spread.items()))

    # 2-3 on the echo denoiser (x0 = x_t): its samples are its noise, so re-noising and the search are visible
    rmodel, _ = load_dp(args.dp_dir, dev)
    rmodel.decoder.decoder.dit = _EchoDiT()
    rsampler = PopulationSampler(rmodel)
    rctx = rsampler.encode(inputs)

    # 2. re-noising
    torch.manual_seed(0)
    raw, pred = rsampler.sample(rctx, 1)
    base = plans_of(pred)[0]
    dists = []
    for t in (0.02, 0.1, 0.3, 1.0):
        _, cp = rsampler.sample(rctx, 32, proposals=raw.expand(32, -1, -1), t_start=t, steps=3 if t < 1 else 10)
        dists.append(float((plans_of(cp)[:, 1:, :2] - base[1:, :2]).norm(dim=-1).mean()))
    gate("re-noising grows with resample_t", all(a < b for a, b in zip(dists, dists[1:])),
         "mean ego displacement from the proposal at t = 0.02 / 0.1 / 0.3 / 1.0: "
         + " / ".join(f"{v:.3f} m" for v in dists) + f"; fresh-population spread {spread_m(plans_of(cp))[0]:.2f} m")

    # 3. ES on a known objective: end 3 s in the left lane (y = 3.5)
    es = ESConfig(population=args.population, elites=8, iterations=4)
    no_repair = lambda plans, pred: (plans, {"moved": torch.zeros(len(plans), dtype=torch.bool)})  # noqa: E731
    left = lambda plans: -(plans[:, 30, 1] - LANES_Y[2]).abs()  # noqa: E731
    torch.manual_seed(1)
    best, info = diffusion_es(rsampler, rctx, es, no_repair, left)
    by = info["best_by_iter"]
    gate("ES climbs a known objective", all(a <= b for a, b in zip(by, by[1:])) and by[-1] > by[0],
         f"best per round {by} (y at 3 s {float(best[30, 1]):.2f} m); born in round {info['born']}")

    # 4. AnchorWM scorer invariances (random init)
    acfg = AnchorWMConfig()
    torch.manual_seed(0)
    awm = AnchorWorldModel(acfg, torch.randn(3, acfg.num_anchors, acfg.future_steps, 2).cumsum(2)).to(dev).eval()
    wm = AnchorWM(model=awm, device=dev).prepare(road_scene())
    fc = wm.forecast()
    torch.manual_seed(2)
    _, pred = sampler.sample(ctx, args.population)
    plans = plans_of(pred)
    s_all, sub = wm.score(plans, chunk=args.population)
    perm = torch.randperm(len(plans), device=dev)
    s_perm, _ = wm.score(plans[perm], chunk=7)
    d = float((s_perm - s_all[perm]).abs().max())
    gate("AnchorWM scorer is batch / chunk consistent", torch.isfinite(s_all).all().item() and d < 1e-4,
         f"scores {tuple(s_all.shape)} finite, sub {tuple(sub.shape)}, permuted + chunked max |diff| {d:.2g}; "
         f"forecast {fc['xy'].shape} for {len(fc['cur_xy'])} agent(s)")

    # 5. batched repair == per-plan repair
    repair = PosthocRepair()
    n = repair.cfg.window
    edges, agents = edges_of(road_scene(), dev), lead_boxes(n, dev)
    drift = plans.clone()
    # the first 8 plans drift left by up to 1..8 m over the repair's 2 s window (the road edge is 5.25 m out)
    ramp = (torch.arange(1, drift.shape[1], device=dev, dtype=drift.dtype) / n).clamp(max=1.0)
    drift[:8, 1:, 1] += torch.arange(1, 9, device=dev, dtype=drift.dtype)[:, None] * ramp

    def batched_vs_single(dtype):
        cast = lambda d: {key: v.to(dtype) if v.is_floating_point() else v for key, v in d.items()}  # noqa: E731
        p, e, a = drift.to(dtype), cast(edges), cast(agents)
        k = len(p)
        out_b, info_b = repair(p, *[{key: v.expand(k, *v.shape[1:]) for key, v in d.items()} for d in (e, a)])
        worst = max(float((repair(p[i:i + 1], e, a)[0][0] - out_b[i]).abs().max()) for i in range(k))
        return worst, info_b

    worst, info_b = batched_vs_single(torch.float64)
    worst32, _ = batched_vs_single(torch.float32)
    gate("batched repair == per-plan repair", worst < 1e-6 and bool(info_b["moved"].any()),
         f"float64 max |diff| {worst:.2g} m over {len(drift)} plans (float32 {worst32:.2g} m); "
         f"charged {int((info_b['loss0'] > 0).sum())}, moved {int(info_b['moved'].sum())}")

    # 6. one full replan
    def repair_fn(plans, pred):
        kk = len(plans)
        bb = lambda d: {key: v.expand(kk, *v.shape[1:]) for key, v in d.items()}  # noqa: E731
        return repair(plans, bb(edges), bb(agents))

    torch.manual_seed(3)
    t0 = time.perf_counter()
    best, info = diffusion_es(sampler, ctx, es, repair_fn, lambda p: wm.score(p)[0])
    sec = time.perf_counter() - t0
    gate("full replan (population -> repair -> AnchorWM score -> ES)", bool(torch.isfinite(best).all()),
         f"{1e3 * sec:.0f} ms on {dev}; " + ", ".join(f"{k}={v}" for k, v in info.items()))
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
