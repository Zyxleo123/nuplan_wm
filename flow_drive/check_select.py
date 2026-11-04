"""Gates for FlowDrive best-of-K (best_of_k.py) on check_repair's synthetic road (no nuPlan data or devkit; CPU).

1. ``sample_population`` at K = 1 is ``FlowDrivePlanner.forward`` bit for bit (same seed), and sample 0 of K = 32 is
   it up to batched-kernel rounding; [info] the population's spread;
2. ``extend_plans`` keeps the 4 s plan and continues it along its last heading to the scorer's 8 s: ``speed`` at its
   last step's length, ``decel`` with that length decaying to 0;
3. under either extension the scorer ranks sample 0 at twice its speed (into the lead car) below the population's
   median; [info] where sample 0 pushed 6 m left through the road edge ranks, the same for ``--ref_scorer``, and how
   much the two scorers agree. The off-road candidate is not gated: this road is out of distribution for the scorer
   (in-lane FlowDrive samples get ``offroad`` sub-logits up to +11, job 93603; the off-road one +5), as ``dp_inputs``'s
   lanes are for FlowDrive. How it ranks real candidates shows in closed loop;
4. one best-of-K step with the repair (batched over the population) runs finite, with its stats.

    PYTHONPATH=. python -m flow_drive.check_select --ckpt flow_drive/checkpoint/flow_drive_model.pth \
        --scorer <anchor_wm checkpoint> [--ref_scorer <another>]
"""

from __future__ import annotations

import argparse

import torch

from diffusion_es.check import edges_of, lead_boxes, road_scene
from diffusion_es.es import spread_m
from flow_drive.best_of_k import best_of_k, extend_plans, sample_population
from flow_drive.check_repair import fd_inputs
from flow_drive.model.flow_drive_planner import FlowDrivePlanner, load_params
from posthoc_repair.anchor_model import AnchorWM
from posthoc_repair.optimizer import PosthocRepair


def ranks(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(x)).float()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scorer", required=True)
    p.add_argument("--ref_scorer", default="")
    p.add_argument("--population", type=int, default=32)
    args = p.parse_args()
    dev = torch.device("cpu")
    ok = True

    def gate(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    params = load_params()
    model = FlowDrivePlanner(params, args.ckpt, device="cpu")
    x = fd_inputs(params, dev)
    k = args.population

    # 1. population sampling vs FlowDrive's own sample
    torch.manual_seed(520)
    ref = model(x)
    torch.manual_seed(520)
    one = sample_population(model, x, 1)
    torch.manual_seed(520)
    pop = sample_population(model, x, k)
    d1, dk = float((one - ref).abs().max()), float((pop[:1] - ref).abs().max())
    gate("population sample 0 is FlowDrive's own sample", d1 == 0.0 and dk < 1e-3,
         f"K = 1 max |diff| {d1:.3g}; K = {k} sample 0 max |diff| {dk:.3g} (batched kernels)")
    plans = torch.cat([torch.zeros_like(pop[:, :1]), pop], 1)                         # [K, 41, 3]
    s_mean, s_end = spread_m(plans)
    print(f"[info] population of {k}: mean |plan - mean plan| {s_mean:.2f} m, max endpoint deviation {s_end:.2f} m")

    # 2. the 8 s extension
    last = (plans[:, -1, :2] - plans[:, -2, :2]).norm(dim=-1)
    want = {"speed": last[:, None].expand(-1, 40), "decel": last[:, None] * (1 - torch.arange(1, 41) / 40)[None]}
    for mode, target in want.items():
        ext = extend_plans(plans, 80, mode)
        step = (ext[:, 41:, :2] - ext[:, 40:-1, :2]).norm(dim=-1)
        err = float((step - target).abs().max())
        gate(f"{mode} extension keeps the plan, its heading and the intended step lengths",
             ext.shape == (k, 81, 3) and torch.equal(ext[:, :41], plans) and err < 1e-4
             and torch.equal(ext[:, 41:, 2], plans[:, -1:, 2].expand(-1, 40)),
             f"{tuple(plans.shape)} -> {tuple(ext.shape)}; step-length error {err:.2g} m; travelled after 4 s "
             f"{float((ext[:, -1, :2] - plans[:, -1, :2]).norm(dim=-1).mean()):.1f} m (mean)")

    # 3. the scorer on FlowDrive's candidates + two broken ones
    n = PosthocRepair().cfg.window
    off_road = plans[:1].clone()
    off_road[:, 1:, 1] += 6.0 * (torch.arange(1, 41, dtype=plans.dtype) / n).clamp(max=1.0)
    rush = plans[:1].clone()
    rush[:, :, :2] *= 2.0
    cands = torch.cat([plans, off_road, rush])
    scores = {}
    for tag, path in (("scorer", args.scorer), ("ref_scorer", args.ref_scorer)):
        if not path:
            continue
        wm = AnchorWM(path, device=dev, need_scorer=True).prepare(road_scene())
        for mode in ("speed", "decel"):
            s = wm.score(extend_plans(cands, int(wm.cfg.future_steps), mode))[0]
            if mode == "speed":
                scores[tag] = s
            med = float(s[:k].median())
            beaten = lambda i: int((s[:k] > s[i]).sum())  # noqa: E731
            line = (f"{mode}: population median {med:.2f}, best s{int(torch.argmax(s[:k]))} {float(s[:k].max()):.2f}, "
                    f"sample 0 {float(s[0]):.2f}; into the lead car {float(s[k + 1]):.2f} (beaten by {beaten(k + 1)}/{k}); "
                    f"[info] off-road {float(s[k]):.2f} (beaten by {beaten(k)}/{k})")
            if tag == "scorer":
                gate("the scorer ranks the candidate driving into the lead car below the median",
                     bool(torch.isfinite(s).all()) and float(s[k + 1]) < med, line)
            else:
                print(f"[info] ref_scorer {line}")
    if len(scores) == 2:
        a, b = ranks(scores["scorer"][:k]), ranks(scores["ref_scorer"][:k])
        rho = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        print(f"[info] rank correlation of the two scorers over the population: {rho:.2f}; their picks: "
              f"{int(torch.argmax(scores['scorer'][:k]))} / {int(torch.argmax(scores['ref_scorer'][:k]))}")

    # 4. one best-of-K step with the repair
    wm = AnchorWM(args.scorer, device=dev, need_scorer=True).prepare(road_scene())
    repair = PosthocRepair()
    edges, agents = edges_of(road_scene(), dev), lead_boxes(n, dev)
    batch = lambda d: {key: v.expand(len(cands), *v.shape[1:]) for key, v in d.items()}  # noqa: E731
    b, scored, s, info = best_of_k(cands, lambda q: wm.score(q)[0], int(wm.cfg.future_steps),
                                   lambda q: repair(q, batch(edges), batch(agents)))
    gate("one best-of-K step with the repair", bool(torch.isfinite(scored).all() and torch.isfinite(s).all())
         and b < k + 2, f"chose {b} of {len(cands)} (score {info['best']:.3f}, sample 0 {info['score0']:.3f}); "
         f"repair charged {info['charged']}, moved {info['repaired']} (off-road moved: "
         f"{not torch.equal(scored[k], cands[k])}); {info['ms_repair']:.0f} ms repair, {info['ms_score']:.0f} ms score")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
