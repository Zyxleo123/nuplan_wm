"""The vendored FlowDrive inference against upstream's, on synthetic scenes (no nuPlan needed).

    python -m flow_drive.check --upstream DIR [--ckpt DIR/flow_drive/checkpoint/flow_drive_model.pth]

DIR is a flow_drive_planner checkout (06f941e). Needs torch, timm and diffusers (upstream's scheduler and EMA
loader); upstream's mlflow / mmengine / box imports are stubbed, nothing on the inference path uses them. Gates:

* the files kept verbatim are byte-identical to upstream's;
* ``euler_sigmas`` == ``FlowMatchEulerDiscreteScheduler(2000).set_timesteps(8).sigmas``;
* every parameter ``load_ema_`` loads == upstream ``load_checkpoint_directly`` (diffusers ``EMAModel``);
* ``FlowDrivePlanner(inputs)`` == upstream's ``FlowDrivePlanner.forward`` (``sample_action`` + the diffusers
  scheduler) under the same seed, bit for bit, on a batch of synthetic scenes and on each scene alone;
* the plan is finite and [B, 40, 3]; another seed gives another plan.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import filecmp
import importlib
import os
import sys
import types

import numpy as np
import torch

VERBATIM = ["model/module/dit.py", "model/module/mixer.py", "config/normalization.json"] + [
    f"data_process/{f}.py" for f in ("agent_process", "data_processor", "ego_process", "map_process",
                                      "roadblock_utils", "utils")]
_UPSTREAM: dict = {}
FAILED: list = []


def gate(ok: bool, what: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILED.append(what)


def _is_ours(name: str) -> bool:
    return name.split(".")[0] == "flow_drive"


@contextlib.contextmanager
def upstream(root: str):
    """Swap upstream's ``flow_drive`` (a namespace package under ``root``) in for ours, imports inside included."""
    ours = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_ours(k)}
    path = sys.path[:]
    sys.path[:] = [root] + [p for p in path if not os.path.isfile(os.path.join(p or ".", "flow_drive", "__init__.py"))]
    sys.modules.update(_UPSTREAM)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        _UPSTREAM.update({k: sys.modules.pop(k) for k in list(sys.modules) if _is_ours(k)})
        sys.path[:] = path
        sys.modules.update(ours)
        importlib.invalidate_caches()


def stub_training_imports() -> None:
    fileio = types.SimpleNamespace(get_text=lambda p: open(p).read(), get=lambda p: open(p, "rb").read())
    for name, attrs in {"mlflow": {}, "mmengine": {"fileio": fileio}, "box": {"ConfigBox": dict}}.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod


def synthetic_inputs(B: int, seed: int) -> dict:
    """Raw (unnormalised) model inputs shaped as ``DataProcessor.observation_adapter``'s, with empty slots."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    keep = lambda n, p: (torch.rand(B, n, generator=g) < p).float()
    nb = r(B, 32, 21, 11) * 5.0
    nb[..., 8:] = torch.nn.functional.one_hot(torch.randint(0, 3, (B, 32, 1), generator=g), 3).float()
    nb *= keep(32, 0.6)[..., None, None]
    nb[:, :, :5] *= (torch.rand(B, 32, 1, 1, generator=g) < 0.7).float()  # some agents appear mid-history
    st = r(B, 5, 10) * 5.0
    st *= keep(5, 0.5)[..., None]
    lanes = r(B, 70, 20, 12) * 10.0
    lanes[..., 8:] = torch.nn.functional.one_hot(torch.randint(0, 4, (B, 70, 1), generator=g), 4).float()
    lanes *= keep(70, 0.7)[..., None, None]
    lanes[:, :3] = r(3, 20, 12).abs() + 0.1  # every scene keeps a few lanes
    route = r(B, 25, 20, 12) * 10.0
    route *= keep(25, 0.5)[..., None, None]
    ego = torch.zeros(B, 10)
    ego[:, 2] = 1.0
    ego[:, 4:] = r(B, 6) * torch.tensor([6.0, 0.3, 1.0, 0.3, 0.1, 0.1])
    return {"neighbor_agents_past": nb, "static_objects": st, "lanes": lanes,
            "lanes_speed_limit": torch.rand(B, 70, 1, generator=g) * 20.0,
            "lanes_has_speed_limit": torch.rand(B, 70, 1, generator=g) < 0.8,
            "lanes_is_route": torch.randint(0, 2, (B, 70, 1), generator=g),
            "route_lanes": route, "route_lanes_speed_limit": torch.rand(B, 25, 1, generator=g) * 20.0,
            "route_lanes_has_speed_limit": torch.rand(B, 25, 1, generator=g) < 0.8,
            "ego_current_state": ego}


def upstream_forward(up: dict, params, encoder, decoder, inputs: dict) -> torch.Tensor:
    """Upstream ``FlowDrivePlanner.__init__`` (scheduler, normalisers) + ``forward``, line for line."""
    noise_scheduler = up["train_utils"].get_noise_scheduler(params)
    action_normalizer = up["normalizer"].StateNormalizer.from_json(params.data_processing)
    observation_normalizer = up["normalizer"].ObservationNormalizer.from_json(params.data_processing)
    with torch.inference_mode():
        inputs = observation_normalizer(inputs)
        encoder_outputs = encoder(inputs)
        action_norm, _ = up["infer_utils"].sample_action(
            params, decoder, encoder_outputs, noise_scheduler, inputs["ego_current_state"][:, :4])
        action = action_normalizer.inverse(action_norm)
        heading = torch.arctan2(action[:, :, 3], action[:, :, 2])
        action = torch.cat([action[:, :, :2], heading.unsqueeze(-1)], dim=-1)
    return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True, help="flow_drive_planner checkout")
    p.add_argument("--ckpt", default="", help="default: the checkout's flow_drive/checkpoint/flow_drive_model.pth")
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--seed", type=int, default=520)
    args = p.parse_args()
    root = os.path.abspath(args.upstream)
    ckpt = args.ckpt or os.path.join(root, "flow_drive", "checkpoint", "flow_drive_model.pth")
    print(f"torch {torch.__version__}, upstream {root}, ckpt {ckpt}")
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))

    from flow_drive.model.flow_drive_planner import FlowDrivePlanner, euler_sigmas, load_params
    here = os.path.dirname(os.path.abspath(__file__))
    same = [f for f in VERBATIM if filecmp.cmp(os.path.join(here, f), os.path.join(root, "flow_drive", f), shallow=False)]
    gate(len(same) == len(VERBATIM), f"verbatim files identical to upstream: {len(same)}/{len(VERBATIM)} "
         f"{sorted(set(VERBATIM) - set(same)) or ''}")

    params = load_params()
    ours = FlowDrivePlanner(params, ckpt, device="cpu")

    import diffusers
    import diffusers.training_utils  # before the stubs: diffusers probes which packages are installed
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    stub_training_imports()
    up_params = copy.deepcopy(params)
    up_params.train = types.SimpleNamespace(ema_power=0.99)
    up_params.data_processing.normalization_file_path = os.path.join(root, "flow_drive", "config", "normalization.json")
    with upstream(root):
        up = {m: importlib.import_module(f"flow_drive.utils.{m}") for m in ("train_utils", "infer_utils", "normalizer")}
        assert up["train_utils"].__file__.startswith(root), up["train_utils"].__file__
        sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=params.inference.flow_train_iter)
        sched.set_timesteps(params.inference.flow_inference_iter)
        up_enc, up_dec = up["train_utils"].load_checkpoint_directly(up_params, ckpt, device="cpu")
        up_enc, up_dec = up_enc.eval(), up_dec.eval()
    print(f"diffusers {diffusers.__version__}")

    mine = euler_sigmas(params.inference.flow_train_iter, params.inference.flow_inference_iter)
    gate(torch.equal(mine, sched.sigmas.cpu()), f"Euler sigmas == diffusers' ({mine.numpy().round(4).tolist()})")

    for name, a, b in (("encoder", ours.encoder, up_enc), ("decoder", ours.decoder, up_dec)):
        sa, sb = a.state_dict(), b.state_dict()
        bad = [k for k in sa if k not in sb or not torch.equal(sa[k], sb[k])]
        gate(sa.keys() == sb.keys() and not bad, f"{name}: {len(sa)} tensors == upstream's EMA load {bad[:3] or ''}")
    fresh = type(ours.decoder)(3, 4, 41, 192, 6, 0.1)
    n_init = sum(int(torch.equal(p, q)) for p, q in zip(ours.decoder.parameters(), fresh.parameters()))
    print(f"       decoder parameters equal to a fresh init: {n_init} of {len(list(fresh.parameters()))}")

    x = synthetic_inputs(args.batch, seed=0)
    torch.manual_seed(args.seed)
    plan = ours(x)
    torch.manual_seed(args.seed)
    with upstream(root):
        ref = upstream_forward(up, up_params, up_enc, up_dec, x)
    d = (plan - ref).abs().max().item()
    gate(torch.equal(plan, ref), f"batch of {args.batch}: plan == upstream's, same seed (max |diff| {d:.3g})")
    for b in range(args.batch):
        xb = {k: v[b:b + 1] for k, v in x.items()}
        torch.manual_seed(args.seed + b)
        pb = ours(xb)
        torch.manual_seed(args.seed + b)
        with upstream(root):
            rb = upstream_forward(up, up_params, up_enc, up_dec, xb)
        gate(torch.equal(pb, rb), f"scene {b} alone, seed {args.seed + b}: == upstream (max |diff| "
             f"{(pb - rb).abs().max().item():.3g}); end {pb[0, -1, :2].numpy().round(2).tolist()} m")
    gate(plan.shape == (args.batch, params.diffuser.pred_horizon, 3) and bool(torch.isfinite(plan).all()),
         f"plan finite, shape {tuple(plan.shape)}")
    torch.manual_seed(args.seed + 1000)
    other = ours(x)
    gate(not torch.equal(other, plan), f"another seed moves the plan (max {(other - plan).abs().max().item():.3g})")

    print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
