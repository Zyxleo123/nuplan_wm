"""Train AnchorWM (forecasting + ego-plan scoring) on the nuPlan AnchorWM cache.

  --joint_grad_scale 0     separate: the scorer trains on detached WM outputs
  --joint_grad_scale 1     joint: scorer gradients reach the decoder and encoder (ramped in after
                           --joint_warmup_steps over --joint_ramp_steps)
  --no_scorer              forecasting only
  --scorer_loss current | simple

  python -m anchor_wm.train --cache_dir .../anchor_wm_cache --anchors .../anchors_k64.npz \
      --save_dir .../anchor_wm/runs --run_name joint_current --joint_grad_scale 1 --scorer_loss current

`--profile_steps N` runs N steps, prints seconds/step (data wait vs compute) and peak GPU memory, and
exits without saving.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from anchor_wm.candidates import KIND_ANCHOR, KIND_PERTURB, build_candidates, label_candidates
from anchor_wm.config import AnchorWMConfig
from anchor_wm.data import make_loader, resolve_files, split_files
from anchor_wm.losses import score_loss, simple_score_loss, wm_loss
from anchor_wm.model import ENCODER_PREFIXES, AnchorWorldModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--file_list", default="", help="JSON list of cache files (default: every npz in cache_dir)")
    p.add_argument("--anchors", required=True, help="npz from anchor_wm.anchors")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--run_name", default="anchor_wm")
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_hours", type=float, default=0.0, help="stop (and save) after this wall time; 0 = off")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_fraction", type=float, default=0.05, help="share of logs held out")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    # scorer / joint training
    p.add_argument("--no_scorer", action="store_true")
    p.add_argument("--joint_grad_scale", type=float, default=1.0, help="0 = separate, > 0 = joint")
    p.add_argument("--joint_warmup_steps", type=int, default=10000)
    p.add_argument("--joint_ramp_steps", type=int, default=10000)
    p.add_argument("--score_weight", type=float, default=1.0)
    p.add_argument("--train_cands", type=int, default=48, help="candidates scored per scene (vocabulary dropout)")
    p.add_argument("--score_chunk", type=int, default=0)
    p.add_argument("--scorer_loss", default="current", choices=("current", "simple"),
                   help="current: sub-score BCE + pairwise + listwise; simple: one utility, listwise ranking + "
                        "time-weighted overlap / offroad BCE (losses.simple_score_loss)")
    p.add_argument("--w_sub", type=float, default=1.0)
    p.add_argument("--w_pair", type=float, default=1.0)
    p.add_argument("--w_list", type=float, default=0.5)
    p.add_argument("--progress_margin", type=float, default=2.0)
    p.add_argument("--simple_w_safe", type=float, default=1.0)
    p.add_argument("--simple_temp", type=float, default=1.0)
    p.add_argument("--w_aux", type=float, default=0.5, help="aux heads (time to violation, route progress)")
    # model
    p.add_argument("--cls_soft_temp_m", type=float, default=1.0)
    p.add_argument("--scorer_hidden", type=int, default=128)
    p.add_argument("--sigma_eps", type=float, default=0.002, help="sigma floor, normalised by max_range (50 m)")
    p.add_argument("--scorer_agents", type=int, default=32)
    p.add_argument("--scorer_modes", type=int, default=6)
    p.add_argument("--scorer_near_agents", type=int, default=8)
    p.add_argument("--scorer_near_modes", type=int, default=12)
    p.add_argument("--use_goal", action="store_true", help="also feed the mission goal (ego query and scorer)")
    # bookkeeping
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=2500)
    p.add_argument("--val_batches", type=int, default=150)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--keep_every", type=int, default=0, help="also keep step_<N>.pt every this many steps; 0 = off")
    p.add_argument("--nan_max_consecutive", type=int, default=50,
                   help="a step with a non-finite loss or gradient is skipped; stop after this many in a row")
    p.add_argument("--profile_steps", type=int, default=0)
    p.add_argument("--wandb_project", default="anchor_wm_nuplan")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_mode", default="online")
    return p.parse_args()


def joint_scale(args, step: int) -> float:
    if args.joint_grad_scale <= 0:
        return 0.0
    ramp = (step - args.joint_warmup_steps) / max(1, args.joint_ramp_steps)
    return float(args.joint_grad_scale) * float(min(1.0, max(0.0, ramp)))


def subsample_candidates(cands: dict, n: int, generator: torch.Generator) -> torch.Tensor:
    """Indices [B,n]: the logged future first, then a random subset of the remaining valid candidates."""
    b, c = cands["valid"].shape
    gt_col = int((cands["kinds"] == 1).nonzero()[0])
    noise = torch.rand((b, c), generator=generator).to(cands["valid"].device)
    noise = noise.masked_fill(~cands["valid"], 2.0)
    noise[:, gt_col] = -1.0
    return noise.argsort(1)[:, :min(n, c)]


def take(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.gather(1, idx.view(*idx.shape, *([1] * (x.dim() - 2))).expand(*idx.shape, *x.shape[2:]))


def scorer_step(model: AnchorWorldModel, enc: dict, dec: dict, f: dict, args, cfg: AnchorWMConfig,
                scale: float, generator: torch.Generator) -> tuple[torch.Tensor, dict]:
    cands = build_candidates(f, dec, cfg.max_range, cfg.future_dt, generator)
    idx = subsample_candidates(cands, args.train_cands, generator)
    xy, yaw, valid = take(cands["xy"], idx), take(cands["yaw"], idx), take(cands["valid"], idx)
    labels = label_candidates(f, xy, yaw, cfg.future_dt)
    # Drop scenes whose logged future is incomplete or itself labelled unsafe (label noise).
    # Column 0 is the logged future (subsample_candidates).
    gt_ok = valid[:, 0] & ~(labels["overlap"][:, 0] | labels["offroad"][:, 0])
    valid = valid & gt_ok[:, None]
    ctx = model.scorer.context(enc, dec, f, scale)
    total, sub, aux = model.scorer(ctx, xy, yaw, chunk=args.score_chunk)
    if args.scorer_loss == "simple":
        loss, metrics = simple_score_loss(total, sub, labels, valid, w_safe=args.simple_w_safe, temp=args.simple_temp,
                                          progress_margin=args.progress_margin, aux=aux, w_aux=args.w_aux)
    else:
        loss, metrics = score_loss(total, sub, labels, valid, progress_margin=args.progress_margin,
                                   w_sub=args.w_sub, w_pair=args.w_pair, w_list=args.w_list, aux=aux, w_aux=args.w_aux)
    with torch.no_grad():
        metrics["scene_kept"] = float(gt_ok.float().mean())
        gt_valid = take(cands["valid"], idx)[:, 0]
        metrics["gt_overlap_rate"] = float((gt_valid & labels["overlap"][:, 0]).float().mean())
        metrics["gt_any_overlap_rate"] = float((gt_valid & labels["any_overlap"][:, 0]).float().mean())
        metrics["cand_any_overlap_rate"] = float((labels["any_overlap"].float() * valid).sum() / valid.sum().clamp_min(1))
        metrics["gt_offroad_rate"] = float((gt_valid & labels["offroad"][:, 0]).float().mean())
        kinds = take(cands["kinds"][None].expand(idx.shape[0], -1), idx)
        unsafe = (labels["overlap"] | labels["offroad"]).float()
        for name, kd in (("anchor", KIND_ANCHOR), ("perturb", KIND_PERTURB)):
            m = (kinds == kd) & valid
            metrics[f"unsafe_rate_{name}"] = float((unsafe * m).sum() / m.sum().clamp_min(1))
    return loss, metrics


@torch.no_grad()
def module_grad_norms(model: AnchorWorldModel) -> dict:
    """Gradient norms (before clipping) of the encoder, decoder and scorer. All three share one global clip."""
    parts = ("encoder", "decoder", "scorer")
    sq = {k: [] for k in parts}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        part = "encoder" if name.startswith(ENCODER_PREFIXES) else "scorer" if name.startswith("scorer.") else "decoder"
        sq[part].append(p.grad.float().square().sum())
    zero = next(model.parameters()).new_zeros((), dtype=torch.float32)
    norms = torch.stack([torch.stack(sq[k]).sum() if sq[k] else zero for k in parts]).sqrt().tolist()
    return {f"grad_norm_{k}": v for k, v in zip(parts, norms)}


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def validate(model, val_batches, args, cfg, device, amp, scale) -> dict:
    model.eval()
    sums, count = {}, 0
    g = torch.Generator().manual_seed(1234)
    for batch in val_batches:
        f = move(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            enc, dec = model(f)
            dec = {k: v.float() for k, v in dec.items()}
            _, m = wm_loss(dec, f, cfg)
            if not args.no_scorer:
                _, ms = scorer_step(model, enc, dec, f, args, cfg, scale, g)
                m.update(ms)
        for k, v in m.items():
            if math.isfinite(v):
                sums[k] = sums.get(k, 0.0) + v
        count += 1
    model.train()
    return {k: v / max(count, 1) for k, v in sums.items()}


def build_config(args, num_anchors: int) -> AnchorWMConfig:
    return AnchorWMConfig(num_anchors=num_anchors, cls_soft_temp_m=args.cls_soft_temp_m,
                          scorer_hidden=args.scorer_hidden, sigma_eps=args.sigma_eps,
                          scorer_agents=args.scorer_agents, scorer_modes=args.scorer_modes,
                          scorer_near_agents=args.scorer_near_agents, scorer_near_modes=args.scorer_near_modes,
                          use_goal=args.use_goal)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp = device.type == "cuda" and not args.no_amp and torch.cuda.is_bf16_supported()

    anchors = torch.from_numpy(np.load(args.anchors)["anchors"])
    cfg = build_config(args, int(anchors.shape[1]))
    model = AnchorWorldModel(cfg, anchors).to(device)
    # AdamW: decay weight matrices only, not biases / norm gains / type tokens.
    groups, base_lrs = [], []
    for enc_part, lr in ((True, args.encoder_lr), (False, args.lr)):
        params = [(n, p) for n, p in model.named_parameters() if n.startswith(ENCODER_PREFIXES) == enc_part]
        skip = lambda n, p: p.dim() < 2 or n.endswith("_token")  # noqa: E731
        groups += [{"params": [p for n, p in params if not skip(n, p)], "lr": lr},
                   {"params": [p for n, p in params if skip(n, p)], "lr": lr, "weight_decay": 0.0}]
        base_lrs += [lr, lr]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"])

    train_files, val_files = split_files(resolve_files(args.cache_dir, args.file_list), args.val_fraction)
    loader = make_loader(train_files, cfg, batch_size=args.batch_size, shuffle=True, seed=args.seed + step,
                         num_workers=args.num_workers)
    batches = infinite(loader)

    run_dir = Path(args.save_dir) / f"{args.run_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wb = None
    if not args.profile_steps:
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {"config": cfg.to_dict(), "args": vars(args),
                "params": int(sum(p.numel() for p in model.parameters())),
                "params_scorer": int(sum(p.numel() for p in model.scorer.parameters())),
                "train_files": len(train_files), "val_files": len(val_files)}
        (run_dir / "config.json").write_text(json.dumps(meta, indent=1, default=str))
        print(json.dumps(meta, default=str), flush=True)
        val_batches = []
        for i, b in enumerate(make_loader(val_files, cfg, batch_size=args.batch_size, shuffle=False, seed=0,
                                          num_workers=min(args.num_workers, 2))):
            if i >= args.val_batches:
                break
            val_batches.append(b)
        if args.wandb_mode != "disabled":
            import wandb
            wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=run_dir.name,
                            mode=args.wandb_mode, dir=str(run_dir), config=meta)

    def save(tag: str) -> None:
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": cfg.to_dict(),
                    "anchors": args.anchors, "args": vars(args), "step": step}, run_dir / f"{tag}.pt")

    gen = torch.Generator().manual_seed(args.seed)
    n_nonfinite = n_consecutive = 0
    t_start = time.time()
    running, n_run = {}, 0
    t_data = t_comp = 0.0
    prof = []
    model.train()
    while step < args.max_steps:
        t0 = time.time()
        f = move(next(batches), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        lr_mult = min(1.0, (step + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.max_steps)))
        for grp, base in zip(opt.param_groups, base_lrs):
            grp["lr"] = base * lr_mult
        scale = joint_scale(args, step)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            enc, dec = model(f)
            dec = {k: v.float() for k, v in dec.items()}
            loss, metrics = wm_loss(dec, f, cfg)
            if not args.no_scorer:
                l_s, m_s = scorer_step(model, enc, dec, f, args, cfg, scale, gen)
                loss = loss + args.score_weight * l_s
                metrics.update(m_s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        metrics.update(module_grad_norms(model))
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if bool(torch.isfinite(loss)) and bool(torch.isfinite(gnorm)):
            opt.step()
            n_consecutive = 0
        else:
            # bf16 autocast has no GradScaler to skip a non-finite step, which would write NaN into every weight.
            n_nonfinite += 1
            n_consecutive += 1
            bad = sorted(k for k, v in metrics.items() if not math.isfinite(v))
            print("NONFINITE " + json.dumps({"step": step, "nonfinite": n_nonfinite, "loss": float(loss),
                                             "grad_norm": float(gnorm), "nonfinite_metrics": bad}), flush=True)
            opt.zero_grad(set_to_none=True)
            if n_consecutive >= args.nan_max_consecutive:
                print(f"stopping: {n_consecutive} non-finite steps in a row at step {step}", flush=True)
                break
        metrics["nonfinite_steps"] = float(n_nonfinite)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        step += 1
        t_data += t1 - t0
        t_comp += t2 - t1
        metrics.update(grad_norm=float(gnorm), joint_scale=scale, lr=opt.param_groups[-1]["lr"])
        for k, v in metrics.items():
            if math.isfinite(v):
                running[k] = running.get(k, 0.0) + v
        n_run += 1

        if args.profile_steps:
            if step > 5:
                prof.append((t1 - t0, t2 - t1))
            if step >= args.profile_steps + 5:
                d, c = np.mean([p[0] for p in prof]), np.mean([p[1] for p in prof])
                out = {"scorer": not args.no_scorer, "joint_grad_scale": args.joint_grad_scale,
                       "batch_size": args.batch_size, "train_cands": args.train_cands,
                       "data_s_per_step": round(float(d), 4), "compute_s_per_step": round(float(c), 4),
                       "total_s_per_step": round(float(d + c), 4),
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device.type == "cuda" else None,
                       "last_metrics": {k: round(v, 4) for k, v in metrics.items()}}
                print("PROFILE " + json.dumps(out), flush=True)
                return
            continue

        if step % args.log_every == 0:
            avg = {k: v / n_run for k, v in running.items()}
            avg.update(s_data=t_data / n_run, s_compute=t_comp / n_run, hours=(time.time() - t_start) / 3600)
            print(json.dumps({"step": step, **{k: round(v, 4) for k, v in avg.items()}}), flush=True)
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in avg.items()}, step=step)
            running, n_run, t_data, t_comp = {}, 0, 0.0, 0.0
        stop = bool(args.max_hours) and (time.time() - t_start) / 3600 >= args.max_hours
        if step % args.val_every == 0 or step == args.max_steps or stop:
            vm = validate(model, val_batches, args, cfg, device, amp, scale)
            print(json.dumps({"step": step, "val": {k: round(v, 4) for k, v in vm.items()}}), flush=True)
            if wb is not None:
                wb.log({f"val/{k}": v for k, v in vm.items()}, step=step)
        if step % args.save_every == 0:
            save("latest")
        if args.keep_every and step % args.keep_every == 0:
            save(f"step_{step:06d}")
        if stop:
            print(f"max_hours reached at step {step}", flush=True)
            break
    save("latest")
    save("final")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
