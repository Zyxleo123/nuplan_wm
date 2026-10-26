"""Fit Cartesian trajectory anchors (k-means) per agent group from the AnchorWM cache.

Futures are taken in each agent's frame (position and heading at t = 0), fully valid over the
horizon; the ego (rear axle) joins the vehicle group. Clustering uses 5 evenly spaced waypoints
(MultiPath style); each anchor is the mean full trajectory of its cluster. Near-stationary futures
(< 0.5 m travelled) are kept at 10% so k-means spends its clusters on motion. Writes

  <out>.npz   anchors [3,K,T,2] (m, agent frame; vehicle / pedestrian / bicycle), counts [3,K]
  <out>.json  oracle best-anchor ADE / FDE / miss rate (FDE > 2 m) on validation files

  python -m anchor_wm.anchors --cache_dir .../anchor_wm_cache --out .../anchors_k64 --num_anchors 64
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from anchor_wm.config import ANCHOR_GROUPS, AnchorWMConfig
from anchor_wm.data import NOW, agent_arrays, resolve_files, split_files

STATIONARY_M = 0.5
STATIONARY_KEEP = 0.1


def agent_frame_futures(z, cfg: AnchorWMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Fully valid futures [M,T,2] in each agent's own frame (ego + every dynamic agent valid at t = 0),
    and their groups [M]."""
    a = agent_arrays(z, 1 + len(z["obj_type"]))
    fut_idx = NOW + np.arange(1, cfg.future_steps + 1) * cfg.future_stride
    ok = a["valid"][:, NOW] & a["valid"][:, fut_idx].all(-1)
    cur, fut = a["pose"][ok, NOW], a["pose"][ok][:, fut_idx, :2]
    c, s = np.cos(cur[:, 2])[:, None], np.sin(cur[:, 2])[:, None]
    d = fut - cur[:, None, :2]
    local = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)
    return local.astype(np.float32), a["type"][ok].astype(np.int64)


def _file_futures(args) -> tuple[np.ndarray, np.ndarray]:
    path, cfg = args
    with np.load(path) as z:
        return agent_frame_futures(z, cfg)


def collect(files, cfg, max_per_group: int, seed: int, num_workers: int) -> dict:
    """Reads in parallel: a cold cache file takes ~0.3 s on NFS, almost all of it latency."""
    rng = np.random.default_rng(seed)
    bank = {g: [] for g in range(len(ANCHOR_GROUPS))}
    with Pool(num_workers) as pool:
        for i, (fut, grp) in enumerate(pool.imap(_file_futures, [(p, cfg) for p in files], chunksize=16)):
            for g in bank:
                bank[g].append(fut[grp == g])
            if (i + 1) % 10000 == 0:
                print(f"read {i + 1}/{len(files)} files", flush=True)
    out = {}
    for g in bank:
        arr = np.concatenate(bank[g]) if bank[g] else np.zeros((0, cfg.future_steps, 2), np.float32)
        moving = np.linalg.norm(arr[:, -1], axis=-1) >= STATIONARY_M
        arr = arr[moving | (rng.random(len(arr)) < STATIONARY_KEEP)]
        if len(arr) > max_per_group:
            arr = arr[rng.choice(len(arr), max_per_group, replace=False)]
        out[g] = arr
    return out


def kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    """k-means++ init + Lloyd. x [M,D] -> assignment [M]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = x.shape[0]
    centers = [x[torch.randint(m, (1,), generator=g).item()]]
    d2 = (x - centers[0]).square().sum(-1)
    for _ in range(1, k):
        probs = (d2 / d2.sum().clamp_min(1e-12)).cpu()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers.append(x[idx])
        d2 = torch.minimum(d2, (x - x[idx]).square().sum(-1))
    c = torch.stack(centers)
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(-1)
        new = torch.zeros_like(c).index_add_(0, assign, x)
        cnt = torch.bincount(assign, minlength=k).float()
        empty = cnt == 0
        c = torch.where(empty[:, None], c, new / cnt.clamp_min(1)[:, None])
        if empty.any():   # re-seed empty clusters at the worst-fit points
            far = torch.cdist(x, c).amin(-1).topk(int(empty.sum())).indices
            c[empty] = x[far]
    return torch.cdist(x, c).argmin(-1)


def fit(futures: np.ndarray, k: int, iters: int, seed: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    t = futures.shape[1]
    if len(futures) < k:
        raise ValueError(f"only {len(futures)} samples for {k} anchors")
    waypoints = [(i + 1) * t // 5 - 1 for i in range(5)]
    x = torch.from_numpy(futures[:, waypoints].reshape(len(futures), -1)).to(device)
    assign = kmeans(x, k, iters, seed)
    full = torch.from_numpy(futures.reshape(len(futures), -1)).to(device)
    sums = torch.zeros((k, full.shape[1]), device=device).index_add_(0, assign, full)
    cnt = torch.bincount(assign, minlength=k).float()
    anchors = (sums / cnt.clamp_min(1)[:, None]).reshape(k, t, 2)
    order = torch.argsort(anchors[:, -1, 0])          # sort by forward endpoint for readability
    return anchors[order].cpu().numpy(), cnt[order].cpu().numpy().astype(np.int64)


def oracle(anchors: np.ndarray, futures: np.ndarray, device: str) -> dict:
    if len(futures) == 0:
        return {"samples": 0}
    a = torch.from_numpy(anchors).to(device)
    f = torch.from_numpy(futures).to(device)
    ade, fde = [], []
    for s in range(0, len(f), 20000):
        err = (f[s:s + 20000, None] - a[None]).norm(dim=-1)                                  # [m,K,T]
        e_ade = err.mean(-1)
        ade.append(e_ade.amin(-1))
        fde.append(err[:, :, -1].amin(-1))
    ade, fde = torch.cat(ade), torch.cat(fde)
    return {"samples": int(len(f)), "oracle_ade_m": float(ade.mean()), "oracle_fde_m": float(fde.mean()),
            "oracle_miss_rate_2m": float((fde > 2.0).float().mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--file_list", default="", help="JSON list of cache files (default: every npz in cache_dir)")
    p.add_argument("--out", required=True, help="output path without extension")
    p.add_argument("--num_anchors", type=int, default=64)
    p.add_argument("--max_train_files", type=int, default=50000)
    p.add_argument("--max_val_files", type=int, default=5000)
    p.add_argument("--val_cache_dir", default="", help="oracle report on this cache (Val14); fit on all of --cache_dir")
    p.add_argument("--val_fraction", type=float, default=0.0,
                   help="instead of --val_cache_dir: hold out this share of the training logs for the report")
    p.add_argument("--max_per_group", type=int, default=300000)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=16, help="file-reading processes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = AnchorWMConfig(num_anchors=args.num_anchors)
    files = resolve_files(args.cache_dir, args.file_list)
    if args.val_cache_dir:
        train_files, val_files = files, resolve_files(args.val_cache_dir)
    else:
        train_files, val_files = split_files(files, args.val_fraction)
    rng = np.random.default_rng(args.seed)
    train_files = [train_files[i] for i in sorted(rng.choice(len(train_files), min(args.max_train_files, len(train_files)), replace=False))]
    val_files = val_files[:args.max_val_files]
    t0 = time.time()
    train = collect(train_files, cfg, args.max_per_group, args.seed, args.num_workers)
    val = collect(val_files, cfg, args.max_per_group, args.seed + 1, args.num_workers)
    g_count = len(ANCHOR_GROUPS)
    anchors = np.zeros((g_count, args.num_anchors, cfg.future_steps, 2), np.float32)
    counts = np.zeros((g_count, args.num_anchors), np.int64)
    report = {"train_files": len(train_files), "val_files": len(val_files), "num_anchors": args.num_anchors,
              "future_steps": cfg.future_steps, "future_dt": cfg.future_dt, "groups": {}}
    for g, name in enumerate(ANCHOR_GROUPS):
        fallback = len(train[g]) < 20 * args.num_anchors and g > 0
        if fallback:   # too few samples: reuse the vehicle anchors for this group
            anchors[g], counts[g] = anchors[0], 0
        else:
            anchors[g], counts[g] = fit(train[g], args.num_anchors, args.iters, args.seed + g, args.device)
        report["groups"][name] = {"fitted_samples": int(len(train[g])), "vehicle_fallback": bool(fallback),
                                  "val": oracle(anchors[g], val[g], args.device),
                                  "empty_clusters": int((counts[g] == 0).sum())}
        print(name, json.dumps(report["groups"][name]), flush=True)
    report["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out) + ".npz", anchors=anchors, counts=counts, fitted_samples=np.array([len(train[g]) for g in range(g_count)]))
    (Path(str(out) + ".json")).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
