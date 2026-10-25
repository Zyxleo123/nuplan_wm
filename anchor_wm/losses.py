"""Forecasting loss (nearest-anchor assignment) and the two scorer losses on self-generated candidates:
`score_loss` (current: sub-score BCE + pairwise + listwise) and `simple_score_loss` (one utility,
listwise ranking + time-weighted safety BCE)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from anchor_wm.config import AnchorWMConfig
from anchor_wm.geometry import rotate_xy
from anchor_wm.model import gather_k, select_modes


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def gaussian_nll_xy(pred_xy: torch.Tensor, target_xy: torch.Tensor, sigma_xy: torch.Tensor) -> torch.Tensor:
    sigma = sigma_xy.clamp_min(1e-6)
    z = (pred_xy - target_xy) / sigma
    return 0.5 * z.square().sum(-1) + sigma.log().sum(-1)


def ade_fde(pred_xy_k: torch.Tensor, target_xy: torch.Tensor, valid: torch.Tensor, scale: float):
    """Per-mode ADE / FDE (m): pred [B,N,K,T,2], target [B,N,T,2], valid [B,N,T]."""
    err = (pred_xy_k - target_xy[:, :, None]).norm(dim=-1) * scale                          # [B,N,K,T]
    vf = valid.float()[:, :, None]
    ade = (err * vf).sum(-1) / vf.sum(-1).clamp_min(1.0)
    t = err.shape[-1]
    idx = torch.arange(t, device=err.device).view(1, 1, t)
    last = torch.where(valid, idx, torch.full_like(idx, -1)).amax(-1)                       # [B,N]
    fde = err.gather(-1, last.clamp_min(0)[:, :, None, None].expand(-1, -1, err.shape[2], 1)).squeeze(-1)
    return ade, fde, last >= 0


@torch.no_grad()
def interaction_distance(f: dict, cfg: AnchorWMConfig, speed_ratios: tuple = (0.0, 0.5, 1.0, 1.5)) -> torch.Tensor:
    """[B,N] closest time-aligned approach (m) of each agent (now and its valid logged future) to the
    ego: the ego's logged future or a constant-velocity rollout along the ego heading at `speed_ratios`
    x the ego speed. inf for agents never valid."""
    rng = float(cfg.max_range)
    cur = f["current_local_xy"].float()                                                     # [B,N,2]
    pos = torch.cat([cur[:, :, None], cur[:, :, None] + f["target_delta_xy"].float() * rng], 2)   # [B,N,T+1,2]
    ok = torch.cat([f["agent_valid"].bool()[:, :, None], f["target_valid_all"].bool()], 2)
    t = pos.shape[2] - 1
    time = torch.arange(t + 1, device=cur.device, dtype=cur.dtype) * cfg.future_dt
    speed = f["ego_vel"].float().norm(dim=-1)
    ratios = torch.tensor(speed_ratios, device=cur.device, dtype=cur.dtype)
    cv_x = speed[:, None, None] * ratios[None, :, None] * time[None, None, :]                # [B,R,T+1]
    refs = torch.stack([cv_x, torch.zeros_like(cv_x)], -1)                                  # [B,R,T+1,2]
    ego = torch.cat([torch.zeros_like(cur[:, 0, None]), f["ego_future_xy"].float()], 1)       # [B,T+1,2]
    ego_ok = torch.cat([torch.ones_like(f["ego_future_valid"][:, :1]), f["ego_future_valid"]], 1).bool()
    d_cv = (pos[:, :, None] - refs[:, None]).norm(dim=-1).amin(2)                            # [B,N,T+1]
    d_log = (pos - ego[:, None]).norm(dim=-1).masked_fill(~ego_ok[:, None], float("inf"))
    d = torch.minimum(d_cv, d_log).masked_fill(~ok, float("inf"))
    return d.amin(-1)


def wm_loss(dec: dict, f: dict, cfg: AnchorWMConfig) -> tuple[torch.Tensor, dict]:
    rng = float(cfg.max_range)
    valid = f["target_valid_all"].bool()                                                    # [B,N,T]
    sin, cos = dec["heading_sc"][..., 0, None], dec["heading_sc"][..., 1, None]
    target_agent = rotate_xy(f["target_delta_xy"].float(), -sin, cos)                       # normalized
    yaw_t = F.normalize(f["target_yaw_vec"].float(), dim=-1, eps=1e-6)

    # Nearest *anchor* (not nearest prediction) owns the target.
    anchor_ade, _, has = ade_fde(dec["anchors_agent"], target_agent, valid, rng)            # [B,N,K]
    k_star = anchor_ade.argmin(-1)
    pick = lambda x: gather_k(x, k_star[..., None]).squeeze(2)  # noqa: E731
    xy = pick(dec["pred_xy_agent_k"])
    sigma = pick(dec["sigma_k"])
    yaw = pick(dec["yaw_vec_k"])

    # Agents likely to interact with the ego count more.
    agent_w = 1.0 + cfg.collision_focus_weight * torch.exp(-interaction_distance(f, cfg) / cfg.interaction_tau_m)
    agent_w = torch.where(f["is_sdc"].bool(), torch.full_like(agent_w, cfg.ego_loss_weight), agent_w)
    time_w = valid.float() * agent_w[..., None]
    denom = time_w.sum().clamp_min(1.0)
    nll = (gaussian_nll_xy(xy, target_agent, sigma) * time_w).sum() / denom
    yaw_loss = ((1.0 - (yaw * yaw_t).sum(-1).clamp(-1.0, 1.0)) * time_w).sum() / denom

    logp = F.log_softmax(dec["logits"].float(), dim=-1)
    if cfg.cls_soft_temp_m > 0:
        target_p = torch.softmax(-anchor_ade / cfg.cls_soft_temp_m, dim=-1)
        ce = -(target_p * logp).sum(-1)
    else:
        ce = -logp.gather(-1, k_star[..., None]).squeeze(-1)
    aw = has.float() * agent_w
    cls = (ce * aw).sum() / aw.sum().clamp_min(1.0)
    loss = nll + cfg.yaw_loss_weight * yaw_loss + cfg.cls_weight * cls

    with torch.no_grad():
        others = has & ~f["is_sdc"].bool()
        hf = others.float()
        ade_k, fde_k, _ = ade_fde(dec["pred_xy_k"], f["target_delta_xy"].float(), valid, rng)
        sel = select_modes(dec, cfg.num_out_modes, cfg.nms_dist_m, rng)
        top1 = dec["probs"].argmax(-1, keepdim=True)
        ent = -(dec["probs"] * dec["probs"].clamp_min(1e-8).log()).sum(-1)
        usage = torch.bincount(top1[others].flatten(), minlength=cfg.num_anchors).float()
        usage = usage / usage.sum().clamp_min(1.0)
        ego = (has & f["is_sdc"].bool()).float()
        metrics = {
            "loss_wm": float(loss), "nll": float(nll), "yaw_loss": float(yaw_loss), "cls_loss": float(cls),
            "min_ade_6": float(_masked_mean(ade_k.gather(-1, sel).amin(-1), hf)),
            "min_fde_6": float(_masked_mean(fde_k.gather(-1, sel).amin(-1), hf)),
            "miss_rate_6": float(_masked_mean((fde_k.gather(-1, sel).amin(-1) > 2.0).float(), hf)),
            "min_ade_all": float(_masked_mean(ade_k.amin(-1), hf)),
            "top1_ade": float(_masked_mean(ade_k.gather(-1, top1).squeeze(-1), hf)),
            "anchor_oracle_ade": float(_masked_mean(anchor_ade.amin(-1), hf)),
            "mode_entropy": float(_masked_mean(ent, hf)),
            "anchor_usage_entropy": float(-(usage * usage.clamp_min(1e-12).log()).sum()),
            "ego_min_ade_6": float(_masked_mean(ade_k.gather(-1, sel).amin(-1), ego)),
            "ego_top1_ade": float(_masked_mean(ade_k.gather(-1, top1).squeeze(-1), ego)),
            # Sigma of the assigned anchor (m) and the share of agents at the floor: a collapsed sigma
            # makes the NLL gradient ~ error / sigma^2 and lets those agents dominate the clipped gradient.
            "sigma_median_m": float(sigma[valid].median() * rng) if valid.any() else float("nan"),
            "sigma_at_floor": float(_masked_mean((sigma < 1.5 * cfg.sigma_eps).float().amax(-1), valid.float())),
            "agent_weight_mean": float(_masked_mean(agent_w, has.float())),
        }
    return loss, metrics


def _bce_soft(logit: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _masked_mean(F.binary_cross_entropy_with_logits(logit.float(), target, reduction="none"), w)


def _time_to_violation(labels: dict) -> torch.Tensor:
    """First violation step / horizon (1 = none)."""
    first = torch.minimum(labels["first_overlap"], labels["first_offroad"])
    return (first / labels["horizon"].float().clamp_min(1.0)[:, None]).clamp(max=1.0)


def _progress_norm(labels: dict, valid: torch.Tensor) -> torch.Tensor:
    """Route progress relative to the best valid candidate of the group (PDM's ego-progress ratio), in [-1, 1]."""
    p = labels["route_progress"]
    best = p.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True).clamp_min(1.0)
    return (p / best).clamp(-1.0, 1.0)


def _aux_loss(aux: dict, labels: dict, valid: torch.Tensor) -> torch.Tensor:
    w = valid.float()
    return (_masked_mean(F.smooth_l1_loss(aux["ttv"].float(), _time_to_violation(labels), reduction="none", beta=0.1), w)
            + _masked_mean(F.smooth_l1_loss(aux["progress"].float(), _progress_norm(labels, valid), reduction="none",
                                            beta=0.1), w))


@torch.no_grad()
def _pick_metrics(total: torch.Tensor, labels: dict, valid: torch.Tensor, unsafe: torch.Tensor) -> dict:
    neg = torch.finfo(torch.float32).min
    pick = total.float().masked_fill(~valid, neg).argmax(-1)
    ar = torch.arange(total.shape[0], device=total.device)
    return {"picked_unsafe": float(unsafe[ar, pick].float().mean()),
            "picked_overlap": float(labels["overlap"][ar, pick].float().mean()),
            "picked_offroad": float(labels["offroad"][ar, pick].float().mean()),
            "picked_route_progress": float(labels["route_progress"][ar, pick].mean()),
            "cand_unsafe_rate": float(_masked_mean(unsafe.float(), valid.float()))}


def score_loss(total: torch.Tensor, sub: torch.Tensor, labels: dict, valid: torch.Tensor, *,
               progress_margin: float = 2.0, w_sub: float = 1.0, w_pair: float = 1.0, w_list: float = 0.5,
               tau_overlap: float = 0.5, tau_offroad: float = 0.5, tau_comfort: float = 0.5,
               list_temp: float = 1.0, aux: dict | None = None, w_aux: float = 0.5) -> tuple[torch.Tensor, dict]:
    """total [B,C], sub [B,C,4] (overlap, offroad, progress, comfort logits), valid [B,C]."""
    w = valid.float()
    unsafe = labels["overlap"] | labels["offroad"]
    prog_norm = _progress_norm(labels, valid)

    # Soft sub-score targets (guidance, not a rule order).
    t_overlap = torch.sigmoid(-labels["clearance"] / tau_overlap)
    t_offroad = torch.sigmoid(labels["offroad_sd"] / tau_offroad)
    t_comfort = torch.sigmoid(labels["comfort"] / tau_comfort)
    l_sub = (_bce_soft(sub[..., 0], t_overlap, w) + _bce_soft(sub[..., 1], t_offroad, w)
             + _masked_mean(F.smooth_l1_loss(sub[..., 2].float(), prog_norm, reduction="none", beta=0.1), w)
             + 0.5 * _bce_soft(sub[..., 3], t_comfort, w))

    # Pairwise preferences: safe > unsafe; among safe, >= margin more route progress.
    vv = valid[:, :, None] & valid[:, None, :]
    safe = ~unsafe
    p = labels["route_progress"]
    safety_pairs = safe[:, :, None] & unsafe[:, None, :] & vv
    prog_pairs = safe[:, :, None] & safe[:, None, :] & ((p[:, :, None] - p[:, None, :]) >= progress_margin) & vv
    diff = total.float()[:, :, None] - total.float()[:, None, :]
    pl = F.softplus(-diff)
    l_safety = _masked_mean(pl, safety_pairs.float())
    l_prog = _masked_mean(pl, prog_pairs.float())
    has_s, has_p = safety_pairs.any(), prog_pairs.any()
    l_pair = (l_safety * has_s + l_prog * has_p) / (has_s.float() + has_p.float()).clamp_min(1.0)

    # Listwise: softmax over the group toward softmax(soft reward / T).
    reward = -5.0 * t_overlap - 5.0 * t_offroad + prog_norm - 0.5 * t_comfort
    neg = torch.finfo(torch.float32).min
    target = torch.softmax(reward.masked_fill(~valid, neg) / list_temp, -1)
    logp = F.log_softmax(total.float().masked_fill(~valid, neg), -1)
    has = valid.any(-1).float()
    l_list = (-(target * logp.masked_fill(~valid, 0.0)).sum(-1) * has).sum() / has.sum().clamp_min(1.0)

    loss = w_sub * l_sub + w_pair * l_pair + w_list * l_list
    l_aux = None
    if aux:
        l_aux = _aux_loss(aux, labels, valid)
        loss = loss + w_aux * l_aux
    with torch.no_grad():
        acc_pairs = safety_pairs | prog_pairs
        metrics = {"loss_score": float(loss), "score_sub": float(l_sub), "score_pair": float(l_pair),
                   "score_list": float(l_list),
                   "score_pair_acc": float(_masked_mean((diff > 0).float() + 0.5 * (diff == 0).float(), acc_pairs.float()))}
        metrics.update(_pick_metrics(total, labels, valid, unsafe))
        if l_aux is not None:
            metrics["score_aux"] = float(l_aux)
    return loss, metrics


def simple_utility(unsafe: torch.Tensor, ttv: torch.Tensor, severity: torch.Tensor, progress_norm: torch.Tensor, *,
                   a_unsafe: float = 5.0, late: float = 0.5, b_severity: float = 1.0, c_progress: float = 2.0) -> torch.Tensor:
    """U [B,C] encoding the preference tiers in one scalar (a_unsafe >> the other terms, so safe beats unsafe):
      -a * unsafe * (1 - late * ttv)    a violation counts less the later it starts (ttv = first / horizon)
      -b * unsafe * min(severity, 2)    and the milder it is (overlap depth + offroad depth, m)
      +c * route progress / group best"""
    return (-a_unsafe * unsafe.float() * (1.0 - late * ttv.clamp(0.0, 1.0))
            - b_severity * unsafe.float() * severity.clamp(0.0, 2.0) + c_progress * progress_norm)


def simple_score_loss(total: torch.Tensor, sub: torch.Tensor, labels: dict, valid: torch.Tensor, *,
                      w_safe: float = 1.0, temp: float = 1.0, late: float = 0.5, progress_margin: float = 2.0,
                      aux: dict | None = None, w_aux: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Listwise cross-entropy of softmax(S) toward softmax(U / temp) over each group's valid candidates, plus
    BCE of the overlap / offroad sub-logits toward time-weighted violation targets."""
    h = labels["horizon"].float().clamp_min(1.0)[:, None]
    unsafe = labels["overlap"] | labels["offroad"]
    severity = (-labels["clearance"]).clamp_min(0.0) + labels["offroad_sd"].clamp_min(0.0)
    u = simple_utility(unsafe, _time_to_violation(labels), severity, _progress_norm(labels, valid), late=late)
    neg = torch.finfo(torch.float32).min
    target = torch.softmax(u.float().masked_fill(~valid, neg) / temp, -1)
    logp = F.log_softmax(total.float().masked_fill(~valid, neg), -1)
    has = valid.any(-1).float()
    l_list = (-(target * logp.masked_fill(~valid, 0.0)).sum(-1) * has).sum() / has.sum().clamp_min(1.0)
    tw = lambda first_k, flag: flag.float() * (1.0 - late * (first_k / h).clamp(0.0, 1.0))  # noqa: E731
    w = valid.float()
    l_safe = (_bce_soft(sub[..., 0], tw(labels["first_overlap"], labels["overlap"]), w)
              + _bce_soft(sub[..., 1], tw(labels["first_offroad"], labels["offroad"]), w))
    loss = l_list + w_safe * l_safe
    l_aux = None
    if aux:
        l_aux = _aux_loss(aux, labels, valid)
        loss = loss + w_aux * l_aux
    with torch.no_grad():
        diff = total.float()[:, :, None] - total.float()[:, None, :]
        vv = valid[:, :, None] & valid[:, None, :]
        p = labels["route_progress"]
        pairs = vv & ~unsafe[:, :, None] & (unsafe[:, None, :] | (~unsafe[:, None, :] & ((p[:, :, None] - p[:, None, :]) >= progress_margin)))
        metrics = {"loss_score": float(loss), "score_list": float(l_list), "score_safe": float(l_safe),
                   "score_pair_acc": float(_masked_mean((diff > 0).float() + 0.5 * (diff == 0).float(), pairs.float()))}
        metrics.update(_pick_metrics(total, labels, valid, unsafe))
        if l_aux is not None:
            metrics["score_aux"] = float(l_aux)
    return loss, metrics
