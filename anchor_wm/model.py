"""AnchorWM network: scene encoder + anchor-query decoder + ego-plan scorer.

Encoder      MF-Gaussian's design (per-step agent MLP + GRU over the history, per-point map MLP +
             max-pool, transformer over agent, static-object and map tokens), trained from scratch.
Decoder      one query per (agent, anchor): encoder token + embed(anchor trajectory) (+ the route
             summary for the ego) -> MLP -> per-step residual around the anchor, sigma, yaw, and a logit
             scored from the query and its predicted waypoints. Decoding is in each agent's frame.
Scorer       candidate-step queries cross-attend to the scene tokens, to time-aligned (agent, mode)
             tokens of the decoded futures (with candidate-relative geometry) and to drivable-area
             boundary tokens. The WM tokens include the decoder's query embeddings, so with
             `grad_scale > 0` the scoring loss trains the decoder and encoder (joint);
             `grad_scale = 0` detaches every WM input of the scorer (separate).

The ego is agent slot 0 and its positions are rear-axle positions (nuPlan's planning frame); geometry
against boxes uses the box centre (geometry.ego_center).
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from anchor_wm.config import SUB_SCORES, AnchorWMConfig
from anchor_wm.geometry import (box_corners, candidate_features, edge_signed_distance, ego_center,
                                polyline_with_extension, project_progress, rotate_xy, sat_gap)

ENCODER_PREFIXES = ("agent_step_encoder.", "agent_history_gru.", "static_encoder.", "map_point_encoder.",
                    "agent_type_token", "static_type_token", "map_type_token", "scene_transformer.", "scene_norm.")
SEL_MIN_PROB = 0.05
SEL_SPEED_RATIOS = (0.0, 0.5, 1.0, 1.5)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TokenMLP(nn.Module):
    """Linear-GELU-Linear-LayerNorm."""

    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                    nn.LayerNorm(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity forward; gradient multiplied by `scale` (0 detaches)."""
    if scale == 0.0:
        return x.detach()
    if scale == 1.0:
        return x
    return _GradScale.apply(x, float(scale))


def gather_k(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B,N,K,...], idx [B,N,M] -> [B,N,M,...]."""
    rest = x.shape[3:]
    return x.gather(2, idx.view(*idx.shape, *([1] * len(rest))).expand(*idx.shape, *rest))


def _masked_pool(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return x.mean(dim), x.amax(dim)


def static_features(f: Mapping[str, torch.Tensor], cfg: AnchorWMConfig) -> torch.Tensor:
    """[B,S,F] static-object token inputs: xy (/max_range), sin / cos yaw, length / width (/max_width), type."""
    yaw = f["static_yaw"].float()[..., None]
    n_types = cfg.static_feature_dim - 6
    return torch.cat([f["static_xy"].float() / float(cfg.max_range), torch.sin(yaw), torch.cos(yaw),
                      f["static_lw"].float() / float(cfg.max_width),
                      F.one_hot(f["static_type"].long().clamp(0, n_types - 1), n_types).float()], -1)


def plan_features(f: Mapping[str, torch.Tensor], cfg: AnchorWMConfig) -> torch.Tensor:
    """[B,D] input of the ego's plan embedding: route summary points (/max_range) and their validity,
    the ego velocity and, with cfg.use_goal, the mission goal (direction, log distance, validity)."""
    pts = f["route_pts"].float() / float(cfg.max_range)
    pv = f["route_pts_valid"].float()
    feats = [(pts * pv[..., None]).flatten(1), pv, f["ego_vel"].float() / float(cfg.max_velocity)]
    if cfg.use_goal:
        g = f["goal_xy"].float()
        dist = g.norm(dim=-1, keepdim=True)
        gv = f["goal_valid"].float()[:, None]
        feats += [gv * g / dist.clamp_min(1e-3), gv * torch.log1p(dist) / 5.0, gv]
    return torch.cat(feats, -1)


def plan_feature_dim(cfg: AnchorWMConfig) -> int:
    return 3 * int(cfg.route_points) + 2 + (4 if cfg.use_goal else 0)


def route_step_features(cand_xy: torch.Tensor, cand_yaw: torch.Tensor, route: dict, chunk: int = 4096) -> torch.Tensor:
    """[B,K,T,3]: distance to the nearest route centreline point and sin / cos of the heading error to
    the route direction there (distance 2.0 / zeros when the scene has no route)."""
    b, k, t, _ = cand_xy.shape
    q = cand_xy.reshape(b, k * t, 2)
    rxy, rdir, rval = route["xy"], route["dir"], route["valid"].bool()
    dists, dirs = [], []
    for s0 in range(0, q.shape[1], chunk):
        qq = q[:, s0:s0 + chunk]
        d2 = (qq.unsqueeze(2) - rxy.unsqueeze(1)).square().sum(-1).masked_fill(~rval.unsqueeze(1), float("inf"))
        m, i = d2.min(-1)
        dists.append(m)
        dirs.append(torch.gather(rdir, 1, i.unsqueeze(-1).expand(-1, -1, 2)))
    dist = torch.sqrt(torch.cat(dists, 1).clamp(max=1e4)).reshape(b, k, t)
    rdir_q = torch.cat(dirs, 1).reshape(b, k, t, 2)
    c, s = torch.cos(cand_yaw), torch.sin(cand_yaw)
    err_sin = c * rdir_q[..., 1] - s * rdir_q[..., 0]
    err_cos = c * rdir_q[..., 0] + s * rdir_q[..., 1]
    has = rval.any(1).view(b, 1, 1)
    feat = torch.stack([(dist / 5.0).clamp(max=2.0), err_sin, err_cos], -1)
    return torch.where(has[..., None], feat, torch.tensor([2.0, 0.0, 0.0], device=cand_xy.device, dtype=feat.dtype))


def route_end_features(cand_xy: torch.Tensor, route: dict, step_feat: torch.Tensor) -> torch.Tensor:
    """[B,K,4]: progress of the endpoint along the route from the ego's projection (/50 m, linear and
    log), and the endpoint's route distance / heading error (from route_step_features)."""
    ext, cum = route["ext"], route["cum"]
    prog = project_progress(cand_xy[:, :, -1], ext, cum) - project_progress(torch.zeros_like(cand_xy[:, :1, 0]), ext, cum)
    has = route["valid"].bool().any(1)[:, None]
    prog = torch.where(has, prog, torch.zeros_like(prog))
    end = step_feat[:, :, -1]
    return torch.stack([prog / 50.0, torch.sign(prog) * torch.log1p(prog.abs()) / 5.0, end[..., 0], end[..., 2]], -1)


RULE_STEP_DIM, RULE_END_DIM = 8, 8


@torch.no_grad()
def rule_features(cand_xy: torch.Tensor, cand_yaw: torch.Tensor, ego_lw: torch.Tensor, wm: dict, statics: dict,
                  edges: dict, edge_budget: float = 8e6, pair_budget: float = 4e5) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact rule geometry of each candidate step: per step [B,K,T,RULE_STEP_DIM], per candidate
    [B,K,RULE_END_DIM]. Fixed inputs (no gradient), computed in fp32 whatever the autocast.
    `cand_xy` are box centres.

    Forecasts: SAT box gaps of the ego box against every selected (agent, mode) box -> smallest gap
    (any mode / modes with p >= SEL_MIN_PROB) and independent-agent collision probabilities
    1 - prod_a (1 - sum_m p_am * c(gap)) with c = sigmoid(-gap / 0.25), 1[gap < 0] and a 1 m near-miss
    sigmoid. Static objects (`statics`: xy / yaw / lw / valid at t = 0) count as certain obstacles in all of
    these. Drivable area: signed distance of the ego box corners (max over corners, > 0 offroad),
    clipped, as an offroad sigmoid and a 0.5 m near-boundary sigmoid. Summary: the per-step maxima /
    minima, the fraction of the horizon before the first hard violation (1 = none) and the mean soft
    collision probability. Only the (agent, mode) slots in wm["slots"] are tested."""
    with torch.autocast(cand_xy.device.type, enabled=False):
        cand_xy, cand_yaw, ego_lw = cand_xy.float(), cand_yaw.float(), ego_lw.float()
        b, k, t, _ = cand_xy.shape
        xy, yaw = wm["xy"].float(), wm["yaw"].float()
        a, m = xy.shape[1], xy.shape[2]
        probs = wm["probs"].float()                                                           # [B,A,M]
        live = wm["valid"][:, :, None] & (probs > 0)
        ec = box_corners(cand_xy, cand_yaw, ego_lw[:, None, None].expand(b, k, t, 2))          # [B,K,T,4,2]
        size = wm["size"].float()[:, :, None, None].expand(b, a, m, t, 2)
        oc = box_corners(xy, yaw, size).permute(0, 3, 1, 2, 4, 5).reshape(b, 1, t, a * m, 4, 2)
        o_yaw = yaw.permute(0, 3, 1, 2).reshape(b, 1, t, a * m)
        slots = wm["slots"]
        oc, o_yaw = oc[:, :, :, slots], o_yaw[:, :, :, slots]
        n_slot = oc.shape[3]
        n_step = max(1, int(pair_budget // max(1, b * k * t)))
        gap = torch.cat([sat_gap(ec.unsqueeze(3), cand_yaw.unsqueeze(-1), oc[:, :, :, n0:n0 + n_step],
                                 o_yaw[:, :, :, n0:n0 + n_step]) for n0 in range(0, n_slot, n_step)], -1)
        gap = torch.full((b, k, t, a * m), 50.0, device=gap.device, dtype=gap.dtype).index_copy(-1, slots, gap)
        gap = gap.view(b, k, t, a, m)
        big = torch.full_like(gap, 50.0)
        g_min = torch.where(live[:, None, None], gap, big).amin((-1, -2))
        likely = live & (probs >= SEL_MIN_PROB)
        g_likely = torch.where(likely[:, None, None], gap, big).amin((-1, -2))
        p = probs[:, None, None]                                                              # [B,1,1,A,M]

        def ind(c):
            # 1 - prod_a (1 - q_a), multiplied out over agents.
            q = (p * c).sum(-1).clamp(0, 1)
            keep = torch.ones_like(q[..., 0])
            for i in range(q.shape[-1]):
                keep = keep * (1 - q[..., i])
            return 1 - keep

        # Static objects: boxes fixed at their t = 0 pose, probability 1.
        s_n = statics["xy"].shape[1]
        sc = box_corners(statics["xy"].float(), statics["yaw"].float(), statics["lw"].float())[:, None, None]
        s_step = max(1, int(pair_budget // max(1, b * k * t)))
        s_gap = torch.cat([sat_gap(ec.unsqueeze(3), cand_yaw.unsqueeze(-1), sc[:, :, :, n0:n0 + s_step],
                                   statics["yaw"].float()[:, None, None, n0:n0 + s_step]) for n0 in range(0, s_n, s_step)], -1)
        s_gap = s_gap.masked_fill(~statics["valid"].bool()[:, None, None], 50.0)              # [B,K,T,S]
        g_min = torch.minimum(g_min, s_gap.amin(-1))
        g_likely = torch.minimum(g_likely, s_gap.amin(-1))
        with_static = lambda p_wm, c: 1 - (1 - p_wm) * torch.prod(1 - c, -1)  # noqa: E731

        p_soft = with_static(ind(torch.sigmoid(-gap / 0.25)), torch.sigmoid(-s_gap / 0.25))
        p_hard = with_static(ind((gap < 0).float()), (s_gap < 0).float())
        p_near = with_static(ind(torch.sigmoid(-(gap - 1.0) / 0.25)), torch.sigmoid(-(s_gap - 1.0) / 0.25))

        q = ec.reshape(b, k * t * 4, 2)
        sd = edge_signed_distance(q, edges, chunk=1 << 20, budget=edge_budget).view(b, k, t, 4).amax(-1).clamp(-5.0, 5.0)
        off = torch.sigmoid(sd / 0.1)
        step = torch.stack([g_min.clamp(-2, 10) / 5, g_likely.clamp(-2, 10) / 5, p_soft, p_hard, p_near,
                            sd / 2, off, torch.sigmoid((sd + 0.5) / 0.2)], -1)
        hard = (p_hard > 0.5) | (sd > 0)
        idx = torch.arange(1, t + 1, device=cand_xy.device, dtype=cand_xy.dtype)
        first = torch.where(hard, idx, torch.full_like(idx, t + 1.0)).amin(-1)
        end = torch.stack([p_soft.amax(-1), p_hard.amax(-1), p_near.amax(-1), off.amax(-1),
                           g_min.amin(-1).clamp(-2, 10) / 5, sd.amax(-1) / 2, ((first - 1) / t).clamp(max=1.0),
                           p_soft.mean(-1)], -1)
    return step, end


def _rotate_cov(cov: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """(sxx, syy, sxy) expressed in a frame rotated by -angle (i.e. R(-a) S R(-a)^T)."""
    c, s = torch.cos(angle), torch.sin(angle)
    sxx, syy, sxy = cov[..., 0], cov[..., 1], cov[..., 2]
    return torch.stack([c * c * sxx + 2 * c * s * sxy + s * s * syy,
                        s * s * sxx - 2 * c * s * sxy + c * c * syy,
                        -c * s * sxx + c * s * syy + (c * c - s * s) * sxy], -1)


class _Temporal(nn.Module):
    """Two residual depthwise Conv1d blocks over the step axis of [..., T, H]."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = x.reshape(-1, shape[-2], shape[-1])
        for conv, norm in zip(self.convs, self.norms):
            y = norm(y + F.gelu(conv(y.transpose(1, 2)).transpose(1, 2)))
        return y.reshape(shape)


class _GeoAttention(nn.Module):
    """Candidate-step queries over tokens with a geometry MLP in logits and values.

    q [B,K,T,H]; tokens [B,Tt,N,H] (Tt = T for time-aligned tokens, 1 for static);
    geo [B,K,T,N,F]; bias [B,1|K,T|1,N] additive logit bias (or None); mask [B,1,Tt,N].
    """

    def __init__(self, hidden: int, heads: int, geo_dim: int, geo_hidden: int = 32) -> None:
        super().__init__()
        self.heads, self.dh = heads, hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.geo = nn.Sequential(nn.Linear(geo_dim, geo_hidden), nn.GELU(), nn.Linear(geo_hidden, geo_hidden))
        self.geo_logit = nn.Linear(geo_hidden, heads)
        self.geo_value = nn.Linear(geo_hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden))

    def forward(self, q, tokens, geo, bias, mask):
        b, k, t, h = q.shape
        nh, dh = self.heads, self.dh
        static = tokens.shape[1] == 1
        qh = self.q(q).view(b, k, t, nh, dh)
        kh = self.k(tokens).view(b, tokens.shape[1], tokens.shape[2], nh, dh)
        vh = self.v(tokens).view(b, tokens.shape[1], tokens.shape[2], nh, dh)
        g = self.geo(geo)
        if static:
            logits = torch.einsum("bktnd,bmnd->bktmn", qh, kh[:, 0])
        else:
            logits = torch.einsum("bktnd,btmnd->bktmn", qh, kh)
        logits = logits / math.sqrt(dh) + self.geo_logit(g)
        if bias is not None:
            logits = logits + bias.unsqueeze(-1)
        keep = mask.unsqueeze(-1).to(logits.dtype)
        logits = logits.masked_fill(keep == 0, -1e4)
        attn = torch.softmax(logits, dim=3) * keep
        if static:
            v_tok = torch.einsum("bktmn,bmnd->bktnd", attn, vh[:, 0])
        else:
            v_tok = torch.einsum("bktmn,btmnd->bktnd", attn, vh)
        g_pooled = torch.einsum("bktm,bktmg->bktg", attn.mean(-1), g)
        return self.out(v_tok.reshape(b, k, t, h) + self.geo_value(g_pooled))


class PlanScorer(nn.Module):
    """Scores ego candidates [B,C,T] against the decoded futures. Outputs a total score (higher is
    better, trained by ranking) and sub-score logits for SUB_SCORES."""

    def __init__(self, cfg: AnchorWMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, c, heads = cfg.scorer_hidden, cfg.hidden_dim, cfg.scorer_heads
        self.pos_scale, self.vel_scale = float(cfg.scorer_pos_scale), float(cfg.scorer_vel_scale)
        self.scene_proj = nn.Sequential(nn.Linear(c, h), nn.LayerNorm(h))
        self.plan_embed = _TokenMLP(plan_feature_dim(cfg), h)
        self.traj_embed = _TokenMLP(6, h)
        self.traj_gru = nn.GRU(h, h, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(h, heads, dropout=cfg.dropout, batch_first=True)
        self.query_proj = nn.Sequential(nn.Linear(c, h), nn.LayerNorm(h))
        self.mode_embed = _TokenMLP(10, h)
        self.mode_temporal = _Temporal(h)
        self.prob_embed = nn.Linear(1, h)
        self.wm_attn = _GeoAttention(h, heads, 12)
        self.edge_embed = _TokenMLP(5, h)
        self.edge_attn = _GeoAttention(h, heads, 7)
        self.trunk = nn.Sequential(nn.LayerNorm(7 * h), nn.Linear(7 * h, h), nn.GELU(), nn.Dropout(cfg.dropout))
        self.sub_head = nn.Linear(h, len(SUB_SCORES))
        self.total_head = nn.Linear(h, 1)
        self.route_step = _MLP(3, h, h)             # added to the candidate step embedding
        self.route_end = _MLP(4, h, h)              # added to the pooled representation
        self.rule_step = _MLP(RULE_STEP_DIM, h, h)
        self.rule_end = _MLP(RULE_END_DIM, h, h)
        self.ttv_head = nn.Linear(h, 1)             # time to first violation / horizon (1 = none)
        self.progress_head = nn.Linear(h, 1)        # route progress / best in the group

    # ---- candidate-independent context ------------------------------------------------------

    @torch.no_grad()
    def select_agents(self, f: Mapping[str, torch.Tensor], mean: torch.Tensor, probs: torch.Tensor):
        """Agents [B,A] nearest to the ego now or along constant-velocity ego rollouts under any mode
        with p >= 0.05."""
        cfg = self.cfg
        b, n, m, t, _ = mean.shape
        valid = f["agent_valid"].bool() & ~f["is_sdc"].bool()
        speed = f["ego_vel"].float().norm(dim=-1)                                             # [B]
        steps = torch.arange(1, t + 1, device=mean.device, dtype=mean.dtype) * cfg.future_dt
        ratios = torch.tensor(SEL_SPEED_RATIOS, device=mean.device, dtype=mean.dtype)
        ego_x = speed[:, None, None] * ratios[None, :, None] * steps[None, None, :]           # [B,R,T]
        ego = torch.stack([ego_x, torch.zeros_like(ego_x)], -1)                               # [B,R,T,2]
        d_mode = (mean[:, :, :, None] - ego[:, None, None]).norm(dim=-1).amin(dim=(3, 4))     # [B,N,M]
        d_fut = torch.where(probs >= SEL_MIN_PROB, d_mode, torch.full_like(d_mode, float("inf"))).amin(-1)
        d = torch.minimum(f["current_local_xy"].float().norm(dim=-1), d_fut)
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        a = min(int(cfg.scorer_agents), n)
        dist, idx = torch.topk(d, a, dim=1, largest=False)
        return idx, torch.isfinite(dist)

    def context(self, enc: dict, dec: dict, f: Mapping[str, torch.Tensor], scale: float) -> dict:
        cfg = self.cfg
        rng = float(cfg.max_range)
        tokens = self.scene_proj(grad_scale(enc["scene_tokens"], scale))
        plan_token = self.plan_embed(plan_features(f, cfg))
        tokens = torch.cat([tokens, plan_token[:, None]], 1)
        valid = torch.cat([enc["scene_valid"], torch.ones_like(enc["scene_valid"][:, :1])], 1)

        # Top modes per agent, then the agents that matter for the ego (nearest first).
        ms = int(cfg.scorer_near_modes)
        top_p, top_k = torch.topk(dec["probs"].detach(), ms, dim=-1)                          # [B,N,M]
        cur = f["current_local_xy"].float()
        mean_all = cur[:, :, None, None] + gather_k(dec["pred_xy_k"].detach(), top_k) * rng
        aidx, avalid = self.select_agents(f, mean_all, top_p)                                 # [B,A]
        a = aidx.shape[1]
        g2 = lambda x: x.gather(1, aidx.view(*aidx.shape, *([1] * (x.dim() - 2))).expand(  # noqa: E731
            *aidx.shape, *x.shape[2:]))
        k_sel = g2(top_k)                                                                     # [B,A,M]
        sel = lambda x: gather_k(g2(x), k_sel)  # noqa: E731

        xy = g2(cur)[:, :, None, None] + grad_scale(sel(dec["pred_xy_k"]), scale) * rng       # [B,A,M,T,2]
        yaw_vec = grad_scale(sel(dec["yaw_vec_k"]), scale)
        cur_yaw = g2(f["current_yaw_local"].float())
        yaw = cur_yaw[:, :, None, None] + torch.atan2(yaw_vec[..., 0], yaw_vec[..., 1])
        sigma = grad_scale(sel(dec["sigma_k"]), scale) * rng                                  # agent frame, m
        probs = grad_scale(sel(dec["probs"]), scale)
        keep, slots = self._mode_slots(a, ms, probs.device)
        probs = probs * keep                                                                  # far agents: top scorer_modes
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-6)
        probs = probs * avalid[..., None]
        size = g2(f["agent_history"][:, :, -1, 6:8].float() * cfg.max_width)                  # length, width
        query = self.query_proj(grad_scale(sel(dec["queries"]), scale))                       # [B,A,M,H]

        wm = {"xy": xy, "yaw": yaw, "sigma": sigma, "cov_yaw": cur_yaw, "probs": probs, "size": size,
              "valid": avalid, "current": g2(cur), "slots": slots}
        wm_tok = self._wm_tokens(wm, query)
        e_mask = f["tok_edge_valid"].bool()
        edges = {"xy": f["tok_edge_xy"].float(), "dir": f["tok_edge_dir"].float(), "mask": e_mask}
        edge_tok = self.edge_embed(torch.cat([edges["xy"] / self.pos_scale, edges["dir"], e_mask[..., None].float()], -1))
        rxy = f["route_xy"].float()
        d = torch.diff(rxy, dim=1)
        d = torch.cat([d, d[:, -1:]], 1)
        ext, cum = polyline_with_extension(rxy)
        route = {"xy": rxy, "dir": d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6), "valid": f["route_valid"].bool(),
                 "ext": ext, "cum": cum}
        label_edges = {k: f[k] for k in ("edge_xy", "edge_dir", "edge_prev_dir", "edge_same_prev", "edge_valid")}
        statics = {"xy": f["static_xy"], "yaw": f["static_yaw"], "lw": f["static_lw"], "valid": f["static_valid"]}
        return dict(tokens=tokens, valid=valid, plan_token=plan_token, wm=wm, wm_tok=wm_tok, edges=edges,
                    edge_tok=edge_tok, ego_lw=f["ego_lw"].float(), agent_idx=aidx, mode_idx=k_sel, route=route,
                    label_edges=label_edges, statics=statics)

    def _mode_slots(self, a: int, m: int, device):
        """Mode keep mask [A,M] and flat (agent, mode) slot indices: the first scorer_near_agents agents
        keep all m (= scorer_near_modes) modes, the others the top scorer_modes."""
        cfg = self.cfg
        keep = ((torch.arange(a, device=device)[:, None] < int(cfg.scorer_near_agents))
                | (torch.arange(m, device=device)[None, :] < int(cfg.scorer_modes)))
        return keep, keep.flatten().nonzero().squeeze(-1)

    def _wm_tokens(self, wm: dict, query: torch.Tensor) -> torch.Tensor:
        xy = wm["xy"]
        b, a, m, t, _ = xy.shape
        prev = torch.cat([wm["current"][:, :, None, None].expand(b, a, m, 1, 2), xy[..., :-1, :]], dim=3)
        vel = (xy - prev) / self.cfg.future_dt
        size = wm["size"][:, :, None, None].expand(b, a, m, t, 2)
        feats = [xy / self.pos_scale, torch.sin(wm["yaw"])[..., None], torch.cos(wm["yaw"])[..., None],
                 vel / self.vel_scale, size / 10.0, torch.log(wm["sigma"].clamp(min=1e-2))]
        tok = self.mode_temporal(self.mode_embed(torch.cat(feats, -1)))                       # [B,A,M,T,H]
        tok = tok + query[:, :, :, None] + self.prob_embed(torch.log(wm["probs"].clamp(min=1e-4))[..., None])[:, :, :, None]
        tok = tok.permute(0, 3, 1, 2, 4).reshape(b, t, a * m, -1)                            # [B,T,A*M,H]
        return tok[:, :, wm["slots"]]                                                         # [B,T,S,H]

    # ---- candidate geometry -------------------------------------------------------------------

    def _wm_geometry(self, cand_xy, cand_yaw, ego_lw, wm: dict) -> torch.Tensor:
        xy = wm["xy"]
        b, a, m, t, _ = xy.shape
        agent_xy = xy.permute(0, 3, 1, 2, 4).reshape(b, 1, t, a * m, 2)
        agent_yaw = wm["yaw"].permute(0, 3, 1, 2).reshape(b, 1, t, a * m)
        size = wm["size"][:, :, None].expand(b, a, m, 2).reshape(b, 1, 1, a * m, 2)
        # The covariance is R(cov_yaw) diag(sigma^2) R(cov_yaw)^T (+0.05 m^2 I), so the Mahalanobis distance,
        # determinant and candidate-frame variances are taken in that eigenbasis (forming them from the
        # ego-frame (sxx, syy, sxy) cancels catastrophically in fp32 for thin, long Gaussians).
        var = (wm["sigma"].permute(0, 3, 1, 2, 4).reshape(b, 1, t, a * m, 2)).square() + 0.05            # [B,1,T,N,2]
        cov_yaw = wm["cov_yaw"][:, :, None].expand(b, a, m).reshape(b, 1, 1, a * m)
        slots = wm["slots"]
        agent_xy, agent_yaw, size = agent_xy[:, :, :, slots], agent_yaw[:, :, :, slots], size[:, :, :, slots]
        var, cov_yaw = var[:, :, :, slots], cov_yaw[:, :, :, slots]
        d = agent_xy - cand_xy.unsqueeze(3)                                                   # [B,K,T,N,2]
        c, s = torch.cos(cand_yaw).unsqueeze(-1), torch.sin(cand_yaw).unsqueeze(-1)
        dx, dy = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        rel = agent_yaw - cand_yaw.unsqueeze(-1)
        reach = 0.5 * size.norm(dim=-1) + 0.5 * ego_lw.norm(dim=-1).view(b, 1, 1, 1)
        geo = [dx / 10.0, dy / 10.0, torch.log1p(dist), torch.sin(rel), torch.cos(rel), (dist - reach) / 5.0,
               torch.exp(-dist / 2.0), torch.exp(-(dist - reach).clamp(min=0) / 1.0)]
        ca, sa = torch.cos(cov_yaw), torch.sin(cov_yaw)
        along, across = ca * d[..., 0] + sa * d[..., 1], -sa * d[..., 0] + ca * d[..., 1]          # agent-heading frame
        maha = along.square() / var[..., 0] + across.square() / var[..., 1]
        det = var[..., 0] * var[..., 1]
        phi = cand_yaw.unsqueeze(-1) - cov_yaw
        cp2, sp2 = torch.cos(phi).square(), torch.sin(phi).square()
        cand_var_x, cand_var_y = cp2 * var[..., 0] + sp2 * var[..., 1], sp2 * var[..., 0] + cp2 * var[..., 1]
        geo += [torch.log1p(maha), 0.5 * torch.log(cand_var_x.clamp(min=1e-4)),
                0.5 * torch.log(cand_var_y.clamp(min=1e-4)), 0.25 * torch.log(det).expand_as(maha)]
        return torch.stack(geo, -1)

    @staticmethod
    def _edge_geometry(cand_xy, cand_yaw, ego_lw, edges: dict) -> torch.Tensor:
        exy = edges["xy"][:, None, None]
        edir = edges["dir"][:, None, None]
        d = cand_xy.unsqueeze(3) - exy
        lateral = edir[..., 0] * d[..., 1] - edir[..., 1] * d[..., 0]
        along = edir[..., 0] * d[..., 0] + edir[..., 1] * d[..., 1]
        dist = torch.sqrt((d * d).sum(-1) + 1e-6)
        rel = torch.atan2(edir[..., 1], edir[..., 0]) - cand_yaw.unsqueeze(-1)
        half_w = 0.5 * ego_lw[:, 1].view(-1, 1, 1, 1)
        return torch.stack([lateral / 5.0, along / 10.0, torch.log1p(dist), torch.sin(rel), torch.cos(rel),
                            (lateral.abs() - half_w) / 2.0, torch.exp(-dist / 2.0)], -1)

    # ---- scoring ------------------------------------------------------------------------------

    def forward(self, ctx: dict, cand_xy: torch.Tensor, cand_yaw: torch.Tensor,
                chunk: int = 0) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """cand_xy [B,C,T,2] (m, ego frame, rear axle), headings [B,C,T] -> total [B,C], sub-logits [B,C,S]
        and the aux heads {"ttv", "progress"} [B,C]."""
        count = cand_xy.shape[1]
        step = chunk if chunk and chunk < count else count
        totals, subs, auxs = [], [], []
        for s0 in range(0, count, step):
            sl = slice(s0, s0 + step)
            tot, sub, aux = self._score_chunk(ctx, cand_xy[:, sl], cand_yaw[:, sl])
            totals.append(tot)
            subs.append(sub)
            auxs.append(aux)
        return torch.cat(totals, 1), torch.cat(subs, 1), {key: torch.cat([a[key] for a in auxs], 1) for key in auxs[0]}

    def _score_chunk(self, ctx: dict, cand_xy: torch.Tensor, cand_yaw: torch.Tensor):
        bsz, count, steps, _ = cand_xy.shape
        cfg = self.cfg
        center = ego_center(cand_xy, cand_yaw)                       # geometry against boxes / boundary
        feats = candidate_features(cand_xy, cand_yaw, cfg.future_dt, self.pos_scale, self.vel_scale)
        traj = self.traj_embed(feats.reshape(bsz * count, steps, -1))
        r_step = route_step_features(cand_xy, cand_yaw, ctx["route"])
        rule_step, rule_end = rule_features(center, cand_yaw, ctx["ego_lw"], ctx["wm"], ctx["statics"], ctx["label_edges"])
        traj = traj + (self.route_step(r_step) + self.rule_step(rule_step)).reshape(bsz * count, steps, -1)
        encoded, _ = self.traj_gru(traj)
        cross, _ = self.cross_attn(encoded.reshape(bsz, count * steps, -1), ctx["tokens"], ctx["tokens"],
                                   key_padding_mask=~ctx["valid"], need_weights=False)
        cross = cross.reshape(bsz * count, steps, -1)
        q = encoded.reshape(bsz, count, steps, -1)
        pooled = [encoded[:, -1], cross.mean(1), ctx["plan_token"].repeat_interleave(count, dim=0)]

        e_geo = self._edge_geometry(center, cand_yaw, ctx["ego_lw"], ctx["edges"])
        e_out = self.edge_attn(q, ctx["edge_tok"][:, None], e_geo, None, ctx["edges"]["mask"][:, None, None, :])
        pooled += [x.reshape(bsz * count, -1) for x in _masked_pool(e_out, 2)]

        wm = ctx["wm"]
        b, a, m, t, _ = wm["xy"].shape
        w_geo = self._wm_geometry(center, cand_yaw, ctx["ego_lw"], wm)
        mask = (wm["valid"][:, :, None] & (wm["probs"] > 0)).reshape(b, 1, 1, a * m)[..., wm["slots"]]
        bias = torch.log(wm["probs"].clamp(min=1e-4)).reshape(b, 1, 1, a * m)[..., wm["slots"]]
        w_out = self.wm_attn(q, ctx["wm_tok"], w_geo, bias, mask.expand(b, 1, t, mask.shape[-1]))
        pooled += [x.reshape(bsz * count, -1) for x in _masked_pool(w_out, 2)]

        z = self.trunk(torch.cat(pooled, -1))
        z = z + (self.route_end(route_end_features(cand_xy, ctx["route"], r_step)) + self.rule_end(rule_end)).reshape(bsz * count, -1)
        # Heads in fp32: bf16 resolves a score of ~10 only to ~0.06, which ties near-equal candidates.
        with torch.autocast(z.device.type, enabled=False):
            z = z.float()
            aux = {"ttv": self.ttv_head(z).squeeze(-1).reshape(bsz, count),
                   "progress": self.progress_head(z).squeeze(-1).reshape(bsz, count)}
            return self.total_head(z).squeeze(-1).reshape(bsz, count), self.sub_head(z).reshape(bsz, count, -1), aux


class AnchorWorldModel(nn.Module):
    def __init__(self, cfg: AnchorWMConfig, anchors_m: torch.Tensor):
        super().__init__()
        self.config = cfg
        c, t = int(cfg.hidden_dim), int(cfg.future_steps)
        if tuple(anchors_m.shape[1:]) != (cfg.num_anchors, t, 2):
            raise ValueError(f"anchors {tuple(anchors_m.shape)} != [G,{cfg.num_anchors},{t},2]")

        # ---- encoder ----
        self.agent_step_encoder = _MLP(cfg.agent_feature_dim, c, c, dropout=cfg.dropout)
        self.agent_history_gru = nn.GRU(c, c, batch_first=True)
        self.static_encoder = _MLP(cfg.static_feature_dim, c, c, dropout=cfg.dropout)
        self.map_point_encoder = _MLP(cfg.map_feature_dim, c, c, dropout=cfg.dropout)
        self.agent_type_token = nn.Parameter(torch.zeros(1, 1, c))
        self.static_type_token = nn.Parameter(torch.zeros(1, 1, c))
        self.map_type_token = nn.Parameter(torch.zeros(1, 1, c))
        layer = nn.TransformerEncoderLayer(d_model=c, nhead=cfg.num_heads, dim_feedforward=4 * c,
                                           dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.scene_transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_scene_layers, enable_nested_tensor=False)
        self.scene_norm = nn.LayerNorm(c)

        # ---- anchor decoder ----
        self.register_buffer("anchors", (anchors_m.float() / float(cfg.max_range)).contiguous())
        self.anchor_embed = _MLP(t * 2, c, c, dropout=cfg.dropout)
        self.plan_embed = _MLP(plan_feature_dim(cfg), c, c, dropout=cfg.dropout)
        self.query_fuse = _MLP(c, c, c, dropout=cfg.dropout)
        self.traj_head = _MLP(c, c, t * 6, dropout=cfg.dropout)
        last = self.traj_head.net[-1]
        nn.init.normal_(last.weight, std=0.01)     # residual ~0 at init: predictions start at the anchors
        nn.init.zeros_(last.bias)
        bias = last.bias.data.view(t, 6)
        bias[:, 2:4] = float(cfg.sigma_raw_bias)
        bias[:, 5] = 1.0                           # yaw delta starts at 0 (sin 0, cos 1)
        waypoint_idx = [(i + 1) * t // 5 - 1 for i in range(5)]
        self.register_buffer("waypoint_idx", torch.tensor(waypoint_idx, dtype=torch.long), persistent=False)
        self.traj_embed = _MLP(2 * len(waypoint_idx), c, c, dropout=cfg.dropout)
        self.mode_cls = _MLP(2 * c, c, 1, dropout=cfg.dropout)
        nn.init.zeros_(self.mode_cls.net[-1].weight)
        nn.init.zeros_(self.mode_cls.net[-1].bias)

        self.scorer = PlanScorer(cfg)

    # ---- encoder ------------------------------------------------------------------------------

    @staticmethod
    def _masked_map_pool(point_tokens, point_valid):
        mask = point_valid.bool()
        pooled = torch.where(mask[..., None], point_tokens, torch.full_like(point_tokens, -1.0e4)).max(dim=2).values
        seg_valid = mask.any(dim=2)
        return torch.where(seg_valid[..., None], pooled, torch.zeros_like(pooled)), seg_valid

    def encode(self, f: Mapping[str, torch.Tensor]) -> dict:
        history = f["agent_history"].float()
        history_valid = f["agent_history_valid"].bool()
        agent_valid = f["agent_valid"].bool()
        b, n, h, _ = history.shape
        c = int(self.config.hidden_dim)

        step_tokens = self.agent_step_encoder(history) * history_valid[..., None].to(history.dtype)
        _, h_n = self.agent_history_gru(step_tokens.reshape(b * n, h, c))
        agent_tokens = h_n[-1].reshape(b, n, c) * agent_valid[..., None].to(history.dtype) + self.agent_type_token
        map_tokens, map_seg_valid = self._masked_map_pool(self.map_point_encoder(f["map_features"].float()),
                                                          f["map_valid"].bool())
        map_tokens = map_tokens + self.map_type_token
        static_valid = f["static_valid"].bool()
        static_tokens = (self.static_encoder(static_features(f, self.config)) + self.static_type_token) * static_valid[..., None].to(history.dtype)

        scene_tokens = torch.cat([agent_tokens, static_tokens, map_tokens], dim=1)
        scene_valid = torch.cat([agent_valid, static_valid, map_seg_valid], dim=1)
        scene_tokens = self.scene_norm(self.scene_transformer(scene_tokens, src_key_padding_mask=~scene_valid))
        latent = scene_tokens[:, :n] * agent_valid[..., None].to(scene_tokens.dtype)
        return {"scene_tokens": scene_tokens, "scene_valid": scene_valid, "latent_agent": latent}

    # ---- decoder ------------------------------------------------------------------------------

    def decode(self, f: Mapping[str, torch.Tensor], enc: dict) -> dict:
        cfg = self.config
        latent = enc["latent_agent"]
        b, n, c = latent.shape
        k, t = int(cfg.num_anchors), int(cfg.future_steps)
        agent_valid = f["agent_valid"].bool()
        is_sdc = f["is_sdc"].bool()

        group = f["object_types"].long().clamp(0, self.anchors.shape[0] - 1)                 # ego is a vehicle
        anchors = self.anchors[group]                                                         # [B,N,K,T,2]
        anchor_feat = self.anchor_embed(self.anchors.flatten(-2))[group]                      # [B,N,K,C]
        plan = self.plan_embed(plan_features(f, cfg))
        latent = latent + is_sdc[..., None].to(latent.dtype) * plan[:, None]
        queries = self.query_fuse(latent[:, :, None, :] + anchor_feat)                        # [B,N,K,C]
        raw = self.traj_head(queries).view(b, n, k, t, 6)

        xy_agent = anchors + raw[..., :2]
        heading_sc = f["agent_history"][:, :, -1, 4:6].float()                                # sin, cos rel. ego
        heading_sc = torch.where(agent_valid[..., None], heading_sc, torch.tensor([0.0, 1.0], device=latent.device))
        xy_ego = rotate_xy(xy_agent, heading_sc[..., 0, None, None], heading_sc[..., 1, None, None])
        sigma = F.softplus(raw[..., 2:4]) + float(cfg.sigma_eps)
        yaw_vec = F.normalize(raw[..., 4:6], dim=-1, eps=1e-6)

        waypoints = xy_agent[..., self.waypoint_idx, :].flatten(-2)
        logits = self.mode_cls(torch.cat([queries, self.traj_embed(waypoints)], -1)).squeeze(-1)
        logits = logits.masked_fill(~agent_valid[..., None], 0.0)
        probs = torch.softmax(logits.float(), dim=-1) * agent_valid[..., None]
        return {
            "pred_xy_agent_k": xy_agent,      # [B,N,K,T,2] normalized, agent frame
            "pred_xy_k": xy_ego,              # [B,N,K,T,2] normalized displacement, ego frame
            "sigma_k": sigma,                 # [B,N,K,T,2] normalized, agent frame
            "yaw_vec_k": yaw_vec,             # [B,N,K,T,2] (sin, cos) of yaw change
            "logits": logits, "probs": probs, # [B,N,K]
            "queries": queries,               # [B,N,K,C]
            "anchors_agent": anchors,         # [B,N,K,T,2] normalized, agent frame
            "heading_sc": heading_sc,
        }

    def forward(self, f: Mapping[str, torch.Tensor]) -> tuple[dict, dict]:
        enc = self.encode(f)
        return enc, self.decode(f, enc)


@torch.no_grad()
def select_modes(dec: dict, m: int, nms_dist_m: float, max_range: float) -> torch.Tensor:
    """Planner modes [B,N,m]: greedy by probability with endpoint NMS; when every remaining anchor is
    suppressed the next most probable unselected one is taken."""
    probs = dec["probs"]
    end = dec["pred_xy_k"][..., -1, :] * max_range                                            # [B,N,K,2]
    suppressed = torch.zeros_like(probs, dtype=torch.bool)
    selected = torch.zeros_like(probs, dtype=torch.bool)
    picks = []
    for _ in range(m):
        score = torch.where(suppressed, probs - 10.0, probs).masked_fill(selected, float("-inf"))
        pick = score.argmax(-1)                                                               # [B,N]
        picks.append(pick)
        selected.scatter_(-1, pick[..., None], True)
        pe = end.gather(2, pick[..., None, None].expand(-1, -1, 1, 2))
        suppressed |= (end - pe).norm(dim=-1) < nms_dist_m
    return torch.stack(picks, -1)
