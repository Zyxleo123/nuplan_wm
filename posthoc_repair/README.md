# `posthoc_repair`: the o_am repair on nuPlan closed-loop simulation

The no-rule post-hoc repair from `yixiz/waymax_rs` (o_am: `rl/diffusion_ft/lateral_opt.py`,
`planner/posthoc_repair.py`), ported to torch and run inside nuPlan's `run_simulation` on Diffusion Planner's
plans. Every replan, Diffusion Planner's 80-pose plan (0.1 s, ego frame) gets one loss over a lateral offset
and an along-path offset on its first 20 steps:

```
L(d, s) = (1/n) sum_t [ sum_corners hinge2(drivable-area clearance) + sum_objects hinge2(box gap) ] + comfort(d, s)
```

minimized by 32 steps of comfort-Hessian-preconditioned descent (CHOMP) under a decaying trust region. A plan
clear of every margin has zero loss and comes back unchanged, so on clean replans the planner is Diffusion
Planner. There is no rule on top: one objective decides whether and where the plan moves.

| File | What it does |
|---|---|
| `optimizer.py` | `RepairConfig` (o_am's constants), comfort Hessian, the edit (lateral offset + path retime), hinge^2 barriers, exact box gap, `PosthocRepair`; the nuPlan terms (TTC, speed limit, scoring the tracked rollout), off by default |
| `tracker.py` | nuPlan's closed-loop controller (`LQRTracker` + `KinematicBicycleModel`) in torch: what the simulator executes when it tracks a plan, differentiable in the plan |
| `scene.py` | online scene in the ego frame: drivable-area boundary (`anchor_wm.build_cache.drivable_area_arrays`, cached and rebuilt every 25 m), object boxes over the window (constant velocity, or a model's per-agent modes weighted by probability) with their velocities, the speed limit where the ego is |
| `anchor_forecast.py` | AnchorWM online: `PlannerInput` -> the cache's per-scenario dict (the cache's own functions) -> `build_sample` -> model -> NMS modes |
| `planner.py` | `RepairedDiffusionPlanner`: Diffusion Planner + repair; seeded sampling; per-replan JSONL stats |
| `check.py` | behavioural gates on a synthetic road (`PYTHONPATH=. python -m posthoc_repair.check`) |
| `check_tracker.py` | `tracker.py` against nuPlan's own controller on random plans (needs nuPlan, no data) |
| `check_ttc.py` | gates for the TTC / speed-limit / rear-agent terms; `--reference <old optimizer.py>`: the defaults repair bit-identically to it |
| `check_anchor_inputs.py` | online vs cached AnchorWM inputs on Val14 scenarios at t = 0, + a random-init forward through the repair |
| `anchor_model.py` | AnchorWM at planning time, nuPlan-free: one forward per replan, then its forecast and its scorer (`diffusion_es` uses the scorer) |

The closed-loop runner (chunk filters, `run_chunks.sh`, `readout.py`) is in the top-level `sim/`, shared by every
arm with or without the repair; Diffusion-ES + the AnchorWM scorer (+ this repair) is `diffusion_es/`.

Planner config: `diffusion_planner/config/planner/diffusion_planner_repair.yaml` (`planner=diffusion_planner_repair`;
`neighbor_forecast: cv | dp | anchor | none`, `anchor_ckpt` for `anchor`, `repair: {...}` overrides `RepairConfig`,
`enable_repair: false` for the same planner without the repair, `seed` for the per-replan sampling seed). The
config's `repair: {}` is empty, so a key is set from the command line with `++`, e.g.
`++planner.diffusion_planner_repair.repair.track=true` (`repair={track:true}` is refused by Hydra).

Diffusion Planner samples its noise from an unseeded `torch.randn`, so two runs of it differ. The repair planner
seeds every replan from the scenario's start time and the iteration: two `enable_repair: false` runs are
bit-identical on every metric of every scenario (checked on 3 Val14 scenarios), so `base` (repair off) vs
`repair_*` is an exact paired test in which every difference is caused by the repair.

## What differs from the WOMD repair

| | WOMD (`waymax_rs`) | nuPlan (here) |
|---|---|---|
| Edge | Waymax road-edge points, clearance from `offroad`'s zero | DRIVABLE_AREA boundary (the AnchorWM cache's extraction), clearance from where `drivable_area_compliance` fires (0.3 m outside) |
| Ego | box centre | rear axle; Pacifica box 1.461 m ahead |
| Other objects | the ES planner's forecaster (CV / MF / AnchorWM modes) | CV from the current tracks, static objects included; or Diffusion Planner's joint neighbour prediction (`dp`); or AnchorWM's 6 modes per agent, probability-weighted (`anchor`) |
| Time | 0.1 s world steps | 0.1 s (simulator and Diffusion Planner) |
| Executed | one pose per step | the plan goes through nuPlan's LQR tracker + kinematic bicycle |

Margins (edge 0.3 m, agent 0.75 m) are o_am's WOMD expert calibration; they have not been re-measured on nuPlan.

## Checks

* Parity with the JAX originals (`lateral_opt.edit_world`, `edit_hessian`, `safety_loss.box_box_signed_gap`,
  `barrier`, and the `lateral_opt` loop on an agents-only loss; random plans incl. stopped / creeping ones):
  edits within 1.5e-5 m, gaps 1e-6 m, Hessian and barrier exact, optimizer output within 0.2 mm after 32
  iterations (float32 accumulation).
* `check.py`: clean plan returned bit-identical; off-road drift pulled back inside; a parked car in the lane
  avoided; smooth; a stopped plan stays finite; the tail holds the window's offset.
* `check_anchor_inputs.py`: the online AnchorWM inputs equal the cached ones exactly (every `build_sample` model
  input, max |diff| 0) on 17 of 18 Val14 scenarios at t = 0 (chunks 000 and 007), with the history buffer built as
  `Simulation.initialize` builds it. The 18th is the cache's 192-object cap: the cache spends slots on objects that
  appear later, so it has 40 agents present at t = 0 where online has 43. Online scene ~75 ms per replan.
* Closed loop with `neighbor_forecast: anchor` and a random-init AnchorWM checkpoint (plumbing only; the
  forecasts are noise): 3/3 Val14 scenarios simulated, 448 replans, no non-finite repair.
* Box pruning (`PosthocRepair.prune_agents`): every edit coordinate moves at most the trust region's reach
  (2.9 m), so boxes out of reach of every scored step are dropped. The loss is bit-identical with and without them
  at random reachable edits; 300 boxes (an AnchorWM scene: 6 modes x 50 agents) -> ~60, 1.1 s -> 0.3 s per
  charged replan on one CPU thread.

## Running (babel)

```bash
# chunk filters for Val14 (token -> log lookup over the val logs, ~4 min)
python -m sim.split_filter --filter diffusion_planner/config/scenario_filter/val14.yaml \
    --data_root /data/datasets/nuPlan-v1.1/nuplan-v1.1/splits/trainval --out /data/user_data/$USER/nuplan_wm/filters/val14
# arms (sbatch from the repo root, or bash inside an allocation with WORKERS=<cpus>)
sbatch sim/run_chunks.sh ARM=base
sbatch sim/run_chunks.sh ARM=repair_cv
python -m sim.readout --run_root /data/user_data/$USER/nuplan_wm/runs/val14/closed_loop_nonreactive_agents --base base
# AnchorWM forecast: REPAIR="planner.diffusion_planner_repair.anchor_ckpt=<run>/final.pt"  ARM=repair_anchor
```

The Diffusion Planner checkpoint is the released one (HuggingFace `ZhengYinan2001/Diffusion-Planner`, revision
`ae5baf1`), in `/data/user_data/yixiz/nuplan_wm/checkpoints/diffusion_planner_release`.

## First results (Val14 chunk_000: 70 scenarios, non-reactive, paired seeded)

| arm | CLS-NR | collisions | drivable | direction | TTC | comfort | vs base |
|---|---|---|---|---|---|---|---|
| base (repair off) | 89.22 | 99.29 | 95.71 | 99.29 | 92.86 | 95.71 | |
| repair_cv | 89.95 | 99.29 | 97.14 | 98.57 | 90.00 | 92.86 | +0.73 [-4.30, +6.30], 11 / 10 |

Diffusion Planner's published Val14 CLS-NR is 89.87. The repair charged 10.9% of replans (6.6 ms p50). Big flips:
fixed 2 drivable-area failures (predicted -0.59 / -0.27 m -> +0.24 m) and 1 collision (-0.14 -> +0.52 m); broke 4:
an unclearable pickup/dropoff scene (large edits, ended off the drivable area), an edge-for-agent trade that still
collided, margin-only nudges that cost TTC and comfort, and a swerve that cleared an overlap through the oncoming
lane (`driving_direction_compliance`; nuPlan's drivable area includes oncoming lanes, so the edge term allows it).

## nuPlan terms (2026-11-03): the tracked rollout, TTC, speed limit

Where o_am still loses points on nuPlan (FlowDrive + `repair_cv`, Val14, read from the simulation logs with nuPlan's own
metric code; `tune/` on autonlab scratch):

* **TTC is the largest loss** (1.6 points). 60 of 87 TTC failures are against stopped objects (parked cars, cones,
  generic objects) ahead of the ego's heading, at a projected hit of exactly 0.9 s (the metric fails one within
  0.95 s): the ego closes at 4-5 m/s on something 4 m away. Every barrier scores positions, so such a plan is clear of
  every margin.
* **The executed ego is not the plan.** In half of the at-fault collisions the repaired plan cleared the object by
  0.1-0.7 m at every replan up to the impact. nuPlan's LQR tracker restarts its reference at the ego every step and
  follows only the plan's smoothed curvature and its speed 1 s ahead: on one fixed plan it covers ~20% of a 0.5-0.9 m
  lateral offset in 1 s, and at 3 m/s a 1 m offset of the plan moves the ego 0.07-0.1 m in 1 s and 0.45 m in 2 s
  (12 m/s: 0.24 / 1.07 m); a speed change is followed within ~1 s. The fixed-plan rollout is within ~0.1 m of the
  closed loop over 1 s, so this is the tracker, not replanning. In pickup/drop-off scenes the repair keeps pushing
  the plan sideways round a cone that the ego then drives into.

`RepairConfig` keys added (each off by default, which is o_am bit for bit: `check_ttc.py --reference` on 40 random
scenes):

| key | |
|---|---|
| `track` | the barriers score `tracker.track(plan)` instead of the plan's poses: nuPlan's `LQRTracker` + `KinematicBicycleModel` (`lqr_tracker.yaml`) in torch, tracking the edited plan from the ego's speed / steering angle / acceleration for the window; within 1e-6 m / 3e-8 rad / 1e-6 m/s of nuPlan's on 30 random plans x 20 steps (`check_tracker.py`), differentiable, float64 inside |
| `ttc_weight` | nuPlan's `time_to_collision_within_bound` as a barrier: at each scored step the ego box (scored speed, along its heading) and every agent ahead of it (`ttc_ahead_deg` 30, the metric's `is_agent_ahead`; forecast velocity) projected to `ttc_samples` look-ahead times in `(0, ttc_horizon_s]` (1 s); the smallest projected gap under `ttc_margin_m` (0.3 m) is charged |
| `agent_behind_deg` | a moving agent behind this bearing when it first comes within the margin is not charged (nuPlan: a rear collision with a moving agent is not the ego's fault); 180 = off |
| `speed_weight` | barrier on the scored speed over the map's speed limit where the ego is (`scene.speed_limit`: the metric's lane / lane-connector lookup at the ego's centre) |

Speeds are the tracked ones with `track`, the plan's central differences otherwise. `posthoc_repair/planner.py`
passes the ego's dynamic state (and the speed limit when `speed_weight > 0`); per-replan stats gain `c_ttc0` /
`c_ttc`. Cost: a charged replan takes ~0.45 s plan-space, ~0.9 s with the TTC term, ~3 s with `track` (the 20-step
rollout runs per iteration), on one CPU thread; a chunk of 70 scenarios takes ~30 min instead of ~25.

Offline, on the logged replans of the 2.5 s before each at-fault collision of `fd_repair_cv` (438 replans, 22
scenarios), tracked by nuPlan's own controller: the logged plans would collide in 59% of replans, re-repaired in plan
space 57%, with `track` 32% (median gap -0.07 -> +0.25 m). Twice the trust-region reach or 64 iterations change
nothing (what is left is too late to avoid), `w_lon_prox: 0` (free retiming) 29%.

### Tuning (FlowDrive, `neighbor_forecast: cv`, Val14's even chunks, then the odd ones held out)

Every arm is paired with a base on the same device (CPU and CUDA draw different noise for one seed: `fd` on CPU vs
on GPU is -0.94 [-2.03, +0.07] on 559 scenarios, 95 better / 222 worse, a seed effect of ~1 point). Even chunks, 559
scenarios unless noted; gain over `fd_repair_cv` (o_am) on the same device, 95% bootstrap CI over scenarios:

| arm (`repair:` keys) | device | gain vs o_am | notes |
|---|---|---|---|
| `ttc1` (`ttc_weight: 1`) | 2080 Ti | +0.68 [+0.23, +1.23] | TTC 92.49 -> 93.92, collisions 98.12 -> 98.75 |
| `trk` (`track: true`) | 2080 Ti | +1.94 [+0.91, +3.07] | collisions 98.12 -> 99.28, drivable 98.57 -> 99.28, TTC 92.49 -> 93.92 |
| `trk_ttc1` (both) | CPU | +3.17 [+1.89, +4.63] (420) | collisions 97.62 -> 99.29, drivable 97.86 -> 99.76, TTC 92.38 -> 95.24; +4.45 [+2.93, +6.10] vs `fd` |

On top of `trk_ttc1` (CPU, vs `trk_ttc1`): `agent_behind_deg: 150` -0.01 (15 scenarios move at all);
`speed_weight: 1` -0.16 [-0.78, +0.39] (speed limit 97.16 -> 99.13 but progress 95.22 -> 94.35, collisions
99.46 -> 98.93); `ttc_ahead_deg: 150` (lateral agents too) -0.80 [-1.53, -0.21]; `edge_weight: 3` -0.05. None kept:
the configuration is `track: true, ttc_weight: 1.0`, the rest at o_am's values.

Val14, FlowDrive (2080 Ti, paired with `fd`, 1118 scenarios): `fd` 91.21, `fd_repair_cv` (o_am) 92.23,
`fd_repair_cv_trk_ttc1` **94.31**: +2.07 [+1.37, +2.81] over o_am (collisions 98.08 -> 98.88, drivable 98.57 -> 99.55,
TTC 92.22 -> 95.44, progress 95.74 -> 95.62). Held-out odd chunks (559): 92.28 -> 93.90, +1.63 [+0.68, +2.64].

Val14, Diffusion Planner (the released checkpoint, seeded; 2080 Ti, paired with `base`, 1118 scenarios), the same
configuration untuned for it: `base` 89.76 (babel's seeded run: 89.76), `repair_cv` (o_am) 90.85, `repair_cv_trk_ttc1`
**94.01**: +3.16 [+2.29, +4.08] over o_am, +4.26 [+3.16, +5.38] over `base` (collisions 97.67 -> 98.93, drivable
97.94 -> 99.37, TTC 92.04 -> 96.24, progress 94.40 -> 94.40). Odd chunks: 90.36 -> 93.69, +3.34 [+2.03, +4.76].
`track` alone on the even chunks: +2.69 [+1.60, +3.87] over o_am. Run as FlowDrive's, with
`++planner.diffusion_planner_repair.repair.*`, in `/zfsauton/scratch/$USER/nuplan_wm/runs/tune_dp` (`base` and
`repair_cv` in `runs/val14`).

## Not done yet

* AnchorWM modes in closed loop: the adapter is in and checked, but there is no trained nuPlan checkpoint
  readable here yet (`/data/user_data/ishanj/anchor_wm_nuplan` is `jeff4_group`).
* The AnchorWM scorer (ES-style candidate ranking): not used; the repair only takes the WM's forecasts.
* nuPlan-calibrated margins (the Val14 expert's own clearance percentiles).
