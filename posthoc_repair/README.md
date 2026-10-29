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
| `optimizer.py` | `RepairConfig` (o_am's constants), comfort Hessian, the edit (lateral offset + path retime), hinge^2 barriers, exact box gap, `PosthocRepair` |
| `scene.py` | online scene in the ego frame: drivable-area boundary (`anchor_wm.build_cache.drivable_area_arrays`, cached and rebuilt every 25 m), object boxes over the window (constant velocity, or a model's per-agent modes weighted by probability) |
| `anchor_forecast.py` | AnchorWM online: `PlannerInput` -> the cache's per-scenario dict (the cache's own functions) -> `build_sample` -> model -> NMS modes |
| `planner.py` | `RepairedDiffusionPlanner`: Diffusion Planner + repair; seeded sampling; per-replan JSONL stats |
| `check.py` | behavioural gates on a synthetic road (`PYTHONPATH=. python -m posthoc_repair.check`) |
| `check_anchor_inputs.py` | online vs cached AnchorWM inputs on Val14 scenarios at t = 0, + a random-init forward through the repair |
| `sim/split_filter.py` | a token-list scenario filter (Val14) -> chunk filters with their own `log_names` |
| `sim/run_chunks.sh` | one arm over the chunks, resumable (done-markers), sbatch or inside an allocation |
| `sim/readout.py` | pooled per-arm CLS table, paired vs a base arm, repair stats |

Planner config: `diffusion_planner/config/planner/diffusion_planner_repair.yaml` (`planner=diffusion_planner_repair`;
`neighbor_forecast: cv | dp | anchor | none`, `anchor_ckpt` for `anchor`, `repair: {...}` overrides `RepairConfig`,
`enable_repair: false` for the same planner without the repair, `seed` for the per-replan sampling seed).

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
python -m posthoc_repair.sim.split_filter --filter diffusion_planner/config/scenario_filter/val14.yaml \
    --data_root /data/datasets/nuPlan-v1.1/nuplan-v1.1/splits/trainval --out /data/user_data/$USER/nuplan_wm/filters/val14
# arms (sbatch from the repo root, or bash inside an allocation with WORKERS=<cpus>)
sbatch posthoc_repair/sim/run_chunks.sh ARM=base
sbatch posthoc_repair/sim/run_chunks.sh ARM=repair_cv
python -m posthoc_repair.sim.readout --run_root /data/user_data/$USER/nuplan_wm/runs/val14/closed_loop_nonreactive_agents --base base
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

## Not done yet

* AnchorWM modes in closed loop: the adapter is in and checked, but there is no trained nuPlan checkpoint
  readable here yet (`/data/user_data/ishanj/anchor_wm_nuplan` is `jeff4_group`).
* The AnchorWM scorer (ES-style candidate ranking): not used; the repair only takes the WM's forecasts.
* nuPlan-calibrated margins (the Val14 expert's own clearance percentiles).
