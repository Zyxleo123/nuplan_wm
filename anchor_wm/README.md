# `anchor_wm`: AnchorWM on nuPlan v1.1 (training)

A port of `waymax_rs/anchor_wm` (WOMD) to nuPlan: an anchor-query world model with a jointly trained
ego-plan scorer. It covers training only: cache, anchors, model, losses and the training loop, with
both scorer losses (`--scorer_loss current|simple`) and the joint / separate / forecasting-only arms.
The WOMD package's ES-bank fine-tuning, bank / closed-loop evaluation and visualisation scripts are
not ported yet (there is no nuPlan planner or ES bank yet).

```
history, statics, map, route ──► scene encoder (MF-Gaussian design, from scratch)
                                  │ agent tokens                  │ scene tokens
k-means anchors ──► embed ─┐      ▼                               │
route summary, ego vel ────┴► query (agent, anchor) ──► MLP ──────┤► per-step residual around the anchor,
                                                                  │  sigma, yaw, logit
ego candidates [C,T] ──► scorer: GRU + cross-attention to scene tokens + time-aligned geometric attention
                         to (agent, mode) tokens + drivable-area boundary tokens + route / rule features
                         ──► total score and sub-scores (overlap, offroad, progress, comfort)
```

## Files

| File | What it does |
|---|---|
| `config.py` | `AnchorWMConfig`, type tables, ego (Pacifica) size, cache time grid |
| `build_cache.py` | nuPlan DBs + maps -> one npz per scenario (nuplan-devkit) |
| `data.py` | cache npz -> fixed-size model inputs, forecasting targets, label geometry; log-level split |
| `anchors.py` | k-means anchors per group (vehicle incl. ego / pedestrian / bicycle) and their oracle ADE/FDE |
| `geometry.py` | SAT box gaps, signed distance to the drivable-area boundary, path utilities |
| `candidates.py` | self-generated candidates (decoded ego anchors, logged future, 16 perturbations) and online log-replay labels |
| `model.py` | encoder, anchor decoder, `PlanScorer` |
| `losses.py` | `wm_loss`, `score_loss` (current), `simple_score_loss` (simple) |
| `train.py` | training loop (bf16 autocast, joint-gradient ramp, non-finite step skipping, profiling, wandb) |
| `slurm/` | babel scripts: `env.sh` (paths), `cache.sh` (cache array job), `train.sh` (anchors if missing + one run) |

## Running (babel)

```bash
# 1. cache: smoke-sized, or the full trainval split as a 20-shard array job
CACHE_DIR=/data/user_data/mineuih/ishan/anchor_wm_nuplan/cache_smoke \
  EXTRA_ARGS="--max_db_files 12 --total_scenarios 300" sbatch anchor_wm/slurm/cache.sh
sbatch --array=0-19 anchor_wm/slurm/cache.sh

# 2. anchors (train.sh fits them when ANCHORS is missing) + training
sbatch anchor_wm/slurm/train.sh                          # joint, current loss
SCORER_LOSS=simple sbatch anchor_wm/slurm/train.sh       # joint, simple loss
ARM=separate sbatch anchor_wm/slurm/train.sh             # scorer on detached WM outputs
SMOKE=1 CACHE_DIR=... ANCHORS=... bash anchor_wm/slurm/train.sh   # 200 steps inside an allocation
```

Train on a subset of the cache with `--file_list subset.json` (a JSON list of cache file names).

## Cache

`build_cache.py` writes everything the WOMD pipeline needs and the Diffusion-Planner cache
(`data_process.py`) lacks:

| Field | Why |
|---|---|
| `ego_pose / ego_vel / ego_acc` [101] | ego history (the ego is decoded like any agent) and logged future |
| `obj_*`: every tracked object within 120 m in any frame, with per-frame validity | agent history / targets with real masks (the DP cache pads missing frames with the nearest state or zeros); log-replay boxes for overlap labels include static objects and objects that appear later |
| `lane_*` (+ LANE / LANE_CONNECTOR type), `poly_*` (crosswalks, stop lines) | map tokens |
| `route_xy` | ordered route centreline (lane path through the corrected route roadblocks), for progress labels and route features |
| `da_*` | drivable-area boundary (PDMS / nuplan `drivable_area_compliance`), for offroad labels and scorer tokens |
| `goal_pose / goal_valid` | mission goal (only used with `--use_goal`) |
| `log_name`, `scenario_type` | log-level train / val split; subsets |

The time grid is 0.1 s (21 past frames incl. t = 0, 80 future frames), which the model uses as is
(`history_stride` / `future_stride` in `config.py` subsample it).
Files are named `<log_name>__<token>.npz` so the split needs no file reads. The smoke cache (300
scenarios) is about 320 KB per scenario, so the full ~600k-scenario cache is about 190 GB.

## Design differences: nuPlan vs WOMD

| | WOMD (`waymax_rs/anchor_wm`) | nuPlan (this package) | Why |
|---|---|---|---|
| Samples | sim-state shards; 3 anchor times per scenario (2 / 3 / 4 s into a 9 s episode) | one npz per scenario token (devkit single-sample scenarios, iteration 0) | nuPlan scenarios are already per-timestamp samples |
| Ego frame | SDC box centre | **rear axle** (nuPlan's planning convention); box geometry uses the centre 1.461 m ahead (`geometry.ego_center`), Pacifica 5.176 x 2.297 m | candidates / predictions stay in the frame a nuPlan planner outputs |
| History / horizon | 10 steps x 0.2 s, **25 steps x 0.2 s (5 s)** | 21 frames x 0.1 s (2 s), **80 steps x 0.1 s (8 s)** | the usual nuPlan setup (PlanTF, PLUTO, Diffusion Planner, `mini_diffusion_policy`); a planner's 0.1 s output needs no resampling for the 10 Hz simulation |
| Agents | up to 128 WOMD objects incl. the SDC | ego (slot 0) + 63 nearest vehicles / pedestrians / bicycles at t = 0 | nuPlan scenes list all detections; the nearest 63 cover the scorer's 32 |
| Static objects | not in WOMD tracks | cones / barriers / signs / generic objects: **encoder tokens** (32 nearest), exact box gaps in the scorer's rule features, and overlap-label obstacles | frequent in nuPlan (construction zones), collisions with them matter |
| Overlap labels | logged boxes of every WOMD object | log-replay boxes of every cached object (192 nearest, incl. static and late-appearing ones); checked at the 0.1 s plan steps (WOMD: 0.2 s steps plus midpoints; midpoints, still used for coarser steps, interpolate box headings modulo pi) | at 0.1 s midpoints change no label but double the cost; nuPlan tracks flip long-vehicle headings by pi between frames (a plain angle midpoint rotated a truck crosswise) |
| Map | WOMD roadgraph polylines (21 types) + traffic-light points | lanes / lane connectors (centre + left / right boundary offsets, on-route flag, traffic-light state, speed limit) and crosswalk / stop-line polygons; no separate light tokens | nuPlan map API; lights are per lane connector |
| Goal | last logged SDC position of the episode, fed to the SDC query and the scorer | **route**: the SDC query and the scorer's plan token get 10 route-centreline points 10 m apart + ego velocity; mission goal optional (`--use_goal`) | the mission goal is often hundreds of metres to kilometres away, or missing; the route says which lanes to take |
| Offroad | Waymax road-edge points, sign from the prior-point rule | boundary of the union of the **DRIVABLE_AREA** layer (road segments, intersections, generic drivable areas, car parks), oriented with the drivable side on the left, 1 m spacing; same signed-distance rule. Map cracks / slivers < 20 cm are removed (closing + opening) | PDMS / nuplan `drivable_area_compliance`: every ego corner inside the drivable area. Without the smoothing, hairline gaps between map polygons flipped the sign 3-4 m away |
| Progress label | `goal_gain` = goal distance reduction; `goal_dist` precision term | `route_progress` along the route centreline from the ego's projection (logged path when there is no route), normalised by the group's best (PDM ego progress) | no near goal in nuPlan |
| Simple loss utility | unsafe tiers + goal gain + `log(1 + goal_dist)` | unsafe tiers + route progress / group best (no goal-distance term) | as above |
| Aux heads | time to violation, goal distance | time to violation, normalised route progress | as above |
| Scorer route features | per-step distance / heading error to the nearest lane point; goal geometry | per-step distance / heading error to the route centreline; endpoint route progress / distance / heading | route replaces the goal |
| Time remaining | episode can end inside the plan (`--time_remaining`) | dropped: nuPlan scenarios always have the 8 s future | |
| Perturbations | lateral ramps over time; heading = path tangent while still | lateral ramps also faded in over the first 5 m of travel; heading = logged yaw at the same arc length while still | 36% of nuPlan samples have a stopped ego, whose logged path is position jitter: the tangent was random and stopped egos slid sideways, labelling copies of the safe logged future as overlapping |
| Encoder init | warm start from MF-Gaussian (`--init_ckpt`) | from scratch (encoder lr = decoder lr = 3e-4, joint scorer gradient ramped in after 10k steps over 10k) | no nuPlan MF-Gaussian checkpoint |
| Defaults | v1 flags; v2 via `RECIPE=v2` | the v2 recipe is the default: route / rule features, aux heads, 32 agents x 6 modes (8 nearest x 12), kernel interaction weight, 0.1 m sigma floor, no decay on norms / biases | single configuration to start from |
| Split | last 10% of shards | 5% of **logs** (stable hash of the log name) | scenarios of one log overlap in time |

## Smoke tests (babel, 300-scenario cache from 12 logs)

- Labels on the logged futures: 0% overlap, 0% offroad (before the fixes in the table: 1.3% / 5%).
  Route progress of the logged future matches its path progress to 0.2 m on average.
- Perturbations of the logged future: +-0.75 m lateral 2-6% overlap, x1.2 speed 3%, stops ~40% (rear-ended
  in log replay after ~4 s); stopped egos 9% (70% before the heading / lateral-fade fix).
- Training, 200 steps at batch 16 on an L40S (`SMOKE=1`, 0.1 s / 80 steps), every arm / loss: no non-finite
  steps; val minADE@6 4.1 -> 2.7-3.1 m; joint scorer pairwise accuracy 0.86-0.87 and picked-unsafe
  0.14-0.16 (simple / current loss). 0.44 s/step and 13.7 GB peak with the scorer, 0.03 s/step forecasting
  only. Resume, `--file_list`, `--use_goal` and the `--no_route_features --no_rule_features --no_aux_heads`
  ablation also run.

## Known gaps

- Overlap labels use any overlap, not nuPlan's at-fault collision logic; no TTC, driving-direction
  or lane-keeping terms from PDMS yet.
- Static objects are taken at their t = 0 pose in the scorer's rule features (labels use the full log).
- Pedestrians and bicycles use Cartesian anchors like vehicles.
