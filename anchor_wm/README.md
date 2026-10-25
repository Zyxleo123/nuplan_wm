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
| `slurm/` | babel scripts: `env.sh` (paths), `cache.sh` (cache: plan + sharded build), `train.sh` (anchors if missing + one run) |

## Running (babel)

Submit from the repo root. `slurm/env.sh` defaults to the conda env `nuplan_wm`, output under
`/data/user_data/$USER/anchor_wm_nuplan` and logs in `logs/`; override any of them from the environment
(e.g. `CONDA_ENV=nuplan_wm_ishan`).

```bash
# 1. cache. Smoke-sized (lists and builds in one job):
STEP=direct CACHE_DIR=/data/user_data/$USER/anchor_wm_nuplan/cache_smoke \
  EXTRA_ARGS="--max_db_files 12 --total_scenarios 300" sbatch anchor_wm/slurm/cache.sh
# Full: plan once, then build in shards (each opens only its own logs and skips files that exist). The plan
# reuses Diffusion-Planner's 1M training list (diffusion_planner_training.json, written by data_process.py),
# finding each token's log; without --plan_from_list it re-lists every log with the same filter / seed.
plan=$(STEP=plan EXTRA_ARGS="--plan_from_list diffusion_planner_training.json" sbatch --parsable --cpus-per-task=32 anchor_wm/slurm/cache.sh)
sbatch --array=0-3 --dependency=afterok:${plan} anchor_wm/slurm/cache.sh

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
| `lane_*` within 120 m (up to 256, + LANE / LANE_CONNECTOR type, ids, topology: successors / predecessors / left / right neighbours), `poly_*` (crosswalks, stop lines, intersections) | map tokens; the lane graph gives reference lines (PLUTO / PlanTF) and nuPlan's exact connected-lane / driving-direction tests |
| `lane_tl_future` | traffic-light state per lane at t = 0, 0.5, ..., 8 s, for red-light labels (NAVSIM EPDMS) |
| `route_xy`, `route_roadblock_ids` | ordered route centreline (lane path through the corrected route roadblocks) and the route itself, for progress labels and route features |
| `da_*` | drivable-area boundary (PDMS / nuplan `drivable_area_compliance`), for offroad labels and scorer tokens |
| `goal_pose / goal_valid` | mission goal (only used with `--use_goal`) |
| `log_name`, `scenario_type` | log-level train / val split; subsets |

The time grid is 0.1 s (21 past frames incl. t = 0, 80 future frames), which the model uses as is
(`history_stride` / `future_stride` in `config.py` subsample it).
Files are named `<log_name>__<token>.npz` so the split needs no file reads. A scenario takes about
500 KB (330 KB on babel's compressed /data/user_data), so the 1M-scenario training set is about 330 GB there;
object arrays dominate, so objects are capped at the 192 the labels use and their velocities are float16.

Building it reads each log once: every scenario's 101 frames come from one pass over the log's lidar
frames / ego poses / boxes (`LogData`, the same rows and frame sampling as the devkit's per-frame queries,
which take ~105 queries per scenario), after a sequential copy of the `.db` from NFS to node-local disk.
On the smoke set this gives byte-identical files at 0.084 s / scenario on 8 CPUs (0.33 s with the devkit's
queries); a cold 100 MB log with 76 scenarios takes 4.7 s to copy, 0.9 s to read and 0.16 s per scenario
for the map / route / drivable-area work. Listing all 13,180 training logs (54.8M scenarios) takes ~19 min
on 16 CPUs, which is why the plan step does it once.

## Design differences: nuPlan vs WOMD

| | WOMD (`waymax_rs/anchor_wm`) | nuPlan (this package) | Why |
|---|---|---|---|
| Samples | sim-state shards; 3 anchor times per scenario (2 / 3 / 4 s into a 9 s episode) | one npz per scenario token (devkit single-sample scenarios, iteration 0) | nuPlan scenarios are already per-timestamp samples |
| Ego frame | SDC box centre | **rear axle** (nuPlan's planning convention); box geometry uses the centre 1.461 m ahead (`geometry.ego_center`), Pacifica 5.176 x 2.297 m | candidates / predictions stay in the frame a nuPlan planner outputs |
| History / horizon | 10 steps x 0.2 s, **25 steps x 0.2 s (5 s)** | 21 frames x 0.1 s (2 s), **80 steps x 0.1 s (8 s)** | the usual nuPlan setup (PlanTF, PLUTO, Diffusion Planner, `mini_diffusion_policy`); a planner's 0.1 s output needs no resampling for the 10 Hz simulation |
| Agents | up to 128 WOMD objects incl. the SDC | ego (slot 0) + 63 nearest vehicles / pedestrians / bicycles at t = 0 | nuPlan scenes list all detections; the nearest 63 cover the scorer's 32 |
| Static objects | not in WOMD tracks | cones / barriers / signs / generic objects: **encoder tokens** (32 nearest), exact box gaps in the scorer's rule features, and overlap-label obstacles | frequent in nuPlan (construction zones), collisions with them matter |
| Collision labels | any overlap with a logged WOMD box | **nuPlan's at-fault rule** (`no_ego_at_fault_collisions`): each object's first contact is classified as nuplan's `_get_collision_type` does (ego stopped <= 0.05 m/s: not at fault; object stopped, incl. every static object: at fault; object behind, > 150 deg from the ego heading: not at fault; front-bumper contact: at fault; else lateral: at fault when the ego's corners are not in one / connected lane, tested on the cached lane polygons without the lane graph). Log-replay boxes of every cached object (192 nearest, incl. static and late-appearing ones) at the 0.1 s plan steps; midpoints (for coarser steps) interpolate box headings modulo pi | CLS is the benchmark. Log replay is non-reactive, so any-overlap counted being rear-ended: stops had 41-44% contact and now 0% at fault. nuPlan tracks flip long-vehicle headings by pi between frames (a plain angle midpoint rotated a truck crosswise) |
| Map | WOMD roadgraph polylines (21 types) + traffic-light points | lanes / lane connectors (centre + left / right boundary offsets, on-route flag, traffic-light state, speed limit) and crosswalk / stop-line polygons; no separate light tokens | nuPlan map API; lights are per lane connector |
| Goal | last logged SDC position of the episode, fed to the SDC query and the scorer | **route**: the SDC query and the scorer's plan token get 10 route-centreline points 10 m apart + ego velocity; mission goal optional (`--use_goal`) | the mission goal is often hundreds of metres to kilometres away, or missing; the route says which lanes to take |
| Offroad | Waymax road-edge points, sign from the prior-point rule, > 0 m | **nuplan `drivable_area_compliance`**: a box corner more than **0.3 m** outside the union of the DRIVABLE_AREA layer (road segments, intersections, generic drivable areas, car parks); the ego-lane exemption is not modelled. Boundary = ring vertices + points <= 1 m apart, oriented with the drivable side on the left, each with its segments; distance to those segments, sign from the prior-point rule. Map cracks / slivers < 20 cm are removed (closing + opening), repeated vertices dropped | CLS. Matches an exact point-in-polygon test on all 6300 smoke candidates. Without the smoothing, hairline gaps between map polygons flipped the sign 3-4 m away; nearest-point distances overestimated it by up to ~0.4 m near the threshold |
| Progress label | `goal_gain` = goal distance reduction; `goal_dist` precision term | `route_progress` along the route centreline from the ego's projection (logged path when there is no route), normalised by the group's best (PDM ego progress) | no near goal in nuPlan |
| Simple loss utility | unsafe tiers + goal gain + `log(1 + goal_dist)` | unsafe tiers + route progress / group best (no goal-distance term) | as above |
| Aux heads | time to violation, goal distance | time to violation, normalised route progress | as above |
| Scorer route features | per-step distance / heading error to the nearest lane point; goal geometry | per-step distance / heading error to the route centreline; endpoint route progress / distance / heading | route replaces the goal |
| Time remaining | episode can end inside the plan (`--time_remaining`) | dropped: nuPlan scenarios always have the 8 s future | |
| Perturbations | lateral ramps over time; heading = path tangent while still | lateral ramps also faded in over the first 5 m of travel; heading = logged yaw at the same arc length while still | 36% of nuPlan samples have a stopped ego, whose logged path is position jitter: the tangent was random and stopped egos slid sideways, labelling copies of the safe logged future as overlapping |
| Ego input | the SDC is decoded from its 1 s history like any agent; the goal token carries its velocity | `--ego_input current_dropout` (default): encoder / decoder see only the t = 0 ego state, and in training its velocity (also in the plan embedding) is zeroed for 75% of samples; `current` / `history` for ablations. The scorer always gets the true velocity | PlanTF: ego history drops NR-CLS 86 -> 57 (copying its own past); state dropout 0.75 fixes the kinematic shortcut. Diffusion Planner drops ego state entirely; ChauffeurNet used past-motion dropout. The scorer is trained on rule labels, not imitation, and needs the velocity to judge feasibility / comfort |
| Encoder init | warm start from MF-Gaussian (`--init_ckpt`) | from scratch (encoder lr = decoder lr = 3e-4, joint scorer gradient ramped in after 10k steps over 10k) | no nuPlan MF-Gaussian checkpoint |
| Recipe | v1 by default, v2 via `RECIPE=v2` | v2 only (no v1 switches): route / rule features, aux heads, 32 agents x 6 modes (8 nearest x 12), kernel interaction weight, 0.1 m sigma floor, no decay on norms / biases | v1 is no longer needed |
| Split | last 10% of shards | 5% of **logs** (stable hash of the log name) | scenarios of one log overlap in time |

## Smoke tests (babel, 300-scenario cache from 12 logs)

- Labels on the logged futures: 0% at-fault collision (0.7% any contact), 0% offroad (before the fixes in
  the table: 1.3% overlap / 5% offroad). Route progress of the logged future matches its path progress to
  0.2 m on average.
- Offroad vs an exact CLS-style test (distance of each corner to the raw DRIVABLE_AREA polygons >= 0.3 m) on
  the logged future, its 16 perturbations and +-4 / 6 m lateral offsets: 6300 / 6300 candidates agree.
- At-fault vs any contact per perturbation: stops 0% vs 41-44%, x0.5 speed 0.7% vs 32%, lateral shifts
  mostly at fault (e.g. +-3 m: 25-26% vs 30-32%); stopped egos rarely touch anything after the heading /
  lateral-fade fix (9% contact, 70% before).
- Training, 200 steps at batch 16 on an L40S (`SMOKE=1`, 0.1 s / 80 steps), every arm / loss: no non-finite
  steps; val minADE@6 4.1 -> 2.7-3.0 m; joint scorer pairwise accuracy 0.81-0.85 and picked-unsafe
  0.07-0.08 (current / simple loss; 48% of candidates unsafe). 0.46 s/step and ~14 GB peak with the scorer,
  0.03 s/step forecasting only. Resume, `--file_list` and `--use_goal` also run.

## Known gaps

- Of nuPlan's CLS terms only at-fault collisions and drivable-area compliance are labels; no TTC,
  driving-direction compliance, making-progress or speed-limit terms yet. Collisions with static objects
  count as unsafe (CLS halves the score for the first one).
- The drivable-area ego-lane exemption is not modelled, and the model does not yet read the lane graph or
  the future traffic lights (they are cached for labels and later inputs).
- Ego-input defaults follow PlanTF's closed-loop ablation; we can only check them once AnchorWM runs as a
  nuPlan planner (open-loop ego metrics favour `--ego_input history`).
- Log replay is non-reactive (CLS-NR); agents never react to the candidate.
- Static objects are taken at their t = 0 pose in the scorer's rule features (labels use the full log).
- Pedestrians and bicycles use Cartesian anchors like vehicles.
