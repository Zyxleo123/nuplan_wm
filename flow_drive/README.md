# `flow_drive`: FlowDrive in place of Diffusion Planner, as nuPlan planners

FlowDrive (Wang et al., [arXiv 2509.21961](https://arxiv.org/abs/2509.21961)) in place of Diffusion Planner: its
released flow model, executed as it comes (`fd`) or through the o_am repair (`fd_repair_*`). One sample per
replan, no guidance, no candidate selection. This is the paper's plain **FlowDrive** (upstream `post_mode: 0`),
not **FlowDrive\*** (`post_mode: 1`: moderated lateral guidance over 30 candidates, smoothing, speed bounding, PDM
scoring + emergency brake). No ES: AnchorWM's scorer takes 8 s plans, FlowDrive plans 4 s.

```
nuPlan history ── DataProcessor.observation_adapter ──► encoder ──► DiT velocity field, 8 Euler steps (1 → 0)
                                                                         │
                     simulator ◄── transform_predictions_to_states ◄─────┘ 40 poses, 0.1 s (4 s), ego frame
```

Reference, Table 1 of the paper (Val14, CLS; the release is the epoch-110 checkpoint all FlowDrive results use):

| | Val14 NR | Val14 R |
|---|---|---|
| Diffusion Planner | 89.87 | 82.38 |
| **FlowDrive** | **91.21** | 85.37 |
| FlowDrive\* | 94.81 | 92.96 |

## What is vendored

From `github.com/einsteinguang/flow_drive_planner` at `06f941e` (Apache-2.0, `LICENSE`), the inference path only:

| File | |
|---|---|
| `model/module/{dit,mixer}.py`, `data_process/*.py`, `config/normalization.json` | verbatim (`check.py` compares the bytes) |
| `model/module/encoder.py` | verbatim but for the `box.ConfigBox` type hint |
| `utils/normalizer.py` | verbatim but `openjson` reads with `json`, not mmengine |
| `config/config.yaml` | upstream's, the keys inference reads |
| `model/flow_drive_planner.py` | upstream `FlowDrivePlanner.forward` with diffusers' `EMAModel.copy_to` and `FlowMatchEulerDiscreteScheduler` (shift 1) written out |
| `planner/planner.py` | upstream `FlowDrivePlannerWrapper`, the `post_mode: 0` branch: no tuplan_garage |

So the `nuplan_wm` env needs nothing it does not have for Diffusion Planner (torch, timm, nuplan): no diffusers,
mlflow, mmengine, python-box or tuplan_garage.

Its data processing is FlowDrive's own, not `diffusion_planner/data_process`: the same map and agent features,
plus `lanes_is_route` (lanes whose ends match a route lane), which FlowDrive's encoder reads instead of
`route_lanes`. It also fills `ego_current_state`'s velocity / acceleration / yaw rate, which Diffusion Planner zeroes,
but the plain model reads only its x, y, cos, sin (the flow's fixed first row; constant in the ego frame), so like
Diffusion Planner it plans without the ego's speed. Each replan seeds python, numpy and torch with `520 + iteration`
and turns cudnn's autotuner off, as upstream does (`seed` in the planner config; `null` = unseeded).

Planner config: `diffusion_planner/config/planner/flow_drive.yaml` (`planner=flow_drive`; `ckpt_path`, `device`,
`seed`, `config_file`). Arm: `ARM=fd` in `sim/run_chunks.sh`, checkpoint `FD_CKPT`.

## FlowDrive + repair (`repair_planner.py`)

`RepairedFlowDrivePlanner` is `posthoc_repair`'s `RepairedDiffusionPlanner` with FlowDrive's plan in place of
Diffusion Planner's: the 40-pose plan goes through `PosthocRepair` (its 20-step window fits in 4 s), and the
repair step and its per-replan stats are `RepairedDiffusionPlanner._repaired` / `_record` themselves, so the two
generators are repaired by the same code (`posthoc_repair/` and `diffusion_es/` are untouched). Config:
`diffusion_planner/config/planner/flow_drive_repair.yaml` (`planner=flow_drive_repair`), the repair keys as
`diffusion_planner_repair`'s, `neighbor_forecast: cv | anchor | none`: FlowDrive predicts no neighbours, so there
is no `dp` forecast.

Its seeding is `fd`'s (`seed + iteration`), not the Diffusion Planner repair planner's per-scenario hash, so the
paired base of every repair arm is `fd` itself:

| ARM | planner |
|---|---|
| `fd` | FlowDrive alone |
| `fd_base` | `flow_drive_repair` with `enable_repair: false`: must equal `fd` on every metric (the plumbing gate) |
| `fd_repair_cv` / `fd_repair_anchor` / `fd_repair_none` | + the repair with that forecast (`anchor` takes `ANCHOR_CKPT`) |

## FlowDrive + best-of-K under AnchorWM's scorer (`best_of_k.py`, `select_planner.py`)

`planner=flow_drive_select`, arms `fd_sel_norepair` / `fd_sel_<f>`: every replan, `population` (32) FlowDrive samples
(one encoder pass; sample 0 drawn as `fd` draws its one) -> each repaired with forecast `<f>` (`fd_sel_<f>`) ->
each extended from 4 s to the scorer's 8 s -> AnchorWM's total score -> the best candidate's own 4 s plan. AnchorWM
runs once per replan on the online scene (as `diffusion_es.planner` builds it). Per-replan stats use the ES planner's
keys, so `sim.readout`'s ES summary reads them.

The scorer's attention is laid out over its forecast's 80 steps, so a 4 s plan must be extended. `extension: speed`
(default) holds the last speed along the last heading: it keeps closing on a slower lead car through the unexecuted
half, so the scorer charges that to the plan (nuPlan weighs collisions and TTC most; TTC is `fd`'s weakest metric).
`decel` brakes to a stop over the extension instead. They pick different candidates (`check_select.py`).

Checks (`check_select.py`, CPU, job 94067): sample 0 is `fd`'s sample (K = 1 bit for bit, K = 32 within 4e-6);
both extensions keep the plan and step as intended; the scorer puts sample 0 at twice its speed (into the lead car)
below the median under both; one step with the repair runs. Not gated: a candidate pushed off the road ranks above
the median, but on this synthetic road in-lane FlowDrive samples also get `offroad` sub-logits up to +11 (job 93603):
the road is out of distribution for the scorer. Closed loop (3 scenarios, jobs 94069-94074): `population: 1` without
the repair reads out as `fd` exactly (+0.00, 0 / 0); at K = 32, 0.84-0.88 s per replan p50 (scene 230 ms, sampling
100 ms, scoring 30 ms), 0.6 GB of GPU per worker, so <= 12 workers on an 11 GB 2080 Ti.

## Results (Val14, closed-loop non-reactive, autonlab, 2026-11-02)

| arm | score | vs `fd` (paired, 95% CI) | at-fault collisions | TTC | drivable area | progress | comfort |
|---|---|---|---|---|---|---|---|
| `fd` | **91.21** (paper: 91.21) | | 96.74 | 91.32 | 98.66 | 95.69 | 95.62 |
| `fd_repair_cv` | 92.23 | +1.02 [+0.24, +1.82] | 98.08 | 92.22 | 98.57 | 95.74 | 95.97 |
| `fd_repair_anchor` | 92.47 | +1.26 [+0.48, +2.06] | 98.08 | 92.58 | 98.66 | 95.74 | 95.71 |
| `fd_sel_anchor` | 92.35 | +1.13 [+0.19, +2.08] | 98.35 | 93.11 | 98.75 | 95.27 | 94.81 |
| `fd_repair_cv_trk_ttc1` (2026-11-03) | **94.31** | +3.09 [+2.13, +4.08] | 98.88 | 95.44 | 99.55 | 95.62 | 95.97 |

1118 scenarios per arm, every chunk `0 failed`. The repair moves 12% (`cv`) / 17% (`anchor`) of replans, at 10 / 275 ms
per replan p50. `fd_sel_anchor` scores with `scorer_cs_factorial_v1_both` (ishanj's fine-tune of joint_current's
scorer on nuPlan metrics; its world model is joint_current's, bit for bit), K = 32, `speed` extension; it picks a
plan other than sample 0 in 97% of replans, at 1.24 s per replan p50. Against `fd_repair_anchor` it is
-0.13 [-0.93, +0.68] (310 better / 349 worse): it trades progress and comfort for fewer collisions and TTC
violations, and stalls 1 scenario (`ego is making progress` 99.91). `readout_vs_fd_repair_anchor.md` next to
`readout.md` has the per-scenario list. `fd_sel_norepair` / `fd_sel_cv` were cancelled.

`fd_repair_cv_trk_ttc1` is `fd_repair_cv` with `repair: {track: true, ttc_weight: 1.0}` (`posthoc_repair/README.md`,
"nuPlan terms": the barriers score nuPlan's tracker rollout of the plan, plus a TTC barrier), run as
`ARM=fd_repair_cv TAG=trk_ttc1 OVERRIDES="++planner.flow_drive_repair.repair.track=true
++planner.flow_drive_repair.repair.ttc_weight=1.0"` on 2080 Tis (as `fd`), without simulation logs, in
`/zfsauton/scratch/$USER/nuplan_wm/runs/tune`. Against `fd_repair_cv` it is +2.07 [+1.37, +2.81] (242 better / 159
worse), against `fd_repair_anchor` +1.83 [+1.13, +2.57]. It was chosen on the even chunks; on the odd ones, held out,
it scores 93.90 against `fd_repair_cv`'s 92.28 (+1.63 [+0.68, +2.64], 559 scenarios). One sample per replan, as `fd`.

## Checks (`check.py`)

```bash
python -m flow_drive.check --upstream <flow_drive_planner checkout>   # needs torch, timm, diffusers; no nuPlan
```

against upstream's code imported side by side (its `flow_drive` swapped into `sys.modules`) and diffusers itself.
All pass (autonlab CPU, torch 2.11, diffusers 0.36.0, job 92294, 2026-11-01):

* the 9 verbatim files are byte-identical to upstream's;
* the Euler sigmas equal `FlowMatchEulerDiscreteScheduler(2000).set_timesteps(8).sigmas` (1, 0.857, ..., 0.0005, 0);
* all 152 encoder / 99 DiT tensors equal upstream's `load_checkpoint_directly` (diffusers `EMAModel`);
* the plan equals upstream's `forward` under the same seed, max |diff| 0, on a batch of 6 synthetic scenes and on
  each alone (seeds 520-525);
* the plan is finite and [B, 40, 3], and another seed changes it.

`check_repair.py` (no upstream, no nuPlan; `diffusion_es.check`'s 3-lane road with a lead car, its lanes as 25 m
map elements) runs the released model on real-shaped inputs and the repair on its plans. All pass (job 92427):

* 31 of 32 seeds plan 15-45 m along the road in 4 s, on it, heading along it; seed 541 ends 6.8 m left, past the
  edge. The samples spread 2.2 m on average (max endpoint deviation 13.5 m): unlike Diffusion Planner, which
  gives the same plan for every seed on this road, plain FlowDrive's samples differ;
* the repair returns the 19 plans clear of every margin bit-identical, and pulls each of them, pushed 6 m left
  through the edge, back inward (edge clearance -3.25..-1.43 m -> -0.32..0.22 m).

On babel's stack (Python 3.9, torch 2.0.0, timm 1.0.10, numpy 1.26; a throwaway env, job 92457) the checkpoint
loads (`weights_only=True`; torch 2.0 only warns that `TypedStorage` is deprecated), `check_repair.py` passes with the
same numbers, and the plans of both scripts' scenes (32 road seeds, 6 random scenes) match torch 2.11's within
6.6e-5 m: the CPU noise is the same, the rest is float rounding.

On the same road with `dp_inputs`'s lanes (one 100 m polyline per lane, 5 m spacing) FlowDrive's samples scatter
(x -20..26 m, y 2..14 m at 4 s over 8 seeds) while Diffusion Planner stays in its lane: FlowDrive is sensitive to
how the map is cut into elements, which nuPlan's map gives it as trained.

Not checked here: the data processing on real scenes (it is upstream's, byte for byte) and closed-loop scores
(nuPlan and its data are on babel only).

## Running (autonlab)

nuPlan is on autonlab too (`/zfsauton/datasets/ArgoRL/brianyan/nuplan/dataset`: `splits/trainval` holds all 1381
val logs, `maps/`), and `sim/autonlab/` runs `sim/run_chunks.sh` there unchanged: `site.sh` (paths; mineuih's
`nuplan_wm` env, py3.9 / torch 2.0 / nuplan-devkit 1.2.2, run with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
because its protobuf 6 cannot load tensorboard 2.11's protos), `split.sbatch` (Val14's 16 chunk filters, once),
`sim.sbatch` (one arm, `legacy` GPU), `readout.sbatch`, and `submit.sh`, which only calls sbatch:

```bash
cp /path/to/flow_drive_model.pth flow_drive/checkpoint/          # once (sha256 3c753352...)
bash sim/autonlab/submit.sh ARMS="fd fd_base fd_repair_cv" JOBS=1 CHUNKS=0 CPUS=4 MEM=32G \
    RUN_ROOT=/zfsauton/scratch/$USER/nuplan_wm/runs/smoke_fd OVERRIDES=scenario_filter.limit_total_scenarios=3
bash sim/autonlab/submit.sh ARMS="fd fd_repair_cv" CPUS=16 MEM=88G GRES=gpu:rtx_2080_ti:1   # Val14, 4 jobs per arm
bash sim/autonlab/submit.sh ARMS=fd_repair_anchor CPUS=16 MEM=88G GRES=gpu:v100:1
```

Outputs: `/zfsauton/scratch/$USER/nuplan_wm/runs/<val14|smoke_fd>/closed_loop_nonreactive_agents/readout.md`. Smoke
(2026-11-02, jobs 93410-93413): `fd_base` == `fd` (+0.00, 0 better / 0 worse); `fd_repair_cv` moved 4% of 448
replans. A legacy node has 187.5 GB, so 88 GB fits two 16-worker jobs per node; qos_legacy caps CPUs per user
(9 x 16 ran, the rest waited on `QOSMaxCpuPerUserLimit`).

## Running (babel)

```bash
# once, on the login node, in the checkout (sim/run_chunks.sh's default FD_CKPT; gitignored)
mkdir -p flow_drive/checkpoint
curl -fL -o flow_drive/checkpoint/flow_drive_model.pth \
    https://github.com/einsteinguang/flow_drive_planner/raw/06f941e3681a6a508b3e0e09ec35e2256ac26771/flow_drive/checkpoint/flow_drive_model.pth
sha256sum flow_drive/checkpoint/flow_drive_model.pth   # 3c7533526574f174815d62598c52be05c2c9c9b3332d64b9109e0028f56cc6b2

# smoke: 3 scenarios of chunk_000, own RUN_ROOT; fd_base must read out as fd exactly (+0.00, 0 better / 0 worse)
for a in fd fd_base fd_repair_cv; do
  sbatch sim/run_chunks.sh ARM=$a CHUNKS=0 RUN_ROOT=/data/user_data/$USER/nuplan_wm/runs/smoke_fd \
      OVERRIDES="scenario_filter.limit_total_scenarios=3"
done
python -m sim.readout --run_root /data/user_data/$USER/nuplan_wm/runs/smoke_fd/closed_loop_nonreactive_agents --base fd
# Val14 NR: fd (target 91.21), then the repair arms, read out against fd
sbatch sim/run_chunks.sh ARM=fd
sbatch sim/run_chunks.sh ARM=fd_repair_cv
sbatch sim/run_chunks.sh ARM=fd_repair_anchor ANCHOR_CKPT=/data/user_data/$USER/nuplan_wm/checkpoints/anchor_wm_joint_current_20261026_011310/final.pt
python -m sim.readout --run_root /data/user_data/$USER/nuplan_wm/runs/val14/closed_loop_nonreactive_agents --base fd
```
