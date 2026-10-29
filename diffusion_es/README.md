# `diffusion_es`: Diffusion-ES + AnchorWM scorer (+ the o_am repair) on nuPlan

The WOMD planner of `yixiz/waymax_rs` (`planner/diffusion_es_planner.py`, branch `es_wm_disc`) on Diffusion
Planner, inside nuPlan's `run_simulation`. Every replan:

```
Diffusion Planner ── K samples (one encoder pass, decoder batched) ──► repair each (posthoc_repair, optional)
        ▲                                                                       │
        │  elites re-noised to resample_t, denoised (resample_steps)            ▼
        └──────────────── top E ◄──── AnchorWM PlanScorer total score (learned reward)
                                   ... iterations rounds ... ──► best repaired plan ──► simulator
```

As on WOMD, candidates are repaired before they are scored, so elites are chosen on the repaired plans and the
executed plan is a repaired one, while re-noising restarts from the elite's raw diffusion sample. Elites keep
their scores, so the best score never drops between rounds. AnchorWM runs once per replan on the online
cache-format scene (`posthoc_repair/anchor_forecast.py`, now with the drivable-area boundary the scorer reads,
cropped as `build_cache` crops it); the same forward gives the repair its `anchor` forecast.

| File | What it does |
|---|---|
| `es.py` | `ESConfig`, `PopulationSampler` (Diffusion Planner's decoder over a population; prior samples or re-noised proposals), `diffusion_es` (the search), nuPlan-free |
| `planner.py` | `ESRepairDiffusionPlanner`: the nuPlan planner (`planner=diffusion_planner_es`) |
| `check.py` | gates on a synthetic road, no nuPlan needed; `--write_random_anchor_ckpt` for plumbing runs |

`posthoc_repair/anchor_model.py` is AnchorWM at planning time (one forward, then `forecast()` / `score()`),
shared with the repair planner.

Config: `diffusion_planner/config/planner/diffusion_planner_es.yaml`. `anchor_ckpt` is required (a checkpoint
trained with the scorer; a `--no_scorer` one is refused), `es: {population, elites, iterations, resample_t,
resample_steps, noise_scale, sample_steps}`, `enable_repair`, `neighbor_forecast` (the repair's), `repair_device:
cuda` (the repair runs on the whole population every round), `seed` (as the repair planner: sample 0 of the
first population is the seeded `base` plan).

## Arms (`sim/run_chunks.sh`)

| ARM | planner |
|---|---|
| `es_anchor` / `es_cv` / `es_dp` / `es_none` | ES + scorer, every candidate repaired with that forecast |
| `es_norepair` | ES + scorer alone |
| `base`, `repair_*`, `dp` | as before (`posthoc_repair/README.md`) |

The ES arms and `repair_anchor` take `ANCHOR_CKPT=<run>/final.pt`. Per-replan stats go to
`<chunk>/es_stats/*.jsonl`; `sim.readout` summarises them (population spread, which round the chosen plan was born
in, score gain over sample 0, repair counts, per-stage times).

## Checks (`check.py`, autonlab CPU, 2026-10-28)

On a 3-lane road with a lead car, released Diffusion Planner (`ae5baf1`) and a random-init AnchorWM:

* population sampling == Diffusion Planner's decoder for the same seed (bit-identical);
* re-noising an echo denoiser's sample to t = 0.02 / 0.1 / 0.3 / 1.0 moves it 0.9 / 4.2 / 12.9 / 34 m;
* the search climbs a known objective (end 3 s into the left lane) every round;
* the scorer is permutation / chunk consistent (2e-7); batched repair == per-plan repair in float64 (3e-12 m);
* one full replan, K = 64, 4 rounds: 5.0 s on CPU, 3.9 s of it scoring.

**The released Diffusion Planner is near-deterministic on that road**: 32 samples spread ~1 mm (mean distance to
the mean plan; ~2 mm at endpoints), at prior noise 0.5, 1 and 2 alike. Its DiT predicts nearly the same x0 whatever
its noisy input (output change <= 0.07, normalised, for an input change of 6) and DPM-Solver++ converges every
sample to it. A population that does not vary gives ES and the scorer nothing to choose between, so on such
scenes this planner is Diffusion Planner + repair. Whether real Val14 scenes (intersections, lane choices) spread
more is the first thing the smoke run measures (`spread0_m` in the stats, `population spread` in the readout).

The float32 repair is sensitive to summation order: batched vs single differs by up to 0.32 m on a charged plan
(32 iterations through a piecewise loss), so two runs whose populations differ do not repair a shared plan
bit-identically.

## Not done yet

* No closed-loop run: nuPlan and its data are on babel only. Smoke: see below.
* No trained AnchorWM checkpoint with a scorer is readable here (`/data/user_data/ishanj/anchor_wm_nuplan` is
  `jeff4_group`); a random-init one runs the plumbing only.
* `es` constants are the WOMD ones mapped to the VP-linear schedule, untuned.

## Running (babel)

```bash
# plumbing smoke: random-init scorer, 3 scenarios of chunk_000, its own RUN_ROOT
PYTHONPATH=. python -m diffusion_es.check --write_random_anchor_ckpt /data/user_data/$USER/nuplan_wm/checkpoints/anchor_random.pt
sbatch sim/run_chunks.sh ARM=es_anchor ANCHOR_CKPT=/data/user_data/$USER/nuplan_wm/checkpoints/anchor_random.pt \
    CHUNKS=0 RUN_ROOT=/data/user_data/$USER/nuplan_wm/runs/smoke_es OVERRIDES="scenario_filter.limit_total_scenarios=3"
python -m sim.readout --run_root /data/user_data/$USER/nuplan_wm/runs/smoke_es/closed_loop_nonreactive_agents
# real arms, once a trained checkpoint is readable
sbatch sim/run_chunks.sh ARM=es_anchor ANCHOR_CKPT=<run>/final.pt
sbatch sim/run_chunks.sh ARM=es_norepair ANCHOR_CKPT=<run>/final.pt
```
