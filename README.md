# nuplan_wm

Preprocesses nuPlan data into training samples (`.npz`) in the [Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner) format.

## 1. Create the Conda Environment

```bash
conda create -n nuplan_wm python=3.9
conda activate nuplan_wm
```

> To create the environment at a specific location, use `conda create -p /path/to/conda_envs/nuplan_wm python=3.9`.

## 2. Install Packages

### 2.1 nuplan-devkit

```bash
# from the repository root
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit

# pip >= 24.1 rejects the legacy metadata of hydra-core==1.1.0rc1 / omegaconf==2.1.0rc1,
# which makes the whole requirements.txt install fail. Downgrade pip first.
pip install "pip<24.1"

pip install -e .
pip install -r requirements.txt
cd ..
```

### 2.2 Replace OpenCV with the headless build

On servers without GUI libraries, `import cv2` fails with
`ImportError: libgthread-2.0.so.0: cannot open shared object file`.
Replace it with the headless build of the same version:

```bash
pip uninstall -y opencv-python
pip install "opencv-python-headless==4.5.1.48"
```

> Reinstalling `nuplan-devkit/requirements.txt` brings `opencv-python` back, so repeat this step if you do.

### 2.3 diffusion_planner and PyTorch

```bash
# from the repository root
pip install -e .
pip install -r requirements_torch.txt
```

`requirements_torch.txt` installs `torch==2.0.0+cu118`, `torchvision==0.15.1+cu118`, `pytorch_lightning==2.0.1`, and a few others.

### 2.4 Verify the installation

```bash
python -c "import numpy, scipy, cv2, torch, nuplan; print(numpy.__version__, cv2.__version__, torch.__version__)"
# 1.23.4 4.5.1 2.0.0+cu118
```

## 3. Data Preprocessing (data_process)

### 3.1 Data paths

The nuPlan dataset follows the layout in the [official docs](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html). `--data_path` must point to the directory that contains the `.db` files.

```
dataset/
├── maps/                     # --map_path
│   ├── nuplan-maps-v1.0.json
│   ├── sg-one-north/
│   ├── us-ma-boston/
│   └── ...
└── nuplan-v1.1/
    └── splits/
        ├── mini/
        ├── test/
        └── trainval/         # --data_path (*.db files)
```

### 3.2 Configuration

Edit the configuration section of [data_process.sh](data_process.sh):

```bash
NUPLAN_DATA_PATH="/path/to/dataset/nuplan-v1.1/splits/trainval"
NUPLAN_MAP_PATH="/path/to/dataset/maps"
TRAIN_SET_PATH="/path/to/cache"   # where the preprocessed .npz files are saved
```

Arguments after `bash data_process.sh` are passed through to `data_process.py` (e.g. `bash data_process.sh --num_workers 32`).

Main arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `/data/nuplan-v1.1/trainval` | Directory of nuPlan `.db` files |
| `--map_path` | `/data/nuplan-v1.1/maps` | nuPlan map directory |
| `--save_path` | `./cache` | Output directory for `.npz` files |
| `--total_scenarios` | `10` | Maximum number of scenarios to process |
| `--scenarios_per_type` | `None` | Maximum number of scenarios per scenario type |
| `--num_workers` | number of available CPU cores | Number of parallel processes |
| `--max_tasks_per_child` | `500` | Restart each worker after this many scenarios to release cached memory |
| `--num_shards` / `--shard_id` | `1` / `0` | Split the scenarios across nodes (see 3.4) |
| `--seed` | `0` | Random seed for scenario sampling; must be the same on every shard |
| `--agent_num` / `--static_objects_num` | `32` / `5` | Max neighbor agents / static objects |
| `--lane_num` / `--lane_len` | `70` / `20` | Max lanes / points per lane |
| `--route_num` / `--route_len` | `25` / `20` | Max route lanes / points per route lane |

### 3.3 Run

```bash
conda activate nuplan_wm
bash data_process.sh
```

The script runs in three steps:

1. Filters training scenarios by the log names in `nuplan_train.json` (`get_scenarios`, multiprocess). Sampling and shuffling use a fixed `--seed`, so the scenario list is the same on every run.
2. Sorts the scenarios by log, then for each scenario extracts ego and neighbor-agent past/future trajectories, static objects, and the vector map, and saves them to `{save_path}/{map_name}_{token}.npz`. This runs in parallel across `--num_workers` processes; the tqdm progress bar advances once per finished scenario.
3. Writes the list of generated `.npz` files to `diffusion_planner_training.json` in the current directory.

**Resuming.** Scenarios whose `.npz` already exists are skipped, so an interrupted run can be restarted with the same command. Each file is written to a temp file (`*.npz.tmp.<host>.<pid>`) and then renamed, so an interruption does not leave a truncated `.npz`.

### 3.4 Multi-node processing

When the dataset and `--save_path` are on a shared filesystem, the work can be split across nodes. Run the same command on each node with a different `--shard_id`:

```bash
bash data_process.sh --num_shards 4 --shard_id 0   # node 1
bash data_process.sh --num_shards 4 --shard_id 1   # node 2
bash data_process.sh --num_shards 4 --shard_id 2   # node 3
bash data_process.sh --num_shards 4 --shard_id 3   # node 4
```

- Each shard processes a contiguous block of the log-sorted scenario list, so the shards never overlap and together cover the same scenarios as a single-node run.
- All other arguments (paths, `--total_scenarios`, `--seed`) must be identical across shards.
- `diffusion_planner_training.json` is written only by the shard that finishes last, once every scenario's `.npz` exists. Shards that finish earlier print `... scenarios from other shards are not finished yet` and exit; this is expected.
- If a shard fails, rerun it with the same arguments. It skips the files already written and writes the JSON when everything is complete.

### 3.5 Checking the cache for broken files

[check_cache.py](check_cache.py) loads every `.npz` under `--save_path` and deletes the ones that cannot be read, along with leftover temp files from interrupted writes. Files written by an older version of the code, before atomic writes were added, can be truncated if the run was killed.

```bash
python check_cache.py --save_path /path/to/cache --dry_run   # list only
python check_cache.py --save_path /path/to/cache             # delete
```

Do not run it while any `data_process.py` (on any node) is writing to the same directory: it would delete in-progress temp files and crash those workers.

### 3.6 Notes

- **Scenario collection is slow.** Step 1 opens every `.db` file under `--data_path` (about 12,000 for trainval) regardless of `--total_scenarios`. On a network filesystem this step alone can take tens of minutes, and no progress bar is shown while it runs.
- **Threads.** `data_process.py` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 (unless already set), so each worker uses a single BLAS thread. Without this, every worker spawns one thread per core and the workers slow each other down.
- **Memory.** Each worker loads its own copy of the maps and caches every map object it queries, so memory grows with `--num_workers` and over time. Workers are restarted every `--max_tasks_per_child` scenarios to release this cache. If memory is still tight, lower `--num_workers` or `--max_tasks_per_child`.
- **Hangs.** If a worker is killed (e.g. by the OOM killer), `multiprocessing.Pool` waits forever and the progress bar stops without an error. Check with `dmesg -T | grep -iE "out of memory|killed process"`, then stop the run, run `check_cache.py`, and restart; finished scenarios are skipped.
- For a small test run, use e.g. `--total_scenarios 10 --num_workers 4`.
- For training and simulation, see the original [Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner) repository.
