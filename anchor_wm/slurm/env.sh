# Babel paths / environment for the AnchorWM scripts (sourced; submit the scripts from the repo root). Every
# variable can be overridden from the caller's environment; sbatch exports it to the job. /data/user_data is
# only mounted on compute nodes.
export ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
export CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-nuplan_wm}"
export NUPLAN_DATA="${NUPLAN_DATA:-/data/datasets/nuPlan-v1.1/nuplan-v1.1/splits/trainval}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/datasets/nuPlan-v1.1/maps}"
export OUT_ROOT="${OUT_ROOT:-/data/user_data/${USER}/anchor_wm_nuplan}"
export CACHE_DIR="${CACHE_DIR:-${OUT_ROOT}/cache}"
export PLAN_FILE="${PLAN_FILE:-${CACHE_DIR}/plan.json}"
export VAL_CACHE_DIR="${VAL_CACHE_DIR:-${OUT_ROOT}/cache_val14}"   # Val14 (validation; README, Running)
export ANCHORS="${ANCHORS:-${OUT_ROOT}/anchors/anchors_k64.npz}"
export WANDB_PROJECT="${WANDB_PROJECT:-anchor_wm_nuplan}"
export WANDB_ENTITY="${WANDB_ENTITY-scbellad2025-carnegie-mellon-university}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"   # the team's W&B host (~/.netrc)

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
fi
# The repo first, so its diffusion_planner (route correction) and anchor_wm are the ones imported.
export PYTHONPATH="${ROOT}"
umask 002   # outputs stay writable by the group (shared /data/user_data directories)
cd "${ROOT}"
mkdir -p logs
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-} cuda=${CUDA_VISIBLE_DEVICES:-} python=$(command -v python) job=${SLURM_JOB_ID:-local}"
