#!/bin/bash
#SBATCH --job-name=awm_cache
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/mineuih/ishan/logs/awm_cache_%A_%a.out
#
# Build the AnchorWM cache (anchor_wm.build_cache). As an array job every task takes one contiguous shard of
# the log-sorted scenario list and skips files that already exist, so a requeued task only fills the rest.
#
#   sbatch --array=0-19 anchor_wm/slurm/cache.sh                                  full trainval cache, 20 shards
#   CACHE_DIR=.../cache_smoke EXTRA_ARGS="--max_db_files 12 --total_scenarios 300" sbatch anchor_wm/slurm/cache.sh
set -euo pipefail
source "${ROOT:-${HOME}/ishan/nuplan_wm}/anchor_wm/slurm/env.sh"

python -m anchor_wm.build_cache --data_path "${NUPLAN_DATA}" --map_path "${NUPLAN_MAPS_ROOT}" --save_path "${CACHE_DIR}" \
  --num_workers "${SLURM_CPUS_PER_TASK:-16}" --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" --shard_id "${SLURM_ARRAY_TASK_ID:-0}" \
  ${EXTRA_ARGS:-}
