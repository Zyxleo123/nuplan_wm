#!/bin/bash
#SBATCH --job-name=awm_cache
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/awm_cache_%A_%a.out
#
# Build the AnchorWM cache (anchor_wm.build_cache). Submit from the repo root.
#   STEP=plan    list the scenarios once (same filter / seed as data_process.py) and write PLAN_FILE
#   (default)    array task: build one shard of PLAN_FILE (balanced by scenario count, existing files skipped,
#                so a preempted / requeued task only fills the rest)
#
#   STEP=plan TOTAL=1000000 sbatch --cpus-per-task=32 anchor_wm/slurm/cache.sh
#   sbatch --array=0-7 --dependency=afterok:<plan job> anchor_wm/slurm/cache.sh
#   CACHE_DIR=.../cache_smoke EXTRA_ARGS="--max_db_files 12 --total_scenarios 300" STEP=direct sbatch anchor_wm/slurm/cache.sh
set -euo pipefail
source "${ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}/anchor_wm/slurm/env.sh"

ARGS=(--data_path "${NUPLAN_DATA}" --map_path "${NUPLAN_MAPS_ROOT}" --save_path "${CACHE_DIR}"
      --num_workers "${SLURM_CPUS_PER_TASK:-16}")
case "${STEP:-build}" in
  plan)   python -m anchor_wm.build_cache "${ARGS[@]}" --total_scenarios "${TOTAL:-1000000}" --plan_file "${PLAN_FILE}" \
            --make_plan ${EXTRA_ARGS:-} ;;
  build)  [[ -f "${PLAN_FILE}" ]] || { echo "ERROR: no plan ${PLAN_FILE}; run STEP=plan first" >&2; exit 1; }
          python -m anchor_wm.build_cache "${ARGS[@]}" --plan_file "${PLAN_FILE}" \
            --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" --shard_id "${SLURM_ARRAY_TASK_ID:-0}" ${EXTRA_ARGS:-} ;;
  direct) python -m anchor_wm.build_cache "${ARGS[@]}" ${EXTRA_ARGS:-} ;;
  *) echo "ERROR: STEP must be plan|build|direct, got ${STEP}" >&2; exit 1 ;;
esac
