#!/bin/bash
#SBATCH --job-name=np_sim
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/np_sim_%j.out
# Closed-loop nuPlan simulation of one arm over the chunk filters written by posthoc_repair.sim.split_filter.
# Resumable: a chunk whose `done` marker exists is skipped, so a requeued job (or a rerun) continues where it
# stopped; an unfinished chunk restarts from scratch. Configure with VAR=value arguments (or the environment):
#
#   ARM        dp | base | repair_cv | repair_dp | repair_none: Diffusion Planner as released (unseeded); the
#              repair planner with the repair off; + the repair with that neighbour forecast. base and repair_*
#              share the per-replan sampling seed, so they differ only where the repair moved a plan.
#   FILTERS    split_filter's --out dir (chunk_*.yaml under scenario_filter/)
#   RUN_ROOT   output root; writes RUN_ROOT/CHALLENGE/ARM/chunk_*
#   CHUNKS     chunk ids to run (default: all)    WORKERS    parallel simulations (ray), default 8
#   CHALLENGE  closed_loop_nonreactive_agents (default) | closed_loop_reactive_agents
#   REPAIR     extra Hydra overrides for the repair arms, e.g. "planner.diffusion_planner_repair.repair.window=30"
#
#   sbatch posthoc_repair/sim/run_chunks.sh ARM=repair_cv RUN_ROOT=/data/user_data/$USER/nuplan_wm/runs/val14
#   bash posthoc_repair/sim/run_chunks.sh ARM=dp CHUNKS="0 1" WORKERS=3    # inside an allocation
set -euo pipefail
for kv in "$@"; do export "${kv?}"; done

export ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-/home/yixiz/nuplan_sim/nuplan_wm}}"
[[ -f "${ROOT}/posthoc_repair/sim/run_chunks.sh" ]] || ROOT=/home/yixiz/nuplan_sim/nuplan_wm
export CONDA_ROOT="${CONDA_ROOT:-/data/user_data/${USER}/miniforge3}" CONDA_ENV="${CONDA_ENV:-nuplan_wm}"
# shellcheck source=/dev/null
source "${ROOT}/anchor_wm/slurm/env.sh"

ARM="${ARM:?ARM=dp|base|repair_cv|repair_dp|repair_none}"
NP_ROOT="${NP_ROOT:-/data/user_data/${USER}/nuplan_wm}"
FILTERS="${FILTERS:-${NP_ROOT}/filters/val14}"
RUN_ROOT="${RUN_ROOT:-${NP_ROOT}/runs/val14}"
CKPT_DIR="${CKPT_DIR:-${NP_ROOT}/checkpoints/diffusion_planner_release}"
CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
REPAIR="${REPAIR:-}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/data/datasets/nuPlan-v1.1}"
export NUPLAN_MAPS_ROOT NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-${NP_ROOT}/exp}" HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
nvidia-smi -L || true
echo "ARM=${ARM} CHALLENGE=${CHALLENGE} WORKERS=${WORKERS} FILTERS=${FILTERS} RUN_ROOT=${RUN_ROOT} REPAIR=${REPAIR}"
echo "code: $(git -C "${ROOT}" rev-parse --short HEAD) $(git -C "${ROOT}" status --porcelain | wc -l) uncommitted"

case "${ARM}" in
  dp)          PLANNER=diffusion_planner; EXTRA=() ;;
  base)        PLANNER=diffusion_planner_repair; EXTRA=("planner.${PLANNER}.enable_repair=false") ;;
  repair_*)    PLANNER=diffusion_planner_repair
               EXTRA=("planner.${PLANNER}.neighbor_forecast=${ARM#repair_}") ;;
  *)           echo "unknown ARM ${ARM}"; exit 2 ;;
esac
GPU_FRAC=$(python -c "print(round(0.9 / ${WORKERS}, 3))")

if [[ -n "${CHUNKS:-}" ]]; then
  ids=(); for c in ${CHUNKS}; do ids+=("$(printf 'chunk_%03d' "$c")"); done
else
  ids=(); for f in "${FILTERS}"/scenario_filter/chunk_*.yaml; do ids+=("$(basename "$f" .yaml)"); done
fi

for id in "${ids[@]}"; do
  # nuPlan's metric aggregator only reads metric files whose path contains the challenge name.
  out="${RUN_ROOT}/${CHALLENGE}/${ARM}/${id}"
  if [[ -f "${out}/done" ]]; then echo "skip ${id} (done)"; continue; fi
  rm -rf "${out}"; mkdir -p "${out}"
  stats=()
  [[ "${ARM}" == repair_* ]] && stats=("planner.${PLANNER}.stats_dir=${out}/repair_stats")
  echo "-- $(date +%T) ${id}"
  rc=0
  # shellcheck disable=SC2086
  python "$(python -c 'import nuplan, os; print(os.path.dirname(nuplan.__file__))')/planning/script/run_simulation.py" \
    +simulation="${CHALLENGE}" planner="${PLANNER}" \
    "planner.${PLANNER}.config.args_file=${CKPT_DIR}/args.json" "planner.${PLANNER}.ckpt_path=${CKPT_DIR}/model.pth" \
    "${EXTRA[@]}" "${stats[@]}" ${REPAIR} \
    scenario_builder=nuplan "scenario_builder.data_root=${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/trainval" \
    scenario_filter="${id}" "output_dir=${out}" "experiment_uid=${ARM}/${id}" \
    worker=ray_distributed "worker.threads_per_node=${WORKERS}" distributed_mode=SINGLE_NODE \
    "number_of_gpus_allocated_per_simulation=${GPU_FRAC}" enable_simulation_progress_bar=false verbose=false \
    "hydra.searchpath=[file://${FILTERS}, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    > "${out}/sim.log" 2>&1 || rc=$?
  n_fail=$(grep -oP 'Number of failed simulations: \K[0-9]+' "${out}/sim.log" | tail -1 || true)
  if [[ ${rc} -eq 0 ]] && ls "${out}"/aggregator_metric/*.parquet > /dev/null 2>&1; then
    echo "${n_fail:-?} failed" > "${out}/done"; echo "   done ${id}: ${n_fail:-?} failed simulations"
  else
    echo "   ${id}: exit ${rc}, no aggregator output; see ${out}/sim.log"; tail -5 "${out}/sim.log"; exit 1
  fi
done
echo "== $(date) all chunks of ${ARM} done"
