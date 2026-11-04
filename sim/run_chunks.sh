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
# Closed-loop nuPlan simulation of one arm over the chunk filters written by sim.split_filter.
# Resumable: a chunk whose `done` marker exists is skipped, so a requeued job (or a rerun) continues where it
# stopped; an unfinished chunk restarts from scratch. Configure with VAR=value arguments (or the environment):
#
#   ARM        dp | base | repair_<f> | es_<f> | es_norepair, with <f> a neighbour forecast cv | dp | anchor | none:
#              dp: Diffusion Planner as released (unseeded). base: the seeded repair planner with the repair off.
#              repair_<f>: + the repair with that forecast. es_<f>: Diffusion-ES scored by AnchorWM, every candidate
#              repaired with that forecast (diffusion_es/planner.py). es_norepair: Diffusion-ES + scorer alone.
#              base, repair_* and es_* share the per-replan sampling seed.
#              FlowDrive in place of Diffusion Planner (flow_drive/, checkpoint FD_CKPT), <f> cv | anchor | none:
#              fd: FlowDrive as released, alone (no guidance, no selection). fd_base: the repair planner with the
#              repair off (== fd: same seeds). fd_repair_<f>: + the repair with that forecast. fd_sel_<f>: best of
#              K FlowDrive samples under AnchorWM's scorer, each repaired with that forecast (flow_drive/select_planner.py);
#              fd_sel_norepair: best-of-K alone.
#   ANCHOR_CKPT  anchor_wm.train checkpoint: required by es_*, repair_anchor, fd_repair_anchor and fd_sel_*
#   FD_CKPT    FlowDrive checkpoint for fd* (default: ROOT/flow_drive/checkpoint/flow_drive_model.pth, the release)
#   FILTERS    split_filter's --out dir (chunk_*.yaml under scenario_filter/)
#   RUN_ROOT   output root; writes RUN_ROOT/CHALLENGE/ARM/chunk_* (ARM_TAG/chunk_* with TAG)
#   TAG        optional suffix of the arm's output dir, for variants of one ARM (e.g. an OVERRIDES sweep)
#   CHUNKS     chunk ids to run (default: all)    WORKERS    parallel simulations (ray), default 8
#   CHALLENGE  closed_loop_nonreactive_agents (default) | closed_loop_reactive_agents
#   OVERRIDES  extra Hydra overrides, e.g. "planner.diffusion_planner_es.es.population=32" (REPAIR: its old name)
#              Simulation logs are written without the planner (sim/sim_log.py); "~callback.simulation_log_callback"
#              drops them, "callback.simulation_log_callback._target_=nuplan.planning.simulation.callback.simulation_log_callback.SimulationLogCallback"
#              restores nuPlan's (weights and all).
#
#   sbatch sim/run_chunks.sh ARM=repair_cv RUN_ROOT=/data/user_data/$USER/nuplan_wm/runs/val14
#   sbatch sim/run_chunks.sh ARM=es_anchor ANCHOR_CKPT=/path/to/final.pt
#   sbatch sim/run_chunks.sh ARM=fd    # then ARM=fd_repair_cv, read out with --base fd
#   bash sim/run_chunks.sh ARM=dp CHUNKS="0 1" WORKERS=3    # inside an allocation
set -euo pipefail
for kv in "$@"; do export "${kv?}"; done

export ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-/home/yixiz/nuplan_sim/nuplan_wm}}"
[[ -f "${ROOT}/sim/run_chunks.sh" ]] || ROOT=/home/yixiz/nuplan_sim/nuplan_wm
export CONDA_ROOT="${CONDA_ROOT:-/data/user_data/${USER}/miniforge3}" CONDA_ENV="${CONDA_ENV:-nuplan_wm}"
# shellcheck source=/dev/null
source "${ROOT}/anchor_wm/slurm/env.sh"

ARM="${ARM:?ARM=dp|base|repair_<f>|es_<f>|es_norepair|fd|fd_base|fd_repair_<f>|fd_sel_<f>|fd_sel_norepair}"
NP_ROOT="${NP_ROOT:-/data/user_data/${USER}/nuplan_wm}"
FILTERS="${FILTERS:-${NP_ROOT}/filters/val14}"
RUN_ROOT="${RUN_ROOT:-${NP_ROOT}/runs/val14}"
CKPT_DIR="${CKPT_DIR:-${NP_ROOT}/checkpoints/diffusion_planner_release}"
FD_CKPT="${FD_CKPT:-${ROOT}/flow_drive/checkpoint/flow_drive_model.pth}"   # upstream's layout; flow_drive/README.md
CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
OVERRIDES="${OVERRIDES:-${REPAIR:-}}"
ANCHOR_CKPT="${ANCHOR_CKPT:-}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/data/datasets/nuPlan-v1.1}"
export NUPLAN_MAPS_ROOT NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-${NP_ROOT}/exp}" HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
nvidia-smi -L || true
NAME="${ARM}${TAG:+_${TAG}}"
echo "ARM=${ARM} NAME=${NAME} CHALLENGE=${CHALLENGE} WORKERS=${WORKERS} FILTERS=${FILTERS} RUN_ROOT=${RUN_ROOT}"
echo "ANCHOR_CKPT=${ANCHOR_CKPT} OVERRIDES=${OVERRIDES}"
echo "code: $(git -C "${ROOT}" rev-parse --short HEAD) $(git -C "${ROOT}" status --porcelain | wc -l) uncommitted"

STATS=""
case "${ARM}" in
  dp)          PLANNER=diffusion_planner; EXTRA=() ;;
  fd)          PLANNER=flow_drive; EXTRA=() ;;
  fd_base)     PLANNER=flow_drive_repair; EXTRA=("planner.${PLANNER}.enable_repair=false") ;;
  fd_repair_*) PLANNER=flow_drive_repair; STATS=repair_stats
               EXTRA=("planner.${PLANNER}.neighbor_forecast=${ARM#fd_repair_}") ;;
  fd_sel_norepair) PLANNER=flow_drive_select; STATS=es_stats
               EXTRA=("planner.${PLANNER}.enable_repair=false") ;;
  fd_sel_*)    PLANNER=flow_drive_select; STATS=es_stats
               EXTRA=("planner.${PLANNER}.neighbor_forecast=${ARM#fd_sel_}") ;;
  base)        PLANNER=diffusion_planner_repair; EXTRA=("planner.${PLANNER}.enable_repair=false") ;;
  repair_*)    PLANNER=diffusion_planner_repair; STATS=repair_stats
               EXTRA=("planner.${PLANNER}.neighbor_forecast=${ARM#repair_}") ;;
  es_norepair) PLANNER=diffusion_planner_es; STATS=es_stats
               EXTRA=("planner.${PLANNER}.enable_repair=false") ;;
  es_*)        PLANNER=diffusion_planner_es; STATS=es_stats
               EXTRA=("planner.${PLANNER}.neighbor_forecast=${ARM#es_}") ;;
  *)           echo "unknown ARM ${ARM}"; exit 2 ;;
esac
if [[ "${PLANNER}" == diffusion_planner_es || "${PLANNER}" == flow_drive_select || "${ARM}" == repair_anchor \
      || "${ARM}" == fd_repair_anchor ]]; then
  [[ -f "${ANCHOR_CKPT}" ]] || { echo "ARM=${ARM} needs ANCHOR_CKPT (an anchor_wm.train checkpoint); got '${ANCHOR_CKPT}'"; exit 2; }
  EXTRA+=("planner.${PLANNER}.anchor_ckpt=${ANCHOR_CKPT}")
fi
if [[ "${PLANNER}" == flow_drive* ]]; then
  [[ -f "${FD_CKPT}" ]] || { echo "ARM=${ARM} needs FD_CKPT (FlowDrive's flow_drive_model.pth); got '${FD_CKPT}'"; exit 2; }
  echo "FD_CKPT=${FD_CKPT} sha256 $(sha256sum "${FD_CKPT}" | cut -c1-16)"
  MODEL=("planner.${PLANNER}.ckpt_path=${FD_CKPT}")
else
  MODEL=("planner.${PLANNER}.config.args_file=${CKPT_DIR}/args.json" "planner.${PLANNER}.ckpt_path=${CKPT_DIR}/model.pth")
fi
GPU_FRAC="${GPU_FRAC:-$(python -c "print(round(0.9 / ${WORKERS}, 3))")}"   # GPU_FRAC=0 for a CPU-only run

if [[ -n "${CHUNKS:-}" ]]; then
  ids=(); for c in ${CHUNKS}; do ids+=("$(printf 'chunk_%03d' "$c")"); done
else
  ids=(); for f in "${FILTERS}"/scenario_filter/chunk_*.yaml; do ids+=("$(basename "$f" .yaml)"); done
fi

for id in "${ids[@]}"; do
  # nuPlan's metric aggregator only reads metric files whose path contains the challenge name.
  out="${RUN_ROOT}/${CHALLENGE}/${NAME}/${id}"
  if [[ -f "${out}/done" ]]; then echo "skip ${id} (done)"; continue; fi
  rm -rf "${out}"; mkdir -p "${out}"
  stats=()
  [[ -n "${STATS}" ]] && stats=("planner.${PLANNER}.stats_dir=${out}/${STATS}")
  echo "-- $(date +%T) ${id}"
  rc=0
  # shellcheck disable=SC2086
  python "$(python -c 'import nuplan, os; print(os.path.dirname(nuplan.__file__))')/planning/script/run_simulation.py" \
    +simulation="${CHALLENGE}" planner="${PLANNER}" "${MODEL[@]}" \
    callback.simulation_log_callback._target_=sim.sim_log.PlannerlessSimulationLogCallback \
    "${EXTRA[@]}" "${stats[@]}" ${OVERRIDES} \
    scenario_builder=nuplan "scenario_builder.data_root=${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/trainval" \
    scenario_filter="${id}" "output_dir=${out}" "experiment_uid=${NAME}/${id}" \
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
echo "== $(date) all chunks of ${NAME} done"
