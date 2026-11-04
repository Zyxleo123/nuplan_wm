#!/bin/bash
#SBATCH --job-name=np_split
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/np_split_%j.out
# Val14's chunk filters (sim.split_filter) into FILTERS, once: under a lock, and a no-op (exit 0) when they exist.
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/nuplan_wm}"
[[ -f "${R}/sim/autonlab/site.sh" ]] || R=/zfsauton2/home/yixiz/nuplan_wm
# shellcheck source=/dev/null
source "${R}/sim/autonlab/site.sh" "$@"
# shellcheck source=/dev/null
source "${ROOT}/anchor_wm/slurm/env.sh"
echo "== autonlab split: host=$(hostname) job=${SLURM_JOB_ID:-local} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
nvidia-smi -L 2>/dev/null || echo "no GPU (CPU job)"
mkdir -p "$(dirname "${FILTERS}")"
exec 9> "${FILTERS}.lock"
flock 9
if ls "${FILTERS}"/scenario_filter/chunk_*.yaml > /dev/null 2>&1; then
  echo "filters already in ${FILTERS}: $(ls "${FILTERS}"/scenario_filter/chunk_*.yaml | wc -l) chunks; nothing to do"
  exit 0
fi
python -m sim.split_filter --filter diffusion_planner/config/scenario_filter/val14.yaml \
    --data_root "${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/trainval" --out "${FILTERS}" \
    --chunks "${N_CHUNKS:-16}" --threads "${SLURM_CPUS_PER_TASK:-8}"
