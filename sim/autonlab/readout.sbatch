#!/bin/bash
#SBATCH --job-name=np_readout
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/np_readout_%j.out
# sim.readout over RUN_ROOT/CHALLENGE (every arm there), paired against BASE (default fd), written to OUT.{md,json}
# (default RUN_ROOT/CHALLENGE/readout); reads saved metrics only.
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/nuplan_wm}"
[[ -f "${R}/sim/autonlab/site.sh" ]] || R=/zfsauton2/home/yixiz/nuplan_wm
# shellcheck source=/dev/null
source "${R}/sim/autonlab/site.sh" "$@"
# shellcheck source=/dev/null
source "${ROOT}/anchor_wm/slurm/env.sh"
echo "== autonlab readout: host=$(hostname) job=${SLURM_JOB_ID:-local} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
python -m sim.readout --run_root "${RUN_ROOT}/${CHALLENGE:-closed_loop_nonreactive_agents}" --base "${BASE:-fd}" ${OUT:+--out "${OUT}"}
