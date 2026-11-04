#!/bin/bash
#SBATCH --job-name=np_sim
#SBATCH --partition=legacy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/np_sim_%j.out
# One arm of sim/run_chunks.sh on autonlab (site.sh's paths); resumable as run_chunks.sh is. Arguments: the
# submitter's env file and VAR=value pairs (ARM, CHUNKS, WORKERS, RUN_ROOT, OVERRIDES, ...), e.g.
#   sbatch sim/autonlab/sim.sbatch ARM=fd CHUNKS="0 1 2 3"
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/nuplan_wm}"
[[ -f "${R}/sim/autonlab/site.sh" ]] || R=/zfsauton2/home/yixiz/nuplan_wm
# shellcheck source=/dev/null
source "${R}/sim/autonlab/site.sh" "$@"
mkdir -p "${RAY_TMPDIR}"
# GPU memory every 2 min: a CUDA allocation failure can kill a worker with no Python traceback
( while sleep 120; do echo "gpu-mem $(date +%T) $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"; done ) &
mem_log=$!
trap 'kill "${mem_log}" 2>/dev/null; rm -rf "${RAY_TMPDIR}"' EXIT
echo "== autonlab sim: host=$(hostname) job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?} gpus=${CUDA_VISIBLE_DEVICES:-none}"
bash "${ROOT}/sim/run_chunks.sh"
