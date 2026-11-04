#!/bin/bash
# Submit sim arms on autonlab (from lov*); only calls sbatch. Val14's chunk filters (cpu; a no-op once built) ->
# each ARM's sim jobs (legacy GPU, its chunks spread over JOBS sibling jobs) -> one readout (cpu, after all of them,
# whether or not each succeeded). Every job gets this submission's env file (NP_ROOT/submissions/) + its own
# ARM= / CHUNKS=. Knobs: ARMS (required), JOBS (default 4), N_CHUNKS (16), CHUNKS (default all), CPUS / MEM / GRES /
# PARTITION (sbatch resources of the sim jobs, default the sbatch file's; a CPU-only run: PARTITION=cpu GRES=none
# GPU_FRAC=0 and the planner's device=cpu in OVERRIDES); anything else is passed on (RUN_ROOT, OVERRIDES, TAG, ...).
#   bash sim/autonlab/submit.sh ARMS="fd fd_repair_cv"
#   bash sim/autonlab/submit.sh ARMS="fd fd_base fd_repair_cv" JOBS=1 CHUNKS=0 CPUS=4 MEM=32G \
#       RUN_ROOT=/zfsauton/scratch/$USER/nuplan_wm/runs/smoke_fd OVERRIDES=scenario_filter.limit_total_scenarios=3
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs
dir="/zfsauton/scratch/${USER}/nuplan_wm/submissions"
mkdir -p "${dir}"
envf="${dir}/$(date +%Y%m%d_%H%M%S)_$$.env"
: > "${envf}"
for kv in "$@"; do
  [[ "${kv}" == *=* ]] || { echo "not VAR=value: ${kv}"; exit 2; }
  printf 'export %s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${envf}"
done
get() { (source "${envf}"; eval "printf '%s' \"\${$1:-$2}\""); }
ARMS="$(get ARMS '')"
[[ -n "${ARMS}" ]] || { echo "ARMS=... is required"; exit 2; }
JOBS="$(get JOBS 4)"
TAG="$(get TAG '')"
CHUNKS="$(get CHUNKS "$(seq -s ' ' 0 $(( $(get N_CHUNKS 16) - 1 )))")"
res=()
[[ -n "$(get CPUS '')" ]] && res+=("--cpus-per-task=$(get CPUS '')")
[[ -n "$(get MEM '')" ]] && res+=("--mem=$(get MEM '')")
[[ -n "$(get GRES '')" ]] && res+=("--gres=$(get GRES '')")
[[ -n "$(get PARTITION '')" ]] && res+=("--partition=$(get PARTITION '')")

# submitting from inside an allocation: the job must not inherit this one's SLURM_* variables
sb() { env $(env | grep -o '^SLURM_[A-Za-z0-9_]*' | sed 's/^/-u /') sbatch --parsable "$@"; }

echo "env file ${envf}:"; sed 's/^/  /' "${envf}"
split=$(sb sim/autonlab/split.sbatch "${envf}")
echo "sbatch sim/autonlab/split.sbatch ${envf} -> ${split}"
read -r -a ids <<< "${CHUNKS}"
sims=()
for arm in ${ARMS}; do
  n=$(( JOBS < ${#ids[@]} ? JOBS : ${#ids[@]} ))
  for (( j = 0; j < n; j++ )); do
    part=()
    for (( i = j; i < ${#ids[@]}; i += n )); do part+=("${ids[i]}"); done
    id=$(sb --dependency=afterok:"${split}" --job-name="np_${arm}${TAG:+_${TAG}}" "${res[@]}" sim/autonlab/sim.sbatch "${envf}" \
         "ARM=${arm}" "CHUNKS=${part[*]}")
    echo "sbatch --dependency=afterok:${split} ${res[*]} sim/autonlab/sim.sbatch ${envf} ARM=${arm} CHUNKS='${part[*]}' -> ${id}"
    sims+=("${id}")
  done
done
deps=$(IFS=:; echo "${sims[*]}")
ro=$(sb --dependency=afterany:"${deps}" sim/autonlab/readout.sbatch "${envf}")
echo "sbatch --dependency=afterany:${deps} sim/autonlab/readout.sbatch ${envf} -> ${ro}"
