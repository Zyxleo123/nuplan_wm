# autonlab (lov*) defaults for the sim jobs, sourced by sim/autonlab/*.sbatch with the job's arguments: an argument
# that is a file is sourced (the submitter's env file), a VAR=value one is exported; the defaults fill what is unset.
for a in "$@"; do
  if [[ -f "${a}" ]]; then
    # shellcheck source=/dev/null
    source "${a}"
  elif [[ "${a}" == *=* ]]; then
    export "${a?}"
  fi
done

export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "${USER}" | cut -d: -f6)}"
export ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/nuplan_wm}}"
[[ -f "${ROOT}/sim/run_chunks.sh" ]] || ROOT=/zfsauton2/home/yixiz/nuplan_wm
export CONDA_ROOT="${CONDA_ROOT:-/zfsauton/scratch/${USER}/miniconda3}"
# mineuih's env: python 3.9, torch 2.0.0+cu118, timm 1.0.10, nuplan-devkit 1.2.2 (editable, world-readable), ray
export CONDA_ENV="${CONDA_ENV:-/zfsauton/scratch/mineuih/conda_envs/nuplan_wm}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/zfsauton/datasets/ArgoRL/brianyan/nuplan/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NUPLAN_DATA_ROOT}/maps}"
export NP_ROOT="${NP_ROOT:-/zfsauton/scratch/${USER}/nuplan_wm}"
export CKPT_DIR="${CKPT_DIR:-${NP_ROOT}/checkpoints/diffusion_planner_release}"
export ANCHOR_CKPT="${ANCHOR_CKPT:-${NP_ROOT}/checkpoints/anchor_wm_joint_current_20261026_011310/final.pt}"
export FILTERS="${FILTERS:-${NP_ROOT}/filters/val14}"
export RUN_ROOT="${RUN_ROOT:-${NP_ROOT}/runs/val14}"
# mineuih's env has protobuf 6.33 with tensorboard 2.11, whose _pb2 files predate protobuf 4: nuPlan's TimingCallback
# imports torch.utils.tensorboard and dies ("Descriptors cannot be created directly"; jobs 93362-93370). The pure-Python
# backend takes both old and new generated code; only protobuf calls (Ray's control messages) get slower.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
# one Ray session directory per job: two jobs on a node must not share /tmp/ray
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-$$}}"
