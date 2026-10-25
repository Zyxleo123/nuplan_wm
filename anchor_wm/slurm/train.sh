#!/bin/bash
#SBATCH --job-name=awm_train
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/awm_train_%j.out
#
# Fit the anchors if ANCHORS does not exist yet, then train one arm. Submit from the repo root (accounts without the
# general partition: sbatch -p preempt --qos=preempt_qos ...).
#   SCORER_LOSS=current|simple   scorer loss (default current)
#   ARM=joint|separate|wm_only   scorer gradients reach the WM / detached / no scorer (default joint)
#   SMOKE=1                      200 steps, frequent validation, no wandb (inside an allocation: bash anchor_wm/slurm/train.sh)
#
#   sbatch anchor_wm/slurm/train.sh
#   SCORER_LOSS=simple sbatch anchor_wm/slurm/train.sh
#   ARM=separate MAX_STEPS=60000 EXTRA_ARGS="--batch_size 32" sbatch anchor_wm/slurm/train.sh
set -euo pipefail
source "${ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}/anchor_wm/slurm/env.sh"

if [[ ! -f "${ANCHORS}" ]]; then
  python -m anchor_wm.anchors --cache_dir "${CACHE_DIR}" --out "${ANCHORS%.npz}" --num_anchors 64
fi

ARM="${ARM:-joint}"
SCORER_LOSS="${SCORER_LOSS:-current}"
case "${ARM}" in
  joint)    ARM_ARGS=(--joint_grad_scale "${JOINT_SCALE:-1}") ;;
  separate) ARM_ARGS=(--joint_grad_scale 0) ;;
  wm_only)  ARM_ARGS=(--no_scorer) ;;
  *) echo "ERROR: ARM must be joint|separate|wm_only, got ${ARM}" >&2; exit 1 ;;
esac
RUN_NAME="${RUN_NAME:-${ARM}_${SCORER_LOSS}}"
ARGS=(--max_steps "${MAX_STEPS:-100000}" --max_hours "${MAX_HOURS:-46}")
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME="smoke_${RUN_NAME}"; WANDB_MODE=disabled
  ARGS=(--max_steps 200 --log_every 20 --val_every 100 --val_batches 10 --save_every 100 --warmup_steps 50
        --joint_warmup_steps 50 --joint_ramp_steps 50)
fi
python -m anchor_wm.train --cache_dir "${CACHE_DIR}" --anchors "${ANCHORS}" --save_dir "${OUT_ROOT}/runs" \
  --run_name "${RUN_NAME}" --scorer_loss "${SCORER_LOSS}" "${ARM_ARGS[@]}" "${ARGS[@]}" \
  --num_workers "${NUM_WORKERS:-12}" --wandb_project "${WANDB_PROJECT}" --wandb_entity "${WANDB_ENTITY}" \
  --wandb_mode "${WANDB_MODE:-online}" ${RESUME:+--resume "${RESUME}"} ${EXTRA_ARGS:-}
