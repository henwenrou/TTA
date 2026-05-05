#!/bin/bash
# Run VPTTA on DCON source-only checkpoints.
#
# VPTTA is source-free TTA: target labels are used only by the evaluator, and
# adaptation updates only the low-frequency prompt.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

NUM_WORKERS=${NUM_WORKERS:-4}
GPU_IDS=${GPU_IDS:-0}
if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi
RESULTS_DIR=${RESULTS_DIR:-results}

VPTTA_OPTIMIZER=${VPTTA_OPTIMIZER:-Adam}
VPTTA_LR=${VPTTA_LR:-1e-2}
VPTTA_MOMENTUM=${VPTTA_MOMENTUM:-0.99}
VPTTA_BETA1=${VPTTA_BETA1:-0.9}
VPTTA_BETA2=${VPTTA_BETA2:-0.99}
VPTTA_WEIGHT_DECAY=${VPTTA_WEIGHT_DECAY:-0.0}
VPTTA_STEPS=${VPTTA_STEPS:-1}
VPTTA_MEMORY_SIZE=${VPTTA_MEMORY_SIZE:-40}
VPTTA_NEIGHBOR=${VPTTA_NEIGHBOR:-16}
VPTTA_PROMPT_ALPHA=${VPTTA_PROMPT_ALPHA:-0.01}
VPTTA_PROMPT_SIZE=${VPTTA_PROMPT_SIZE:-}
VPTTA_IMAGE_SIZE=${VPTTA_IMAGE_SIZE:-192}
VPTTA_WARM_N=${VPTTA_WARM_N:-5}

run_vptta() {
  local expname=$1
  local data_name=$2
  local nclass=$3
  local source=$4
  local target=$5
  local ckpt=$6

  echo "=========================================="
  echo "VPTTA: ${data_name}, source=${source}, target=${target}, ckpt=${ckpt}"
  echo "=========================================="

  local prompt_size_args=()
  if [ -n "${VPTTA_PROMPT_SIZE}" ]; then
    prompt_size_args=(--vptta_prompt_size "${VPTTA_PROMPT_SIZE}")
  fi

  "${PYTHON_BIN}" train.py \
    --phase test \
    --expname "${expname}" \
    --ckpt_dir "${RESULTS_DIR}" \
    --dataset "${data_name}" \
    --nclass "${nclass}" \
    --source "${source}" \
    --target "${target}" \
    --restore_from "${ckpt}" \
    --gpu_ids "${GPU_IDS}" \
    --num_workers "${NUM_WORKERS}" \
    --tta vptta \
    --vptta_optimizer "${VPTTA_OPTIMIZER}" \
    --vptta_lr "${VPTTA_LR}" \
    --vptta_momentum "${VPTTA_MOMENTUM}" \
    --vptta_beta1 "${VPTTA_BETA1}" \
    --vptta_beta2 "${VPTTA_BETA2}" \
    --vptta_weight_decay "${VPTTA_WEIGHT_DECAY}" \
    --vptta_steps "${VPTTA_STEPS}" \
    --vptta_memory_size "${VPTTA_MEMORY_SIZE}" \
    --vptta_neighbor "${VPTTA_NEIGHBOR}" \
    --vptta_prompt_alpha "${VPTTA_PROMPT_ALPHA}" \
    "${prompt_size_args[@]}" \
    --vptta_image_size "${VPTTA_IMAGE_SIZE}" \
    --vptta_warm_n "${VPTTA_WARM_N}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0

  echo ""
}

run_vptta "vptta_dcon_sc_chaost2" "ABDOMINAL" 5 "SABSCT" "CHAOST2" "../ckpts/dcon-sc-300.pth"
run_vptta "vptta_dcon_cs_sabsct" "ABDOMINAL" 5 "CHAOST2" "SABSCT" "../ckpts/dcon-cs-200.pth"
run_vptta "vptta_dcon_bl_lge" "CARDIAC" 4 "bSSFP" "LGE" "../ckpts/dcon-bl-1200.pth"
run_vptta "vptta_dcon_lb_bssfp" "CARDIAC" 4 "LGE" "bSSFP" "../ckpts/dcon-lb-500.pth"

echo "=========================================="
echo "VPTTA DCON TTA evaluations completed!"
echo "=========================================="
