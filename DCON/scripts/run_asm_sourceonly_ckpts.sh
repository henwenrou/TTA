#!/bin/bash
# Run ASM on DCON source-only checkpoints.
#
# ASM is source-dependent TTA: each target batch uses target image statistics
# only, while a labeled source-domain training batch provides the supervised
# adaptation loss.

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

ASM_LR=${ASM_LR:-1e-4}
ASM_STEPS=${ASM_STEPS:-1}
ASM_INNER_STEPS=${ASM_INNER_STEPS:-2}
ASM_LAMBDA_REG=${ASM_LAMBDA_REG:-2e-4}
ASM_SAMPLING_STEP=${ASM_SAMPLING_STEP:-20.0}
ASM_SRC_BATCH_SIZE=${ASM_SRC_BATCH_SIZE:-4}
ASM_STYLE_BACKEND=${ASM_STYLE_BACKEND:-medical_adain}

run_asm() {
  local expname=$1
  local data_name=$2
  local nclass=$3
  local source=$4
  local target=$5
  local ckpt=$6

  echo "=========================================="
  echo "ASM: ${data_name}, source=${source}, target=${target}, ckpt=${ckpt}"
  echo "=========================================="

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
    --tta asm \
    --asm_lr "${ASM_LR}" \
    --asm_steps "${ASM_STEPS}" \
    --asm_inner_steps "${ASM_INNER_STEPS}" \
    --asm_lambda_reg "${ASM_LAMBDA_REG}" \
    --asm_sampling_step "${ASM_SAMPLING_STEP}" \
    --asm_src_batch_size "${ASM_SRC_BATCH_SIZE}" \
    --asm_style_backend "${ASM_STYLE_BACKEND}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0

  echo ""
}

run_asm "asm_dcon_sc_chaost2" "ABDOMINAL" 5 "SABSCT" "CHAOST2" "../ckpts/dcon-sc-300.pth"
run_asm "asm_dcon_cs_sabsct" "ABDOMINAL" 5 "CHAOST2" "SABSCT" "../ckpts/dcon-cs-200.pth"
run_asm "asm_dcon_bl_lge" "CARDIAC" 4 "bSSFP" "LGE" "../ckpts/dcon-bl-1200.pth"
run_asm "asm_dcon_lb_bssfp" "CARDIAC" 4 "LGE" "bSSFP" "../ckpts/dcon-lb-500.pth"

echo "=========================================="
echo "ASM DCON TTA evaluations completed!"
echo "=========================================="
