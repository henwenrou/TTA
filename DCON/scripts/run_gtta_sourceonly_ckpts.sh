#!/bin/bash
# Run medical GTTA on DCON source-only checkpoints.
#
# This lightweight GTTA is source-dependent TTA: source labels supervise
# adaptation, while target labels are used only by the evaluation code.

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

GTTA_LR=${GTTA_LR:-2.5e-4}
GTTA_MOMENTUM=${GTTA_MOMENTUM:-0.9}
GTTA_WD=${GTTA_WD:-5e-4}
GTTA_STEPS=${GTTA_STEPS:-1}
GTTA_SRC_BATCH_SIZE=${GTTA_SRC_BATCH_SIZE:-2}
GTTA_LAMBDA_CE_TRG=${GTTA_LAMBDA_CE_TRG:-0.1}
GTTA_PSEUDO_MOMENTUM=${GTTA_PSEUDO_MOMENTUM:-0.9}
GTTA_STYLE_ALPHA=${GTTA_STYLE_ALPHA:-1.0}
GTTA_INCLUDE_ORIGINAL=${GTTA_INCLUDE_ORIGINAL:-1}
GTTA_EPISODIC=${GTTA_EPISODIC:-false}

run_gtta() {
  local expname=$1
  local data_name=$2
  local nclass=$3
  local source=$4
  local target=$5
  local ckpt=$6

  echo "=========================================="
  echo "GTTA: ${data_name}, source=${source}, target=${target}, ckpt=${ckpt}"
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
    --tta gtta \
    --gtta_lr "${GTTA_LR}" \
    --gtta_momentum "${GTTA_MOMENTUM}" \
    --gtta_wd "${GTTA_WD}" \
    --gtta_steps "${GTTA_STEPS}" \
    --gtta_src_batch_size "${GTTA_SRC_BATCH_SIZE}" \
    --gtta_lambda_ce_trg "${GTTA_LAMBDA_CE_TRG}" \
    --gtta_pseudo_momentum "${GTTA_PSEUDO_MOMENTUM}" \
    --gtta_style_alpha "${GTTA_STYLE_ALPHA}" \
    --gtta_include_original "${GTTA_INCLUDE_ORIGINAL}" \
    --gtta_episodic "${GTTA_EPISODIC}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0

  echo ""
}

run_gtta "gtta_dcon_sc_chaost2" "ABDOMINAL" 5 "SABSCT" "CHAOST2" "../ckpts/dcon-sc-300.pth"
run_gtta "gtta_dcon_cs_sabsct" "ABDOMINAL" 5 "CHAOST2" "SABSCT" "../ckpts/dcon-cs-200.pth"
run_gtta "gtta_dcon_bl_lge" "CARDIAC" 4 "bSSFP" "LGE" "../ckpts/dcon-bl-1200.pth"
run_gtta "gtta_dcon_lb_bssfp" "CARDIAC" 4 "LGE" "bSSFP" "../ckpts/dcon-lb-500.pth"

echo "=========================================="
echo "GTTA DCON TTA evaluations completed!"
echo "=========================================="
