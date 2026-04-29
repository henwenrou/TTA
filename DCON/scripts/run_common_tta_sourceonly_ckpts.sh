#!/bin/bash
# Run the first-batch common TTA baselines on DCON source-only checkpoints.
#
# Default methods:
#   none, norm_test, norm_alpha, norm_ema, tent, cotta
#
# Usage examples:
#   bash scripts/run_common_tta_sourceonly_ckpts.sh
#   METHODS="none tent cotta" bash scripts/run_common_tta_sourceonly_ckpts.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

METHODS=${METHODS:-"none norm_test norm_alpha norm_ema tent cotta"}
NUM_WORKERS=${NUM_WORKERS:-4}
GPU_IDS=${GPU_IDS:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

BN_ALPHA=${BN_ALPHA:-0.1}

TENT_LR=${TENT_LR:-1e-4}
TENT_STEPS=${TENT_STEPS:-1}

COTTA_LR=${COTTA_LR:-1e-4}
COTTA_STEPS=${COTTA_STEPS:-1}
COTTA_MT=${COTTA_MT:-0.999}
COTTA_RST=${COTTA_RST:-0.01}
COTTA_AP=${COTTA_AP:-0.9}

run_tta() {
  local method=$1
  local expname=$2
  local data_name=$3
  local nclass=$4
  local tr_domain=$5
  local ckpt=$6

  echo "=========================================="
  echo "${method}: ${data_name}, source=${tr_domain}, ckpt=${ckpt}"
  echo "=========================================="

  "${PYTHON_BIN}" train.py \
    --phase test \
    --expname "${method}_${expname}" \
    --data_name "${data_name}" \
    --nclass "${nclass}" \
    --tr_domain "${tr_domain}" \
    --resume_path "${ckpt}" \
    --gpu_ids "${GPU_IDS}" \
    --num_workers "${NUM_WORKERS}" \
    --tta "${method}" \
    --bn_alpha "${BN_ALPHA}" \
    --tent_lr "${TENT_LR}" \
    --tent_steps "${TENT_STEPS}" \
    --cotta_lr "${COTTA_LR}" \
    --cotta_steps "${COTTA_STEPS}" \
    --cotta_mt "${COTTA_MT}" \
    --cotta_rst "${COTTA_RST}" \
    --cotta_ap "${COTTA_AP}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0

  echo ""
}

for method in ${METHODS}; do
  run_tta "${method}" "dcon_sc_sabsct" "ABDOMINAL" 5 "SABSCT" "../ckpts/dcon-sc-300.pth"
  run_tta "${method}" "dcon_cs_chaost2" "ABDOMINAL" 5 "CHAOST2" "../ckpts/dcon-cs-200.pth"
  run_tta "${method}" "dcon_bl_bssfp" "CARDIAC" 4 "bSSFP" "../ckpts/dcon-bl-1200.pth"
  run_tta "${method}" "dcon_lb_lge" "CARDIAC" 4 "LGE" "../ckpts/dcon-lb-500.pth"
done

echo "=========================================="
echo "Common DCON TTA baseline evaluations completed!"
echo "=========================================="
