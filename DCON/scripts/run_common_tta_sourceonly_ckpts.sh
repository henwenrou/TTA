#!/bin/bash
# Run the first-batch common TTA baselines on DCON source-only checkpoints.
#
# Default methods:
#   none, norm_test, norm_alpha, norm_ema, tent, cotta
#
# Usage examples:
#   bash scripts/run_common_tta_sourceonly_ckpts.sh
#   METHODS="none tent sar cotta tent_source_ce sar_source_ce cotta_source_ce source_ce_only" bash scripts/run_common_tta_sourceonly_ckpts.sh

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

SAR_LR=${SAR_LR:-1e-4}
SAR_STEPS=${SAR_STEPS:-1}
SAR_RHO=${SAR_RHO:-0.05}

COTTA_LR=${COTTA_LR:-1e-4}
COTTA_STEPS=${COTTA_STEPS:-1}
COTTA_MT=${COTTA_MT:-0.999}
COTTA_RST=${COTTA_RST:-0.01}
COTTA_AP=${COTTA_AP:-0.9}

LAMBDA_SOURCE=${LAMBDA_SOURCE:-1.0}
SMPPM_LR=${SMPPM_LR:-2.5e-4}
SMPPM_MOMENTUM=${SMPPM_MOMENTUM:-0.9}
SMPPM_WD=${SMPPM_WD:-5e-4}
SMPPM_STEPS=${SMPPM_STEPS:-1}
SMPPM_SRC_BATCH_SIZE=${SMPPM_SRC_BATCH_SIZE:-2}
SMPPM_PLAIN_SOURCE_LOADER=${SMPPM_PLAIN_SOURCE_LOADER:-true}

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

  local tta_method="${method}"
  local source_access=false
  local smppm_ablation_mode=full
  case "${method}" in
    tent_source_ce)
      tta_method=tent
      source_access=true
      ;;
    sar_source_ce)
      tta_method=sar
      source_access=true
      ;;
    cotta_source_ce)
      tta_method=cotta
      source_access=true
      ;;
    source_ce_only)
      tta_method=source_ce_only
      smppm_ablation_mode=source_ce_only
      ;;
  esac

  "${PYTHON_BIN}" train.py \
    --phase test \
    --expname "${method}_${expname}" \
    --data_name "${data_name}" \
    --nclass "${nclass}" \
    --tr_domain "${tr_domain}" \
    --resume_path "${ckpt}" \
    --gpu_ids "${GPU_IDS}" \
    --num_workers "${NUM_WORKERS}" \
    --tta "${tta_method}" \
    --source_access "${source_access}" \
    --lambda_source "${LAMBDA_SOURCE}" \
    --bn_alpha "${BN_ALPHA}" \
    --tent_lr "${TENT_LR}" \
    --tent_steps "${TENT_STEPS}" \
    --sar_lr "${SAR_LR}" \
    --sar_steps "${SAR_STEPS}" \
    --sar_rho "${SAR_RHO}" \
    --cotta_lr "${COTTA_LR}" \
    --cotta_steps "${COTTA_STEPS}" \
    --cotta_mt "${COTTA_MT}" \
    --cotta_rst "${COTTA_RST}" \
    --cotta_ap "${COTTA_AP}" \
    --smppm_lr "${SMPPM_LR}" \
    --smppm_momentum "${SMPPM_MOMENTUM}" \
    --smppm_wd "${SMPPM_WD}" \
    --smppm_steps "${SMPPM_STEPS}" \
    --smppm_src_batch_size "${SMPPM_SRC_BATCH_SIZE}" \
    --smppm_ablation_mode "${smppm_ablation_mode}" \
    --smppm_plain_source_loader "${SMPPM_PLAIN_SOURCE_LOADER}" \
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
