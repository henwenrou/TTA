#!/bin/bash
# Run SM-PPM on DCON source-only checkpoints.
#
# SM-PPM is source-dependent TTA: each target batch uses target feature
# prototypes only, while a labeled source-domain training batch provides the
# supervised adaptation loss.

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

SMPPM_LR=${SMPPM_LR:-2.5e-4}
SMPPM_MOMENTUM=${SMPPM_MOMENTUM:-0.9}
SMPPM_WD=${SMPPM_WD:-5e-4}
SMPPM_STEPS=${SMPPM_STEPS:-1}
SMPPM_SRC_BATCH_SIZE=${SMPPM_SRC_BATCH_SIZE:-2}
SMPPM_PATCH_SIZE=${SMPPM_PATCH_SIZE:-8}
SMPPM_FEATURE_SIZE=${SMPPM_FEATURE_SIZE:-32}
SMPPM_EPISODIC=${SMPPM_EPISODIC:-false}

run_smppm() {
  local expname=$1
  local data_name=$2
  local nclass=$3
  local source=$4
  local target=$5
  local ckpt=$6

  echo "=========================================="
  echo "SM-PPM: ${data_name}, source=${source}, target=${target}, ckpt=${ckpt}"
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
    --tta sm_ppm \
    --smppm_lr "${SMPPM_LR}" \
    --smppm_momentum "${SMPPM_MOMENTUM}" \
    --smppm_wd "${SMPPM_WD}" \
    --smppm_steps "${SMPPM_STEPS}" \
    --smppm_src_batch_size "${SMPPM_SRC_BATCH_SIZE}" \
    --smppm_patch_size "${SMPPM_PATCH_SIZE}" \
    --smppm_feature_size "${SMPPM_FEATURE_SIZE}" \
    --smppm_episodic "${SMPPM_EPISODIC}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0

  echo ""
}

run_smppm "smppm_dcon_sc_chaost2" "ABDOMINAL" 5 "SABSCT" "CHAOST2" "../ckpts/dcon-sc-300.pth"
run_smppm "smppm_dcon_cs_sabsct" "ABDOMINAL" 5 "CHAOST2" "SABSCT" "../ckpts/dcon-cs-200.pth"
run_smppm "smppm_dcon_bl_lge" "CARDIAC" 4 "bSSFP" "LGE" "../ckpts/dcon-bl-1200.pth"
run_smppm "smppm_dcon_lb_bssfp" "CARDIAC" 4 "LGE" "bSSFP" "../ckpts/dcon-lb-500.pth"

echo "=========================================="
echo "SM-PPM DCON TTA evaluations completed!"
echo "=========================================="
