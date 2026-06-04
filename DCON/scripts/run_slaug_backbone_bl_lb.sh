#!/usr/bin/env bash
# Run SLAug-only baseline on CARDIAC BL/LB for backbone comparison.
#
# This keeps the DCON data split/evaluation protocol but disables SAAM/CGSD/RCCS/SGF
# so the run can be used as a same-backbone baseline against SAA/SAAM runs.
#
# Examples:
#   BACKBONES="nnunet swinunet" CARDIAC_BL_EPOCHS=500 CARDIAC_LB_EPOCHS=900 bash scripts/run_slaug_backbone_bl_lb.sh
#   BACKBONES="swinunet" SWIN_BATCH_SIZE=8 ONLY_TASKS="bl" DRY_RUN=1 bash scripts/run_slaug_backbone_bl_lb.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "No python executable found. Set PYTHON_BIN=/path/to/env/bin/python." >&2
    exit 1
  fi
fi

if [ -z "${SAA_DATA_ROOT:-}" ]; then
  if [ -d "/Users/RexRyder/PycharmProjects/Dataset" ]; then
    export SAA_DATA_ROOT="/Users/RexRyder/PycharmProjects/Dataset"
  else
    export SAA_DATA_ROOT="${PROJECT_DIR}/data"
  fi
fi

RESULTS_ROOT="${RESULTS_ROOT:-results_slaug_backbone_bl_lb}"
BACKBONES="${BACKBONES:-nnunet swinunet}"
ONLY_TASKS="${ONLY_TASKS:-bl lb}"
GPU_IDS="${GPU_IDS:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-20}"
SWIN_BATCH_SIZE="${SWIN_BATCH_SIZE:-${BATCH_SIZE}}"
LR="${LR:-0.0005}"
SEED="${SEED:-42}"
VALIDATION_FREQ="${VALIDATION_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-100}"
SAVE_PREDICTION="${SAVE_PREDICTION:-False}"
EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-False}"
LOCAL_AUG_TYPE="${LOCAL_AUG_TYPE:-lla}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

if [ ! -d "${SAA_DATA_ROOT}" ]; then
  echo "SAA_DATA_ROOT does not exist: ${SAA_DATA_ROOT}" >&2
  exit 1
fi

should_run_task() {
  local task="$1"
  if [ "${ONLY_TASKS}" = "all" ]; then
    return 0
  fi
  local item
  for item in ${ONLY_TASKS}; do
    if [ "${item}" = "${task}" ]; then
      return 0
    fi
  done
  return 1
}

run_one() {
  local backbone="$1"
  local source="$2"
  local target="$3"
  local epochs="$4"

  local run_name="${backbone}_slaug_${source}_to_${target}"
  local run_dir="${PROJECT_DIR}/${RESULTS_ROOT}/${run_name}"
  local work_dir="${run_dir}/_work"
  local expname="slaug"
  local ckpt_work="${work_dir}/ckpts"
  local run_log="${run_dir}/train_stdout.log"
  local effective_batch_size="${BATCH_SIZE}"

  if [ "${backbone}" = "swinunet" ]; then
    effective_batch_size="${SWIN_BATCH_SIZE}"
  fi

  if [ "${FORCE}" != "1" ] && [ -f "${run_dir}/log/out.csv" ]; then
    echo "Skipping existing run: ${run_dir}"
    return
  fi

  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" train.py
    --phase train
    --expname "${expname}"
    --ckpt_dir "${ckpt_work}"
    --gpu_ids "${GPU_IDS}"
    --f_seed "${SEED}"
    --lr "${LR}"
    --model unet
    --backbone "${backbone}"
    --batchSize "${effective_batch_size}"
    --all_epoch "${epochs}"
    --validation_freq "${VALIDATION_FREQ}"
    --display_freq 5000
    --save_freq "${SAVE_FREQ}"
    --data_name CARDIAC
    --nclass 4
    --tr_domain "${source}"
    --target_domain "${target}"
    --num_workers "${NUM_WORKERS}"
    --save_prediction "${SAVE_PREDICTION}"
    --eval_source_domain "${EVAL_SOURCE_DOMAIN}"
    --local_aug_type "${LOCAL_AUG_TYPE}"
    --w_ce 1.0
    --w_dice 1.0
    --w_seg 1.0
    --seg_alpha_view2 1.0
    --use_sgf 0
    --use_cgsd 0
    --use_projector 0
    --use_saam 0
    --use_rccs 0
  )

  echo "=========================================="
  echo "SLAug backbone baseline: CARDIAC ${source}->${target}, backbone=${backbone}"
  echo "Output: ${run_dir}"
  echo "Epochs: ${epochs}; batch_size=${effective_batch_size}; local_aug_type=${LOCAL_AUG_TYPE}"
  echo "=========================================="
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${cmd[@]}"
    echo
  else
    "${cmd[@]}" 2>&1 | tee "${run_log}"
    local produced_dir="${ckpt_work}/${source}/${expname}"
    if [ ! -d "${produced_dir}" ]; then
      echo "Expected run output not found: ${produced_dir}" >&2
      echo "Training stdout log: ${run_log}" >&2
      exit 1
    fi
    cp -R "${produced_dir}/." "${run_dir}/"
  fi
  echo
}

echo "Project: ${PROJECT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "SAA_DATA_ROOT: ${SAA_DATA_ROOT}"
echo "Results root: ${PROJECT_DIR}/${RESULTS_ROOT}"
echo "Backbones: ${BACKBONES}"
echo "Tasks: ${ONLY_TASKS}"
echo

for backbone in ${BACKBONES}; do
  cardiac_default_epochs="${EPOCHS_OVERRIDE:-1800}"
  card_bl_epochs="${CARDIAC_BL_EPOCHS:-${cardiac_default_epochs}}"
  card_lb_epochs="${CARDIAC_LB_EPOCHS:-${cardiac_default_epochs}}"

  if should_run_task "bl"; then
    run_one "${backbone}" "bSSFP" "LGE" "${card_bl_epochs}"
  fi
  if should_run_task "lb"; then
    run_one "${backbone}" "LGE" "bSSFP" "${card_lb_epochs}"
  fi
done

echo "SLAug backbone baseline runs completed."
