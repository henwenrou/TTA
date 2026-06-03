#!/usr/bin/env bash
# Run DCON/SAA backbone ablations on the four source-target shifts.
#
# Examples:
#   bash scripts/run_backbone_ablation.sh
#   BACKBONES="nnunet swinunet" EPOCHS_OVERRIDE=20 DRY_RUN=1 bash scripts/run_backbone_ablation.sh
#   BACKBONES="nnunet swinunet" ONLY_TASKS="bl lb" CARDIAC_BL_EPOCHS=500 CARDIAC_LB_EPOCHS=900 bash scripts/run_backbone_ablation.sh
#   PYTHON_BIN=/path/to/python SAA_DATA_ROOT=/path/to/data bash scripts/run_backbone_ablation.sh

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

RESULTS_ROOT="${RESULTS_ROOT:-results_backbone_ablation}"
BACKBONES="${BACKBONES:-unet nnunet swinunet}"
ONLY_TASKS="${ONLY_TASKS:-all}"
GPU_IDS="${GPU_IDS:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-20}"
LR="${LR:-0.0005}"
SEED="${SEED:-42}"
VALIDATION_FREQ="${VALIDATION_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-100}"
SAVE_PREDICTION="${SAVE_PREDICTION:-True}"
EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-True}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

USE_SGF="${USE_SGF:-1}"
USE_CGSD="${USE_CGSD:-1}"
USE_PROJECTOR="${USE_PROJECTOR:-1}"
USE_SAAM="${USE_SAAM:-1}"
USE_RCCS="${USE_RCCS:-0}"

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
  local data_name="$2"
  local nclass="$3"
  local source="$4"
  local target="$5"
  local epochs="$6"
  local sgf_grid="$7"
  local display_freq="$8"

  local run_name="${backbone}_${source}_to_${target}"
  local run_dir="${PROJECT_DIR}/${RESULTS_ROOT}/${run_name}"
  local work_dir="${run_dir}/_work"
  local expname="dcon_saa"
  local ckpt_work="${work_dir}/ckpts"

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
    --batchSize "${BATCH_SIZE}"
    --all_epoch "${epochs}"
    --validation_freq "${VALIDATION_FREQ}"
    --display_freq "${display_freq}"
    --save_freq "${SAVE_FREQ}"
    --data_name "${data_name}"
    --nclass "${nclass}"
    --tr_domain "${source}"
    --target_domain "${target}"
    --num_workers "${NUM_WORKERS}"
    --save_prediction "${SAVE_PREDICTION}"
    --eval_source_domain "${EVAL_SOURCE_DOMAIN}"
    --w_ce 1.0
    --w_dice 1.0
    --w_seg 1.0
    --use_sgf "${USE_SGF}"
    --sgf_grid_size "${sgf_grid}"
    --use_cgsd "${USE_CGSD}"
    --cgsd_layer 1
    --use_projector "${USE_PROJECTOR}"
    --use_separate_cgsd_optimizer 1
    --lambda_str 0.3
    --lambda_sty 0.3
    --use_saam "${USE_SAAM}"
    --saam_tau 0.5
    --saam_topk 0.3
    --saam_stability_mode mean
    --lambda_01 1.0
    --lambda_02 1.0
    --saam_warmup_epochs 50
    --saam_rampup_epochs 100
    --anchor_seg_alpha 0.0
    --strong_seg_alpha 1.0
    --use_rccs "${USE_RCCS}"
    --p_rccs 0.3
    --rccs_candidates 4
    --rccs_metric cos
    --rccs_embed_dim 128
  )

  echo "=========================================="
  echo "Backbone ablation: ${data_name} ${source}->${target}, backbone=${backbone}"
  echo "Output: ${run_dir}"
  echo "=========================================="
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
    local produced_dir="${ckpt_work}/${source}/${expname}"
    if [ ! -d "${produced_dir}" ]; then
      echo "Expected run output not found: ${produced_dir}" >&2
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
  ab_epochs="${EPOCHS_OVERRIDE:-1500}"
  cardiac_default_epochs="${EPOCHS_OVERRIDE:-1800}"
  card_bl_epochs="${CARDIAC_BL_EPOCHS:-${cardiac_default_epochs}}"
  card_lb_epochs="${CARDIAC_LB_EPOCHS:-${cardiac_default_epochs}}"

  if should_run_task "sc"; then
    run_one "${backbone}" "ABDOMINAL" 5 "SABSCT" "CHAOST2" "${ab_epochs}" 3 2000
  fi
  if should_run_task "cs"; then
    run_one "${backbone}" "ABDOMINAL" 5 "CHAOST2" "SABSCT" "${ab_epochs}" 3 2000
  fi
  if should_run_task "bl"; then
    run_one "${backbone}" "CARDIAC" 4 "bSSFP" "LGE" "${card_bl_epochs}" 18 5000
  fi
  if should_run_task "lb"; then
    run_one "${backbone}" "CARDIAC" 4 "LGE" "bSSFP" "${card_lb_epochs}" 18 5000
  fi
done

echo "Backbone ablation runs completed."
