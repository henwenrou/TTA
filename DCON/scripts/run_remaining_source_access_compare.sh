#!/usr/bin/env bash
# Resume the source-access comparison by running only missing/incomplete DCON jobs.
#
# Intended after the first full command was interrupted:
#   METHODS="tent tent_source_ce sar sar_source_ce cotta cotta_source_ce source_ce_only sm_ppm" \
#   RESULTS_DIR=results_source_access_compare \
#   bash scripts/run_medseg_tta_dcon.sh
#
# This script checks each expected out.csv for the final target Dice line and
# skips jobs that already completed.

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

RESULTS_DIR="${RESULTS_DIR:-results_source_access_compare}"
CKPT_ROOT="${CKPT_ROOT:-../ckpts}"
GPU_IDS="${GPU_IDS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_PREDICTION="${SAVE_PREDICTION:-false}"
EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-false}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# These are the methods left from the 8 x 4 source-access comparison.
REMAINING_METHODS="${REMAINING_METHODS:-cotta_source_ce source_ce_only sm_ppm}"

LAMBDA_SOURCE="${LAMBDA_SOURCE:-1.0}"

COTTA_LR="${COTTA_LR:-1e-4}"
COTTA_STEPS="${COTTA_STEPS:-1}"
COTTA_MT="${COTTA_MT:-0.999}"
COTTA_RST="${COTTA_RST:-0.01}"
COTTA_AP="${COTTA_AP:-0.9}"

SMPPM_LR="${SMPPM_LR:-2.5e-4}"
SMPPM_MOMENTUM="${SMPPM_MOMENTUM:-0.9}"
SMPPM_WD="${SMPPM_WD:-5e-4}"
SMPPM_STEPS="${SMPPM_STEPS:-1}"
SMPPM_SRC_BATCH_SIZE="${SMPPM_SRC_BATCH_SIZE:-2}"
SMPPM_PATCH_SIZE="${SMPPM_PATCH_SIZE:-8}"
SMPPM_FEATURE_SIZE="${SMPPM_FEATURE_SIZE:-32}"
SMPPM_EPISODIC="${SMPPM_EPISODIC:-false}"
SMPPM_ABLATION_MODE="${SMPPM_ABLATION_MODE:-full}"
SMPPM_SOURCE_FREE_TAU="${SMPPM_SOURCE_FREE_TAU:-0.7}"
SMPPM_SOURCE_FREE_ENTROPY_WEIGHT="${SMPPM_SOURCE_FREE_ENTROPY_WEIGHT:-1.0}"
SMPPM_SOURCE_FREE_LAMBDA_PROTO="${SMPPM_SOURCE_FREE_LAMBDA_PROTO:-1.0}"
SMPPM_PLAIN_SOURCE_LOADER="${SMPPM_PLAIN_SOURCE_LOADER:-true}"
SMPPM_STYLE_ALPHA="${SMPPM_STYLE_ALPHA:-1.0}"
SMPPM_LOG_INTERVAL="${SMPPM_LOG_INTERVAL:-0}"

if [ ! -d "${SAA_DATA_ROOT}" ]; then
  echo "SAA_DATA_ROOT does not exist: ${SAA_DATA_ROOT}" >&2
  exit 1
fi

require_ckpt() {
  local ckpt="$1"
  if [ ! -f "${ckpt}" ]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
}

is_complete() {
  local source="$1"
  local expname="$2"
  local out_csv="${RESULTS_DIR}/${source}/${expname}/log/out.csv"

  [ "${FORCE_RERUN}" = "0" ] \
    && [ -s "${out_csv}" ] \
    && grep -q "Overall mean dice by sample" "${out_csv}"
}

method_args() {
  local method="$1"
  METHOD_TTA="${method}"
  METHOD_EXTRA=()

  case "${method}" in
    cotta_source_ce)
      METHOD_TTA="cotta"
      METHOD_EXTRA=(
        --cotta_lr "${COTTA_LR}"
        --cotta_steps "${COTTA_STEPS}"
        --cotta_mt "${COTTA_MT}"
        --cotta_rst "${COTTA_RST}"
        --cotta_ap "${COTTA_AP}"
        --source_access true
        --lambda_source "${LAMBDA_SOURCE}"
      )
      ;;
    source_ce_only)
      METHOD_EXTRA=(
        --smppm_lr "${SMPPM_LR}"
        --smppm_momentum "${SMPPM_MOMENTUM}"
        --smppm_wd "${SMPPM_WD}"
        --smppm_steps "${SMPPM_STEPS}"
        --smppm_src_batch_size "${SMPPM_SRC_BATCH_SIZE}"
        --smppm_patch_size "${SMPPM_PATCH_SIZE}"
        --smppm_feature_size "${SMPPM_FEATURE_SIZE}"
        --smppm_episodic "${SMPPM_EPISODIC}"
        --smppm_ablation_mode source_ce_only
        --smppm_plain_source_loader "${SMPPM_PLAIN_SOURCE_LOADER}"
        --smppm_log_interval "${SMPPM_LOG_INTERVAL}"
      )
      ;;
    sm_ppm)
      METHOD_EXTRA=(
        --smppm_lr "${SMPPM_LR}"
        --smppm_momentum "${SMPPM_MOMENTUM}"
        --smppm_wd "${SMPPM_WD}"
        --smppm_steps "${SMPPM_STEPS}"
        --smppm_src_batch_size "${SMPPM_SRC_BATCH_SIZE}"
        --smppm_patch_size "${SMPPM_PATCH_SIZE}"
        --smppm_feature_size "${SMPPM_FEATURE_SIZE}"
        --smppm_episodic "${SMPPM_EPISODIC}"
        --smppm_ablation_mode "${SMPPM_ABLATION_MODE}"
        --smppm_source_free_tau "${SMPPM_SOURCE_FREE_TAU}"
        --smppm_source_free_entropy_weight "${SMPPM_SOURCE_FREE_ENTROPY_WEIGHT}"
        --smppm_source_free_lambda_proto "${SMPPM_SOURCE_FREE_LAMBDA_PROTO}"
        --smppm_plain_source_loader "${SMPPM_PLAIN_SOURCE_LOADER}"
        --smppm_style_alpha "${SMPPM_STYLE_ALPHA}"
        --smppm_log_interval "${SMPPM_LOG_INTERVAL}"
      )
      ;;
    *)
      echo "Unknown remaining method: ${method}" >&2
      exit 1
      ;;
  esac
}

run_one() {
  local method="$1"
  local data_name="$2"
  local nclass="$3"
  local source="$4"
  local target="$5"
  local ckpt="$6"
  local suffix="$7"
  local expname="${method}_dcon_${suffix}"

  if is_complete "${source}" "${expname}"; then
    echo "[skip] ${method}: ${source}->${target} already has final Dice in ${RESULTS_DIR}/${source}/${expname}/log/out.csv"
    return
  fi

  require_ckpt "${ckpt}"
  method_args "${method}"

  local cmd=(
    "${PYTHON_BIN}" train.py
    --phase test
    --expname "${expname}"
    --ckpt_dir "${RESULTS_DIR}"
    --dataset "${data_name}"
    --nclass "${nclass}"
    --source "${source}"
    --target "${target}"
    --restore_from "${ckpt}"
    --gpu_ids "${GPU_IDS}"
    --num_workers "${NUM_WORKERS}"
    --save_prediction "${SAVE_PREDICTION}"
    --eval_source_domain "${EVAL_SOURCE_DOMAIN}"
    --tta "${METHOD_TTA}"
    --use_cgsd 0
    --use_projector 0
    --use_saam 0
    --use_rccs 0
    "${METHOD_EXTRA[@]}"
  )

  echo "=========================================="
  echo "[run] ${method}: ${data_name} ${source}->${target}"
  echo "checkpoint: ${ckpt}"
  echo "expname: ${expname}"
  echo "=========================================="

  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
  fi
  echo
}

echo "Project: ${PROJECT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "SAA_DATA_ROOT: ${SAA_DATA_ROOT}"
echo "Results: ${RESULTS_DIR}"
echo "Remaining methods: ${REMAINING_METHODS}"
echo "Force rerun: ${FORCE_RERUN}"
echo

for method in ${REMAINING_METHODS}; do
  run_one "${method}" "ABDOMINAL" 5 "SABSCT" "CHAOST2" "${CKPT_ROOT}/dcon-sc-300.pth" "sabsct_to_chaost2"
  run_one "${method}" "ABDOMINAL" 5 "CHAOST2" "SABSCT" "${CKPT_ROOT}/dcon-cs-200.pth" "chaost2_to_sabsct"
  run_one "${method}" "CARDIAC" 4 "bSSFP" "LGE" "${CKPT_ROOT}/dcon-bl-1200.pth" "bssfp_to_lge"
  run_one "${method}" "CARDIAC" 4 "LGE" "bSSFP" "${CKPT_ROOT}/dcon-lb-500.pth" "lge_to_bssfp"
done

if [ "${DRY_RUN}" != "1" ]; then
  echo "=========================================="
  echo "Resume run completed. Summary:"
  echo "=========================================="
  "${PYTHON_BIN}" scripts/summarize_medseg_tta_dcon.py \
    --roots "${RESULTS_DIR}" \
    --methods tent tent_source_ce sar sar_source_ce cotta cotta_source_ce source_ce_only sm_ppm || true
fi
