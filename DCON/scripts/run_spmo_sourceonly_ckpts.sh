#!/usr/bin/env bash
# Run the DCON-adapted SPMO-TTA method on the four bundled source-only DCON checkpoints.
#
# Examples:
#   bash scripts/run_spmo_sourceonly_ckpts.sh
#   SPMO_STEPS=2 SPMO_MOMENT_MODE=centroid bash scripts/run_spmo_sourceonly_ckpts.sh
#   PYTHON_BIN=/path/to/env/bin/python SAA_DATA_ROOT=/path/to/data bash scripts/run_spmo_sourceonly_ckpts.sh

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

RESULTS_DIR="${RESULTS_DIR:-results_spmo_dcon}"
CKPT_ROOT="${CKPT_ROOT:-../ckpts}"
GPU_IDS="${GPU_IDS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_PREDICTION="${SAVE_PREDICTION:-false}"
EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-false}"
DRY_RUN="${DRY_RUN:-0}"

SPMO_LR="${SPMO_LR:-1e-4}"
SPMO_WEIGHT_DECAY="${SPMO_WEIGHT_DECAY:-0.0}"
SPMO_STEPS="${SPMO_STEPS:-1}"
SPMO_ENTROPY_WEIGHT="${SPMO_ENTROPY_WEIGHT:-1.0}"
SPMO_PRIOR_WEIGHT="${SPMO_PRIOR_WEIGHT:-1.0}"
SPMO_MOMENT_WEIGHT="${SPMO_MOMENT_WEIGHT:-0.05}"
SPMO_MOMENT_MODE="${SPMO_MOMENT_MODE:-all}"
SPMO_SOFTMAX_TEMP="${SPMO_SOFTMAX_TEMP:-1.0}"
SPMO_SIZE_POWER="${SPMO_SIZE_POWER:-1.0}"
SPMO_BG_ENTROPY_WEIGHT="${SPMO_BG_ENTROPY_WEIGHT:-0.1}"
SPMO_PRIOR_EPS="${SPMO_PRIOR_EPS:-1e-6}"
SPMO_MIN_PIXELS="${SPMO_MIN_PIXELS:-10}"
SPMO_SOURCE_PSEUDO="${SPMO_SOURCE_PSEUDO:-hard}"
SPMO_UPDATE_SCOPE="${SPMO_UPDATE_SCOPE:-bn_affine}"
SPMO_EPISODIC="${SPMO_EPISODIC:-false}"

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

run_one() {
  local data_name="$1"
  local nclass="$2"
  local source="$3"
  local target="$4"
  local ckpt="$5"
  local suffix="$6"
  local expname="spmo_dcon_${suffix}"

  require_ckpt "${ckpt}"

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
    --tta spmo
    --spmo_lr "${SPMO_LR}"
    --spmo_weight_decay "${SPMO_WEIGHT_DECAY}"
    --spmo_steps "${SPMO_STEPS}"
    --spmo_entropy_weight "${SPMO_ENTROPY_WEIGHT}"
    --spmo_prior_weight "${SPMO_PRIOR_WEIGHT}"
    --spmo_moment_weight "${SPMO_MOMENT_WEIGHT}"
    --spmo_moment_mode "${SPMO_MOMENT_MODE}"
    --spmo_softmax_temp "${SPMO_SOFTMAX_TEMP}"
    --spmo_size_power "${SPMO_SIZE_POWER}"
    --spmo_bg_entropy_weight "${SPMO_BG_ENTROPY_WEIGHT}"
    --spmo_prior_eps "${SPMO_PRIOR_EPS}"
    --spmo_min_pixels "${SPMO_MIN_PIXELS}"
    --spmo_source_pseudo "${SPMO_SOURCE_PSEUDO}"
    --spmo_update_scope "${SPMO_UPDATE_SCOPE}"
    --spmo_episodic "${SPMO_EPISODIC}"
    --use_cgsd 0
    --use_projector 0
    --use_saam 0
    --use_rccs 0
  )

  echo "=========================================="
  echo "SPMO: ${data_name} ${source}->${target}"
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
echo

run_one "ABDOMINAL" 5 "SABSCT" "CHAOST2" "${CKPT_ROOT}/dcon-sc-300.pth" "sabsct_to_chaost2"
run_one "ABDOMINAL" 5 "CHAOST2" "SABSCT" "${CKPT_ROOT}/dcon-cs-200.pth" "chaost2_to_sabsct"
run_one "CARDIAC" 4 "bSSFP" "LGE" "${CKPT_ROOT}/dcon-bl-1200.pth" "bssfp_to_lge"
run_one "CARDIAC" 4 "LGE" "bSSFP" "${CKPT_ROOT}/dcon-lb-500.pth" "lge_to_bssfp"

if [ "${DRY_RUN}" != "1" ]; then
  echo "=========================================="
  echo "SPMO runs completed. Summary:"
  echo "=========================================="
  "${PYTHON_BIN}" scripts/summarize_medseg_tta_dcon.py --roots "${RESULTS_DIR}" --methods spmo || true
fi
