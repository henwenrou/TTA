#!/usr/bin/env bash
# Test backbone ablation checkpoints with the matching backbone.
#
# Examples:
#   BACKBONES="nnunet swinunet" ONLY_TASKS="bl lb" bash scripts/test_backbone_ablation_ckpts.sh
#   BACKBONES="swinunet" CKPT_PATTERN="best*_net_Seg.pth" bash scripts/test_backbone_ablation_ckpts.sh

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

RESULTS_ROOT="${RESULTS_ROOT:-results_backbone_ablation}"
TEST_RESULTS_ROOT="${TEST_RESULTS_ROOT:-results_backbone_ablation_test}"
BACKBONES="${BACKBONES:-nnunet swinunet}"
ONLY_TASKS="${ONLY_TASKS:-bl lb}"
CKPT_PATTERN="${CKPT_PATTERN:-*00_net_Seg.pth}"
GPU_IDS="${GPU_IDS:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_PREDICTION="${SAVE_PREDICTION:-False}"
EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-True}"
DRY_RUN="${DRY_RUN:-0}"

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

test_task() {
  local backbone="$1"
  local task="$2"
  local source="$3"
  local target="$4"

  local run_name="${backbone}_${source}_to_${target}"
  local snapshot_dir="${PROJECT_DIR}/${RESULTS_ROOT}/${run_name}/snapshots"
  local test_root="${PROJECT_DIR}/${TEST_RESULTS_ROOT}/${run_name}"

  if [ ! -d "${snapshot_dir}" ]; then
    echo "Missing snapshot directory: ${snapshot_dir}" >&2
    return
  fi

  shopt -s nullglob
  local ckpts=("${snapshot_dir}"/${CKPT_PATTERN})
  shopt -u nullglob
  if [ "${#ckpts[@]}" -eq 0 ]; then
    echo "No checkpoints matched ${snapshot_dir}/${CKPT_PATTERN}" >&2
    return
  fi

  local ckpt
  for ckpt in "${ckpts[@]}"; do
    local ckpt_base
    ckpt_base="$(basename "${ckpt}" .pth)"
    local expname="${ckpt_base}"
    local log_dir="${test_root}/logs"
    mkdir -p "${log_dir}"

    local cmd=(
      "${PYTHON_BIN}" train.py
      --phase test
      --ckpt_dir "${test_root}/ckpts"
      --expname "${expname}"
      --data_name CARDIAC
      --tr_domain "${source}"
      --target_domain "${target}"
      --nclass 4
      --restore_from "${ckpt}"
      --backbone "${backbone}"
      --gpu_ids "${GPU_IDS}"
      --num_workers "${NUM_WORKERS}"
      --save_prediction "${SAVE_PREDICTION}"
      --eval_source_domain "${EVAL_SOURCE_DOMAIN}"
      --use_cgsd 0
      --use_projector 0
      --use_saam 0
      --use_rccs 0
    )

    echo "=========================================="
    echo "Testing ${task}: ${source}->${target}, backbone=${backbone}"
    echo "Checkpoint: ${ckpt}"
    echo "Output root: ${test_root}"
    echo "=========================================="
    if [ "${DRY_RUN}" = "1" ]; then
      printf '%q ' "${cmd[@]}"
      echo
    else
      "${cmd[@]}" 2>&1 | tee "${log_dir}/${ckpt_base}.log"
    fi
    echo
  done
}

for backbone in ${BACKBONES}; do
  if should_run_task "bl"; then
    test_task "${backbone}" "bl" "bSSFP" "LGE"
  fi
  if should_run_task "lb"; then
    test_task "${backbone}" "lb" "LGE" "bSSFP"
  fi
done

echo "Backbone checkpoint testing completed."
