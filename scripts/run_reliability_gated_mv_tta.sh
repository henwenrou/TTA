#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="outputs/reliability_tta"
DIRECTION=""
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=("$@")
idx=0
while [[ ${idx} -lt $# ]]; do
  arg="${ARGS[$idx]}"
  case "${arg}" in
    --out_dir)
      idx=$((idx + 1))
      OUT_DIR="${ARGS[$idx]}"
      ;;
    --direction)
      idx=$((idx + 1))
      DIRECTION="${ARGS[$idx]}"
      ;;
  esac
  idx=$((idx + 1))
done

resolve_ckpt() {
  local name="$1"
  local ckpt_root="${CKPT_ROOT:-}"
  local candidates=()

  if [[ -n "${ckpt_root}" ]]; then
    candidates+=("${ckpt_root}/${name}")
  fi
  candidates+=(
    "${REPO_ROOT}/ckpts/${name}"
    "${REPO_ROOT}/DCON/ckpts/${name}"
    "${REPO_ROOT}/${name}"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  echo "Cannot find checkpoint ${name}." >&2
  echo "Checked CKPT_ROOT, ${REPO_ROOT}/ckpts, ${REPO_ROOT}/DCON/ckpts, and repo root." >&2
  return 1
}

run_one() {
  local dataset="$1"
  local direction="$2"
  local ckpt_name="$3"
  local ckpt
  ckpt="$(resolve_ckpt "${ckpt_name}")"

  echo "=========================================="
  echo "Reliability-gated MV-TTA: ${dataset} ${direction}"
  echo "Checkpoint: ${ckpt}"
  echo "=========================================="

  "${PYTHON_BIN}" -m tta.reliability_gated_mv_tta \
    --dataset "${dataset}" \
    --direction "${direction}" \
    --ckpt "${ckpt}" \
    --out_dir "${OUT_DIR}"
}

if [[ $# -gt 0 ]]; then
  "${PYTHON_BIN}" -m tta.reliability_gated_mv_tta "$@"

  SUMMARY_ARGS=(--out_dir "${OUT_DIR}")
  if [[ -n "${DIRECTION}" ]]; then
    SUMMARY_ARGS+=(--direction "${DIRECTION}")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_reliability_tta.py" "${SUMMARY_ARGS[@]}"
else
  run_one "cardiac" "bSSFP_to_LGE" "dcon-bl-1200.pth"
  run_one "cardiac" "LGE_to_bSSFP" "dcon-lb-500.pth"
  run_one "abdominal" "CHAOST2_to_SABSCT" "dcon-cs-200.pth"
  run_one "abdominal" "SABSCT_to_CHAOST2" "dcon-sc-300.pth"

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_reliability_tta.py" --out_dir "${OUT_DIR}"
fi
