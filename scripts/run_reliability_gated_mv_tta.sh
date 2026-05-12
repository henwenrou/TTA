#!/usr/bin/env bash

set -euo pipefail

OUT_DIR="outputs/reliability_tta"
DIRECTION=""
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"

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

"${PYTHON_BIN}" -m tta.reliability_gated_mv_tta "$@"

SUMMARY_ARGS=(--out_dir "${OUT_DIR}")
if [[ -n "${DIRECTION}" ]]; then
  SUMMARY_ARGS+=(--direction "${DIRECTION}")
fi
"${PYTHON_BIN}" scripts/summarize_reliability_tta.py "${SUMMARY_ARGS[@]}"
