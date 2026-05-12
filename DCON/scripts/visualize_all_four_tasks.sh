#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-results_vis}"
METHOD="${METHOD:-none}"
TILE_SIZE="${TILE_SIZE:-224}"
ALPHA="${ALPHA:-0.45}"
WORST_K="${WORST_K:-24}"
SKIP_ALL_EMPTY_IN_GLOBAL="${SKIP_ALL_EMPTY_IN_GLOBAL:-1}"

TASKS=(
  "SABSCT none_dcon_sabsct_to_chaost2 SABSCT_to_CHAOST2"
  "CHAOST2 none_dcon_chaost2_to_sabsct CHAOST2_to_SABSCT"
  "bSSFP none_dcon_bssfp_to_lge bSSFP_to_LGE"
  "LGE none_dcon_lge_to_bssfp LGE_to_bSSFP"
)

if [ "${METHOD}" != "none" ]; then
  TASKS=(
    "SABSCT ${METHOD}_dcon_sabsct_to_chaost2 SABSCT_to_CHAOST2"
    "CHAOST2 ${METHOD}_dcon_chaost2_to_sabsct CHAOST2_to_SABSCT"
    "bSSFP ${METHOD}_dcon_bssfp_to_lge bSSFP_to_LGE"
    "LGE ${METHOD}_dcon_lge_to_bssfp LGE_to_bSSFP"
  )
fi

echo "Project: ${PROJECT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Results: ${RESULTS_DIR}"
echo "Method: ${METHOD}"

for item in "${TASKS[@]}"; do
  read -r source expname tag <<< "${item}"
  volume_dir="${RESULTS_DIR}/${source}/${expname}/log/volumes"
  output_dir="${RESULTS_DIR}/visualization/${tag}"

  if [ ! -d "${volume_dir}" ]; then
    echo "Missing volume dir: ${volume_dir}" >&2
    echo "Run test inference first with SAVE_PREDICTION=true." >&2
    exit 1
  fi

  cmd=(
    "${PYTHON_BIN}" scripts/visualize_volume_predictions.py
    --volume-dir "${volume_dir}"
    --output-dir "${output_dir}"
    --tile-size "${TILE_SIZE}"
    --alpha "${ALPHA}"
    --worst-k "${WORST_K}"
  )

  if [ "${SKIP_ALL_EMPTY_IN_GLOBAL}" = "1" ]; then
    cmd+=(--skip-all-empty-in-global)
  fi

  echo "=========================================="
  echo "Visualizing: ${tag}"
  echo "Input: ${volume_dir}"
  echo "Output: ${output_dir}"
  echo "=========================================="
  "${cmd[@]}"
done

echo
echo "All 4 tasks finished."
echo "Outputs are under: ${RESULTS_DIR}/visualization"
