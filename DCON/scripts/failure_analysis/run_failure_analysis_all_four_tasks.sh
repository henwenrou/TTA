#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "Cannot find python or python3. Set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-results_vis}"
METHOD="${METHOD:-none}"
OUT_ROOT="${OUT_ROOT:-${RESULTS_DIR}/failure_analysis/${METHOD}}"
TOP_K="${TOP_K:-20}"
LOW_DICE_THRESHOLD="${LOW_DICE_THRESHOLD:-0.50}"
ONLY_ANALYZE="${ONLY_ANALYZE:-0}"

MIN_PRED_AREA="${MIN_PRED_AREA:-10}"
MIN_GT_AREA="${MIN_GT_AREA:-10}"
LOW_DICE="${LOW_DICE:-0.30}"
OVER_AREA_RATIO="${OVER_AREA_RATIO:-1.50}"
UNDER_AREA_RATIO="${UNDER_AREA_RATIO:-0.50}"
MAX_COMPONENTS="${MAX_COMPONENTS:-3}"
MIN_LARGEST_COMPONENT_RATIO="${MIN_LARGEST_COMPONENT_RATIO:-0.75}"
SMALL_ORGAN_DICE="${SMALL_ORGAN_DICE:-0.60}"
TOPOLOGY_MAX_COMPONENTS="${TOPOLOGY_MAX_COMPONENTS:-3}"
TOPOLOGY_MIN_LARGEST_COMPONENT_RATIO="${TOPOLOGY_MIN_LARGEST_COMPONENT_RATIO:-0.70}"
FOREGROUND_MAX_COMPONENTS="${FOREGROUND_MAX_COMPONENTS:-10}"

TASKS=(
  "SABSCT none_dcon_sabsct_to_chaost2 SABSCT_to_CHAOST2 abdominal configs/abdominal_classes.json"
  "CHAOST2 none_dcon_chaost2_to_sabsct CHAOST2_to_SABSCT abdominal configs/abdominal_classes.json"
  "bSSFP none_dcon_bssfp_to_lge bSSFP_to_LGE cardiac configs/cardiac_classes.json"
  "LGE none_dcon_lge_to_bssfp LGE_to_bSSFP cardiac configs/cardiac_classes.json"
)

if [ "${METHOD}" != "none" ]; then
  TASKS=(
    "SABSCT ${METHOD}_dcon_sabsct_to_chaost2 SABSCT_to_CHAOST2 abdominal configs/abdominal_classes.json"
    "CHAOST2 ${METHOD}_dcon_chaost2_to_sabsct CHAOST2_to_SABSCT abdominal configs/abdominal_classes.json"
    "bSSFP ${METHOD}_dcon_bssfp_to_lge bSSFP_to_LGE cardiac configs/cardiac_classes.json"
    "LGE ${METHOD}_dcon_lge_to_bssfp LGE_to_bSSFP cardiac configs/cardiac_classes.json"
  )
fi

mkdir -p "${OUT_ROOT}/csv" "${OUT_ROOT}/top_cases"

echo "Project: ${PROJECT_DIR}"
echo "Python: ${PYTHON}"
echo "Results: ${RESULTS_DIR}"
echo "Method: ${METHOD}"
echo "Output: ${OUT_ROOT}"

CSV_FILES=()

for item in "${TASKS[@]}"; do
  read -r source expname direction dataset class_map <<< "${item}"
  volume_dir="${RESULTS_DIR}/${source}/${expname}/log/volumes"
  out_csv="${OUT_ROOT}/csv/${direction}_failures.csv"

  if [ ! -d "${volume_dir}" ]; then
    echo "Missing volume dir: ${volume_dir}" >&2
    echo "Run inference first with SAVE_PREDICTION=true, then run visualization/failure analysis." >&2
    exit 1
  fi

  echo "=========================================="
  echo "Analyzing: ${direction}"
  echo "Input: ${volume_dir}"
  echo "CSV: ${out_csv}"
  echo "=========================================="

  "${PYTHON}" scripts/failure_analysis/failure_analyzer.py \
    --volume_dir "${volume_dir}" \
    --dataset "${dataset}" \
    --direction "${direction}" \
    --class_map "${class_map}" \
    --out_csv "${out_csv}" \
    --min_pred_area "${MIN_PRED_AREA}" \
    --min_gt_area "${MIN_GT_AREA}" \
    --low_dice "${LOW_DICE}" \
    --over_area_ratio "${OVER_AREA_RATIO}" \
    --under_area_ratio "${UNDER_AREA_RATIO}" \
    --max_components "${MAX_COMPONENTS}" \
    --min_largest_component_ratio "${MIN_LARGEST_COMPONENT_RATIO}" \
    --small_organ_dice "${SMALL_ORGAN_DICE}" \
    --topology_max_components "${TOPOLOGY_MAX_COMPONENTS}" \
    --topology_min_largest_component_ratio "${TOPOLOGY_MIN_LARGEST_COMPONENT_RATIO}" \
    --foreground_max_components "${FOREGROUND_MAX_COMPONENTS}"

  CSV_FILES+=("${out_csv}")

  if [ "${ONLY_ANALYZE}" != "1" ]; then
    "${PYTHON}" scripts/failure_analysis/visualize_failure_cases.py \
      --failure_csv "${out_csv}" \
      --out_dir "${OUT_ROOT}/top_cases/${direction}" \
      --top_k "${TOP_K}" \
      --only_failures
  fi
done

if [ "${ONLY_ANALYZE}" != "1" ]; then
  echo "=========================================="
  echo "Writing summary reports"
  echo "=========================================="

  "${PYTHON}" scripts/failure_analysis/summarize_failures.py \
    --csv_files "${CSV_FILES[@]}" \
    --out_md "${OUT_ROOT}/failure_report.md" \
    --out_csv "${OUT_ROOT}/top_failure_rows.csv" \
    --low_dice_threshold "${LOW_DICE_THRESHOLD}" \
    --top_k "${TOP_K}"

  "${PYTHON}" scripts/failure_analysis/recommend_constraints.py \
    --csv_files "${CSV_FILES[@]}" \
    --out_md "${OUT_ROOT}/failure_recommendations.md" \
    --out_json "${OUT_ROOT}/failure_recommendations.json" \
    --low_dice_threshold "${LOW_DICE_THRESHOLD}"
fi

echo
echo "All 4 failure-analysis tasks finished."
echo "Report: ${OUT_ROOT}/failure_report.md"
echo "Recommendations: ${OUT_ROOT}/failure_recommendations.md"
echo "Top-case PNGs: ${OUT_ROOT}/top_cases"
