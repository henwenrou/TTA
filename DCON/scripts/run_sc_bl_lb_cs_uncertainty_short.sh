#!/usr/bin/env bash
# One-shot short-run script for reviewer uncertainty-weighting checks.
#
# Runs:
#   sc: SABSCT -> CHAOST2, entropy/confidence/fsda_uncertainty, 500 epochs
#   bl: bSSFP -> LGE, fsda_uncertainty, 600 epochs
#   lb: LGE -> bSSFP, fsda_uncertainty, 900 epochs
#   cs: CHAOST2 -> SABSCT, fsda_uncertainty, 800 epochs

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DCON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DCON_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/tpsdg/bin/python}"
export SAA_DATA_ROOT="${SAA_DATA_ROOT:-data}"

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
GPU_IDS="${GPU_IDS:-0}"

echo "Python: ${PYTHON_BIN}"
echo "SAA_DATA_ROOT: ${SAA_DATA_ROOT}"
echo "GPU_IDS: ${GPU_IDS}"

PYTHON_BIN="${PYTHON_BIN}" \
SAA_DATA_ROOT="${SAA_DATA_ROOT}" \
GPU_IDS="${GPU_IDS}" \
DRY_RUN="${DRY_RUN}" \
FORCE="${FORCE}" \
RESULTS_DIR="${SC_RESULTS_DIR:-./results_sc_500_uncertainty_weights}" \
bash scripts/run_sc_500_uncertainty_weights.sh

PYTHON_BIN="${PYTHON_BIN}" \
SAA_DATA_ROOT="${SAA_DATA_ROOT}" \
GPU_IDS="${GPU_IDS}" \
DRY_RUN="${DRY_RUN}" \
FORCE="${FORCE}" \
RESULTS_DIR="${FSDA_RESULTS_DIR:-./results_bl_lb_cs_fsda_uncertainty}" \
bash scripts/run_bl_lb_cs_fsda_uncertainty.sh

echo "All requested short uncertainty runs completed."
