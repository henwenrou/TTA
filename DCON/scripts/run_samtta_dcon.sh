#!/usr/bin/env bash
# Run the SAM-TTA-adapted DCON evaluations on all four DCON shifts.
#
# Examples:
#   bash scripts/run_samtta_dcon.sh
#   SAMTTA_STEPS=2 SAMTTA_TRANSFORM_LR=5e-3 bash scripts/run_samtta_dcon.sh
#   PYTHON_BIN=/path/to/env/bin/python SAA_DATA_ROOT=/path/to/data bash scripts/run_samtta_dcon.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHODS="${METHODS:-samtta}"
export RESULTS_DIR="${RESULTS_DIR:-results_samtta_dcon}"
export SAVE_PREDICTION="${SAVE_PREDICTION:-false}"
export EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-false}"

bash "${SCRIPT_DIR}/run_medseg_tta_dcon.sh"
