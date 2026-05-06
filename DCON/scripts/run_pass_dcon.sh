#!/usr/bin/env bash
# Run the PASS-adapted DCON evaluations on all four DCON shifts.
#
# Examples:
#   bash scripts/run_pass_dcon.sh
#   PASS_STEPS=2 PASS_LR=1e-3 bash scripts/run_pass_dcon.sh
#   PYTHON_BIN=/path/to/env/bin/python SAA_DATA_ROOT=/path/to/data bash scripts/run_pass_dcon.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHODS="${METHODS:-pass}"
export RESULTS_DIR="${RESULTS_DIR:-results_pass_dcon}"
export SAVE_PREDICTION="${SAVE_PREDICTION:-false}"
export EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-false}"

bash "${SCRIPT_DIR}/run_medseg_tta_dcon.sh"


