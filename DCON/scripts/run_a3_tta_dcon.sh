#!/usr/bin/env bash
# Run A3-TTA on all four DCON source-only checkpoint shifts.
#
# Examples:
#   bash scripts/run_a3_tta_dcon.sh
#   A3_STEPS=2 A3_LR=5e-5 bash scripts/run_a3_tta_dcon.sh
#   DRY_RUN=1 bash scripts/run_a3_tta_dcon.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHODS="${METHODS:-a3_tta}"
export RESULTS_DIR="${RESULTS_DIR:-results_a3_tta_dcon}"
export SAVE_PREDICTION="${SAVE_PREDICTION:-false}"
export EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-false}"

bash "${SCRIPT_DIR}/run_medseg_tta_dcon.sh"
