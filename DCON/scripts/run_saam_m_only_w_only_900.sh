#!/usr/bin/env bash
# Run the foreground-prior isolation comparison requested for the M ablation:
#   M only: A=M
#   W only: A=W
#   Uniform align: A=1
# with a 900-epoch training budget.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DCON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export ONLY_VARIANTS="${ONLY_VARIANTS:-m_only w_only uniform_align}"
export EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-900}"
export RUN_PREFIX="${RUN_PREFIX:-saam_m_vs_w_900}"

cd "${DCON_DIR}"
bash scripts/run_saam_mask_ablation.sh
