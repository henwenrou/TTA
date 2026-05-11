#!/bin/bash
# Continue SAAM-SPMM testing for source domains other than SABSCT.
# This is useful after SABSCT -> CHAOST2 has already been completed.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conservative defaults for remote sessions.
export SAAM_SPMM_ABLATION_MODE="${SAAM_SPMM_ABLATION_MODE:-full}"
export NUM_VIEWS="${NUM_VIEWS:-2}"
export SAAM_SPMM_STEPS="${SAAM_SPMM_STEPS:-1}"
export SAAM_SPMM_UPDATE_SCOPE="${SAAM_SPMM_UPDATE_SCOPE:-bn_affine}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
export SAVE_PREDICTION="${SAVE_PREDICTION:-false}"
export EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-false}"
export QUIET_CONSOLE="${QUIET_CONSOLE:-true}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export SKIP_FINISHED="${SKIP_FINISHED:-true}"

# Keep this relative to DCON by default:
#   /path/to/TTA/DCON/ckpts/<source>/source_prototypes_*.pt
export PROTO_ROOT="${PROTO_ROOT:-ckpts}"

run_pair() {
  local filter=$1
  echo "=========================================="
  echo "Continue SAAM-SPMM pair filter: ${filter}"
  echo "=========================================="
  PAIR_FILTER="${filter}" bash "${SCRIPT_DIR}/run_saam_spmm.sh"
}

# Remaining source domains besides SABSCT.
run_pair "CHAOST2_SABSCT"
run_pair "bSSFP_LGE"
run_pair "LGE_bSSFP"

echo "=========================================="
echo "Non-SABSCT SAAM-SPMM continuation completed."
echo "=========================================="
