#!/bin/bash
# Run all available SM-PPM ablations on DCON source-only checkpoints.
#
# Available modes in the current DCON implementation:
#   full, source_ce_only, sm_ce, ppm_ce, source_free_proto

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SMPPM_ABLATION_MODE="${SMPPM_ABLATION_MODE:-all}"

bash "${SCRIPT_DIR}/run_smppm_sourceonly_ckpts.sh"
