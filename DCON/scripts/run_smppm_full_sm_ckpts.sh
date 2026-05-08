#!/bin/bash
# Run only the SM-enabled SM-PPM full path and the SM-only CE ablation.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMPPM_ABLATION_MODE=${SMPPM_ABLATION_MODE:-"full sm_ce"} \
  bash "${SCRIPT_DIR}/run_smppm_sourceonly_ckpts.sh"
