#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
POSTFIX="${POSTFIX:-_affine_clp}"

run_one() {
  local task="$1"
  local config="$2"
  local epochs="$3"
  local source="$4"
  local target="$5"

  echo "================================================================"
  echo "SLAug affine-CLP: ${task} ${source}->${target}, epochs=${epochs}"
  echo "Config: ${config}"
  echo "================================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" main.py \
    --base "${config}" \
    --seed "${SEED}" \
    --postfix "${POSTFIX}_${task}_${epochs}ep" \
    optimizer.max_epoch="${epochs}" \
    data.params.train.params.local_aug_type=affine_clp
}

cd "${ROOT_DIR}"

run_one "bl" "configs/efficientUnet_bSSFP_to_LEG.yaml" "700" "bSSFP" "LGE"
run_one "cs" "configs/efficientUnet_CHAOS_to_SABSCT.yaml" "800" "CHAOST2" "SABSCT"

echo "Done. Check SLAug/logs for checkpoints and validation/test logs."
