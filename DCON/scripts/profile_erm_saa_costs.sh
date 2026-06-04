#!/usr/bin/env bash
# Profile ERM and SAA only, using the main DCON/train.py entry.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python 2>/dev/null || command -v python3)}"
GPU="${GPU:-${GPU_IDS:-0}}"
SOURCE="${SOURCE:-CHAOST2}"
EPOCHS="${EPOCHS:-20}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_NAME="${RUN_NAME:-}"

SAA_DATA_ROOT="${SAA_DATA_ROOT:-${PROJECT_DIR}/data}"
if [[ "${SAA_DATA_ROOT}" != /* ]]; then
  SAA_DATA_ROOT="${PROJECT_DIR}/${SAA_DATA_ROOT}"
fi
export SAA_DATA_ROOT

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_training_costs.py"
  --methods ERM SAA
  --gpu "${GPU}"
  --source "${SOURCE}"
  --epochs "${EPOCHS}"
  --warmup-epochs "${WARMUP_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)

if [[ -n "${RUN_NAME}" ]]; then
  cmd+=(--run-name "${RUN_NAME}")
fi
cmd+=("$@")

cd "${PROJECT_DIR}"
echo "SAA_DATA_ROOT=${SAA_DATA_ROOT}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"
