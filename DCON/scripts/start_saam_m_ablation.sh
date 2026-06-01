#!/usr/bin/env bash
# One-command launcher for the reviewer-facing foreground prior M ablation.
#
# It runs four SAAM mask variants:
#   1. uniform_align: A=1
#   2. m_only:        A=M
#   3. w_only:        A=W
#   4. w_times_m:     A=W*M

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SAA_DATA_ROOT="${SAA_DATA_ROOT:-/Users/RexRyder/PycharmProjects/Dataset}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/bin/python}"
export GPU_IDS="${GPU_IDS:-0}"
export RUN_PREFIX="${RUN_PREFIX:-saam_m_isolation}"

# Optional filters:
#   ONLY_TASKS="CARDIAC:bSSFP" bash scripts/start_saam_m_ablation.sh
#   ONLY_VARIANTS="m_only w_only w_times_m" bash scripts/start_saam_m_ablation.sh
export ONLY_VARIANTS="${ONLY_VARIANTS:-uniform_align m_only w_only w_times_m}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -d "${SAA_DATA_ROOT}" ]]; then
  echo "SAA_DATA_ROOT does not exist: ${SAA_DATA_ROOT}" >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available in this Python environment. "
        "Set PYTHON_BIN to a CUDA-enabled environment before launching training."
    )

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

cd "${PROJECT_DIR}"
bash scripts/run_saam_mask_ablation.sh
