#!/bin/bash
# Run test-time reliability estimation on the four DCON domain shifts.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi
GPU_IDS=${GPU_IDS:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
OUT_ROOT=${OUT_ROOT:-"${PROJECT_DIR}/reliability_outputs"}
RUN_TENT=${RUN_TENT:-true}
TENT_LR=${TENT_LR:-1e-4}
TENT_STEPS=${TENT_STEPS:-1}
TTA_VIEWS=${TTA_VIEWS:-5}
MAX_VIZ=${MAX_VIZ:-40}
MAX_CASES=${MAX_CASES:-0}
SEED=${SEED:-23}

run_eval() {
  local expname=$1
  local data_name=$2
  local nclass=$3
  local source_domain=$4
  local target_domain=$5
  local ckpt=$6

  local out_dir="${OUT_ROOT}/${expname}"
  mkdir -p "${out_dir}"

  echo "=========================================="
  echo "Reliability: ${data_name} ${source_domain} -> ${target_domain}"
  echo "Checkpoint: ${ckpt}"
  echo "Output: ${out_dir}"
  echo "=========================================="

  "${PYTHON_BIN}" reliability_eval.py \
    --expname "${expname}" \
    --data_name "${data_name}" \
    --nclass "${nclass}" \
    --tr_domain "${source_domain}" \
    --target_domain "${target_domain}" \
    --resume_path "${ckpt}" \
    --output_dir "${out_dir}" \
    --gpu_ids "${GPU_IDS}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --run_tent "${RUN_TENT}" \
    --tent_lr "${TENT_LR}" \
    --tent_steps "${TENT_STEPS}" \
    --tta_views "${TTA_VIEWS}" \
    --max_viz "${MAX_VIZ}" \
    --max_cases "${MAX_CASES}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 1
}

run_eval "bSSFP_to_LGE" "CARDIAC" 4 "bSSFP" "LGE" "./ckpts/dcon-bl-1200.pth"
run_eval "LGE_to_bSSFP" "CARDIAC" 4 "LGE" "bSSFP" "./ckpts/dcon-lb-500.pth"
run_eval "CHAOST2_to_SABSCT" "ABDOMINAL" 5 "CHAOST2" "SABSCT" "./ckpts/dcon-cs-200.pth"
run_eval "SABSCT_to_CHAOST2" "ABDOMINAL" 5 "SABSCT" "CHAOST2" "./ckpts/dcon-sc-300.pth"

echo "Reliability evaluation completed. Outputs are under ${OUT_ROOT}."
