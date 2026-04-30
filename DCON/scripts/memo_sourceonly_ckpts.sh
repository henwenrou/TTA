#!/bin/bash
# Run MEMO test-time adaptation on the four bundled DCON source-only checkpoints.
#
# The checkpoints come from source-only DCON models, so CGSD/SAAM/RCCS are
# disabled to keep the test architecture aligned with the saved weights.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-0}
NUM_WORKERS=${NUM_WORKERS:-4}

MEMO_LR=${MEMO_LR:-1e-5}
MEMO_STEPS=${MEMO_STEPS:-1}
MEMO_N_AUGMENTATIONS=${MEMO_N_AUGMENTATIONS:-8}
MEMO_INCLUDE_IDENTITY=${MEMO_INCLUDE_IDENTITY:-1}
MEMO_HFLIP_P=${MEMO_HFLIP_P:-0.0}
MEMO_UPDATE_SCOPE=${MEMO_UPDATE_SCOPE:-all}

run_memo() {
  local expname=$1
  local data_name=$2
  local nclass=$3
  local tr_domain=$4
  local ckpt=$5

  echo "=========================================="
  echo "MEMO: ${data_name}, source=${tr_domain}, ckpt=${ckpt}"
  echo "=========================================="

  "${PYTHON_BIN}" train.py \
    --phase test \
    --expname "${expname}" \
    --data_name "${data_name}" \
    --nclass "${nclass}" \
    --tr_domain "${tr_domain}" \
    --resume_path "${ckpt}" \
    --gpu_ids "${GPU_IDS}" \
    --num_workers "${NUM_WORKERS}" \
    --tta memo \
    --memo_lr "${MEMO_LR}" \
    --memo_steps "${MEMO_STEPS}" \
    --memo_n_augmentations "${MEMO_N_AUGMENTATIONS}" \
    --memo_include_identity "${MEMO_INCLUDE_IDENTITY}" \
    --memo_hflip_p "${MEMO_HFLIP_P}" \
    --memo_update_scope "${MEMO_UPDATE_SCOPE}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0

  echo ""
}

run_memo "memo_dcon_sc_sabsct" "ABDOMINAL" 5 "SABSCT" "../ckpts/dcon-sc-300.pth"
run_memo "memo_dcon_cs_chaost2" "ABDOMINAL" 5 "CHAOST2" "../ckpts/dcon-cs-200.pth"
run_memo "memo_dcon_bl_bssfp" "CARDIAC" 4 "bSSFP" "../ckpts/dcon-bl-1200.pth"
run_memo "memo_dcon_lb_lge" "CARDIAC" 4 "LGE" "../ckpts/dcon-lb-500.pth"

echo "=========================================="
echo "MEMO source-only DCON evaluations completed!"
echo "=========================================="
