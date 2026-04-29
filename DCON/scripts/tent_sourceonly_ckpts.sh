#!/bin/bash
# Run TENT test-time adaptation on the source-only DCON checkpoints in ../ckpts.
#
# These checkpoints come from the original DCON source-only model, so CGSD/SAAM/RCCS
# are disabled here to keep the test architecture aligned with the saved weights.

set -e

TENT_LR=${TENT_LR:-1e-4}
TENT_STEPS=${TENT_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
GPU_IDS=${GPU_IDS:-0}

run_tent() {
  local expname=$1
  local data_name=$2
  local nclass=$3
  local tr_domain=$4
  local ckpt=$5

  echo "=========================================="
  echo "TENT: ${data_name}, source=${tr_domain}, ckpt=${ckpt}"
  echo "=========================================="

  python train.py \
    --phase test \
    --expname "${expname}" \
    --data_name "${data_name}" \
    --nclass "${nclass}" \
    --tr_domain "${tr_domain}" \
    --resume_path "${ckpt}" \
    --gpu_ids "${GPU_IDS}" \
    --num_workers "${NUM_WORKERS}" \
    --tta tent \
    --tent_lr "${TENT_LR}" \
    --tent_steps "${TENT_STEPS}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0

  echo ""
}

run_tent "tent_dcon_sc_sabsct" "ABDOMINAL" 5 "SABSCT" "../ckpts/dcon-sc-300.pth"
run_tent "tent_dcon_cs_chaost2" "ABDOMINAL" 5 "CHAOST2" "../ckpts/dcon-cs-200.pth"
run_tent "tent_dcon_bl_bssfp" "CARDIAC" 4 "bSSFP" "../ckpts/dcon-bl-1200.pth"
run_tent "tent_dcon_lb_lge" "CARDIAC" 4 "LGE" "../ckpts/dcon-lb-500.pth"

echo "=========================================="
echo "TENT source-only DCON evaluations completed!"
echo "=========================================="
