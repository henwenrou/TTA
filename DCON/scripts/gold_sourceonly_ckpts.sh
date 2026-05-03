#!/bin/bash
# Run GOLD test-time adaptation on the four bundled DCON source-only checkpoints.
#
# GOLD is source-free TTA. Target labels are used only by the evaluation code.
# The checkpoints are source-only DCON models, so CGSD/SAAM/RCCS are disabled to
# keep the test architecture aligned with the saved weights.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-0}
NUM_WORKERS=${NUM_WORKERS:-4}

GOLD_LR=${GOLD_LR:-2.5e-4}
GOLD_MOMENTUM=${GOLD_MOMENTUM:-0.9}
GOLD_WD=${GOLD_WD:-5e-4}
GOLD_STEPS=${GOLD_STEPS:-1}
GOLD_RANK=${GOLD_RANK:-128}
GOLD_TAU=${GOLD_TAU:-0.95}
GOLD_ALPHA=${GOLD_ALPHA:-0.02}
GOLD_T_EIG=${GOLD_T_EIG:-10}
GOLD_MT=${GOLD_MT:-0.999}
GOLD_S_LR=${GOLD_S_LR:-5e-3}
GOLD_S_INIT_SCALE=${GOLD_S_INIT_SCALE:-0.0}
GOLD_S_CLIP=${GOLD_S_CLIP:-0.5}
GOLD_ADAPTER_SCALE=${GOLD_ADAPTER_SCALE:-0.05}
GOLD_MAX_PIXELS_PER_BATCH=${GOLD_MAX_PIXELS_PER_BATCH:-512}
GOLD_MIN_PIXELS_PER_BATCH=${GOLD_MIN_PIXELS_PER_BATCH:-64}
GOLD_N_AUGMENTATIONS=${GOLD_N_AUGMENTATIONS:-6}
GOLD_RST=${GOLD_RST:-0.01}
GOLD_AP=${GOLD_AP:-0.9}
GOLD_EPISODIC=${GOLD_EPISODIC:-false}

run_gold() {
  local expname=$1
  local data_name=$2
  local nclass=$3
  local tr_domain=$4
  local ckpt=$5

  echo "=========================================="
  echo "GOLD: ${data_name}, source=${tr_domain}, ckpt=${ckpt}"
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
    --tta gold \
    --gold_lr "${GOLD_LR}" \
    --gold_momentum "${GOLD_MOMENTUM}" \
    --gold_wd "${GOLD_WD}" \
    --gold_steps "${GOLD_STEPS}" \
    --gold_rank "${GOLD_RANK}" \
    --gold_tau "${GOLD_TAU}" \
    --gold_alpha "${GOLD_ALPHA}" \
    --gold_t_eig "${GOLD_T_EIG}" \
    --gold_mt "${GOLD_MT}" \
    --gold_s_lr "${GOLD_S_LR}" \
    --gold_s_init_scale "${GOLD_S_INIT_SCALE}" \
    --gold_s_clip "${GOLD_S_CLIP}" \
    --gold_adapter_scale "${GOLD_ADAPTER_SCALE}" \
    --gold_max_pixels_per_batch "${GOLD_MAX_PIXELS_PER_BATCH}" \
    --gold_min_pixels_per_batch "${GOLD_MIN_PIXELS_PER_BATCH}" \
    --gold_n_augmentations "${GOLD_N_AUGMENTATIONS}" \
    --gold_rst "${GOLD_RST}" \
    --gold_ap "${GOLD_AP}" \
    --gold_episodic "${GOLD_EPISODIC}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0

  echo ""
}

run_gold "gold_dcon_sc_sabsct" "ABDOMINAL" 5 "SABSCT" "../ckpts/dcon-sc-300.pth"
run_gold "gold_dcon_cs_chaost2" "ABDOMINAL" 5 "CHAOST2" "../ckpts/dcon-cs-200.pth"
run_gold "gold_dcon_bl_bssfp" "CARDIAC" 4 "bSSFP" "../ckpts/dcon-bl-1200.pth"
run_gold "gold_dcon_lb_lge" "CARDIAC" 4 "LGE" "../ckpts/dcon-lb-500.pth"

echo "=========================================="
echo "GOLD source-only DCON evaluations completed!"
echo "=========================================="
