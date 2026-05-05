#!/bin/bash
# Run SicTTA on the four bundled DCON source-only checkpoints.
#
# SicTTA is source-free, single-image continual TTA. Target labels are used only
# by DCON's evaluation code. The checkpoints are source-only DCON models, so
# CGSD/SAAM/RCCS are disabled to keep the test architecture aligned.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
SAVE_PREDICTION=${SAVE_PREDICTION:-false}
EVAL_SOURCE_DOMAIN=${EVAL_SOURCE_DOMAIN:-true}

SICTTA_MAX_LENS=${SICTTA_MAX_LENS:-40}
SICTTA_TOPK=${SICTTA_TOPK:-5}
SICTTA_THRESHOLD=${SICTTA_THRESHOLD:-0.9}
SICTTA_SELECT_POINTS=${SICTTA_SELECT_POINTS:-200}
SICTTA_EPISODIC=${SICTTA_EPISODIC:-false}

run_sictta() {
  local expname=$1
  local data_name=$2
  local nclass=$3
  local tr_domain=$4
  local ckpt=$5

  echo "=========================================="
  echo "SicTTA: ${data_name}, source=${tr_domain}, ckpt=${ckpt}"
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
    --save_prediction "${SAVE_PREDICTION}" \
    --eval_source_domain "${EVAL_SOURCE_DOMAIN}" \
    --tta sictta \
    --sictta_max_lens "${SICTTA_MAX_LENS}" \
    --sictta_topk "${SICTTA_TOPK}" \
    --sictta_threshold "${SICTTA_THRESHOLD}" \
    --sictta_select_points "${SICTTA_SELECT_POINTS}" \
    --sictta_episodic "${SICTTA_EPISODIC}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0

  echo ""
}

run_sictta "sictta_dcon_sc_sabsct" "ABDOMINAL" 5 "SABSCT" "../ckpts/dcon-sc-300.pth"
run_sictta "sictta_dcon_cs_chaost2" "ABDOMINAL" 5 "CHAOST2" "../ckpts/dcon-cs-200.pth"
run_sictta "sictta_dcon_bl_bssfp" "CARDIAC" 4 "bSSFP" "../ckpts/dcon-bl-1200.pth"
run_sictta "sictta_dcon_lb_lge" "CARDIAC" 4 "LGE" "../ckpts/dcon-lb-500.pth"

echo "=========================================="
echo "SicTTA source-only DCON evaluations completed!"
echo "=========================================="
