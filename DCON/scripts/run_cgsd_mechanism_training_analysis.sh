#!/usr/bin/env bash
set -euo pipefail

# Train full SAA and w/o CGSD checkpoints, then run CGSD-SAAM mechanism analysis.
#
# Examples:
#   DATA_NAME=CARDIAC SOURCE=bSSFP TARGET=LGE bash scripts/run_cgsd_mechanism_training_analysis.sh
#   DATA_NAME=ABDOMINAL SOURCE=SABSCT TARGET=CHAOST2 ALL_EPOCH=500 SAVE_FREQ=50 bash scripts/run_cgsd_mechanism_training_analysis.sh

DATA_NAME="${DATA_NAME:-CARDIAC}"
SOURCE="${SOURCE:-bSSFP}"
TARGET="${TARGET:-}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-42}"
CKPT_DIR="${CKPT_DIR:-./ckpts}"
ALL_EPOCH="${ALL_EPOCH:-300}"
SAVE_FREQ="${SAVE_FREQ:-50}"
VAL_FREQ="${VAL_FREQ:-50}"
MECH_INTERVAL="${MECH_INTERVAL:-200}"
ANALYZE_EPOCHS="${ANALYZE_EPOCHS:-50:${ALL_EPOCH}:${SAVE_FREQ}}"
MAX_SLICES="${MAX_SLICES:-120}"
NUM_VISUAL_CASES="${NUM_VISUAL_CASES:-3}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
RUN_TRAINING_CURVES="${RUN_TRAINING_CURVES:-1}"

if [[ "${DATA_NAME}" == "CARDIAC" ]]; then
  NCLASS=4
  SGF_GRID_SIZE="${SGF_GRID_SIZE:-18}"
  if [[ -z "${TARGET}" ]]; then
    if [[ "${SOURCE}" == "LGE" ]]; then TARGET="bSSFP"; else TARGET="LGE"; fi
  fi
elif [[ "${DATA_NAME}" == "ABDOMINAL" ]]; then
  NCLASS=5
  SGF_GRID_SIZE="${SGF_GRID_SIZE:-3}"
  if [[ -z "${TARGET}" ]]; then
    if [[ "${SOURCE}" == "SABSCT" ]]; then TARGET="CHAOST2"; else TARGET="SABSCT"; fi
  fi
else
  echo "Unsupported DATA_NAME=${DATA_NAME}; use CARDIAC or ABDOMINAL." >&2
  exit 1
fi

PAIR_TAG="${DATA_NAME}_${SOURCE}_to_${TARGET}"
FULL_EXP="${FULL_EXP:-cgsd_mech_full_${PAIR_TAG}}"
WO_EXP="${WO_EXP:-cgsd_mech_wo_cgsd_${PAIR_TAG}}"
OUT_DIR="${OUT_DIR:-results/cgsd_mechanism_analysis/${PAIR_TAG}}"

COMMON_ARGS=(
  --use_sgf 1
  --sgf_grid_size "${SGF_GRID_SIZE}"
  --num_workers "${NUM_WORKERS:-8}"
  --phase train
  --ckpt_dir "${CKPT_DIR}"
  --gpu_ids "${GPU_IDS}"
  --f_seed "${SEED}"
  --lr "${LR:-0.0005}"
  --model unet
  --batchSize "${BATCH_SIZE:-20}"
  --all_epoch "${ALL_EPOCH}"
  --validation_freq "${VAL_FREQ}"
  --display_freq "${DISPLAY_FREQ:-5000}"
  --save_freq "${SAVE_FREQ}"
  --data_name "${DATA_NAME}"
  --nclass "${NCLASS}"
  --tr_domain "${SOURCE}"
  --target_domain "${TARGET}"
  --save_prediction False
  --w_ce 1.0
  --w_dice 1.0
  --w_seg 1.0
  --cgsd_layer 1
  --use_projector 1
  --use_separate_cgsd_optimizer 1
  --lambda_str "${LAMBDA_STR:-0.3}"
  --lambda_sty "${LAMBDA_STY:-0.3}"
  --use_saam 1
  --saam_tau "${SAAM_TAU:-0.5}"
  --saam_topk "${SAAM_TOPK:-0.3}"
  --saam_stability_mode mean
  --lambda_01 1.0
  --lambda_02 1.0
  --saam_warmup_epochs "${SAAM_WARMUP:-50}"
  --saam_rampup_epochs "${SAAM_RAMPUP:-100}"
  --anchor_seg_alpha 0.0
  --strong_seg_alpha 1.0
  --mechanism_log_interval "${MECH_INTERVAL}"
  --mechanism_morph_kernel "${MECH_KERNEL:-3}"
  --use_rccs 1
  --p_rccs "${P_RCCS:-0.3}"
  --rccs_candidates 4
  --rccs_metric cos
  --rccs_embed_dim 128
)

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "Training full SAA: ${FULL_EXP}"
  python train.py \
    "${COMMON_ARGS[@]}" \
    --expname "${FULL_EXP}" \
    --use_cgsd 1

  echo "Training w/o CGSD: ${WO_EXP}"
  python train.py \
    "${COMMON_ARGS[@]}" \
    --expname "${WO_EXP}" \
    --use_cgsd 0
fi

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
  echo "Running CGSD-SAAM mechanism analysis: ${PAIR_TAG}"
  python tools/analyze_cgsd_saam_mechanism.py \
    --full_ckpt_template "${CKPT_DIR}/${SOURCE}/${FULL_EXP}/snapshots/{epoch}_net_Seg.pth" \
    --wo_cgsd_ckpt_template "${CKPT_DIR}/${SOURCE}/${WO_EXP}/snapshots/{epoch}_net_Seg.pth" \
    --epochs "${ANALYZE_EPOCHS}" \
    --data_name "${DATA_NAME}" \
    --source "${SOURCE}" \
    --target "${TARGET}" \
    --split target_test \
    --saam_tau "${SAAM_TAU:-0.5}" \
    --saam_topk "${SAAM_TOPK:-0.3}" \
    --morph_kernel "${MECH_KERNEL:-3}" \
    --max_slices "${MAX_SLICES}" \
    --num_visual_cases "${NUM_VISUAL_CASES}" \
    --out_dir "${OUT_DIR}"
fi

if [[ "${RUN_TRAINING_CURVES}" == "1" ]]; then
  echo "Plotting training-time CGSD mechanism curves: ${FULL_EXP}"
  python tools/plot_training_mechanism_curves.py \
    --exp_dir "${CKPT_DIR}/${SOURCE}/${FULL_EXP}" \
    --out_dir "${OUT_DIR}/figures/training_curves" \
    --smooth "${CURVE_SMOOTH:-5}" \
    --collapse_threshold "${STYLE_COLLAPSE_THRESHOLD:-0.02}"
fi

echo "Done. Results: ${OUT_DIR}"
