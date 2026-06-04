#!/usr/bin/env bash
set -euo pipefail

# Replay w/ CGSD and w/o CGSD checkpoints and collect SAAM d_stab metrics.
#
# Example:
#   DATA_NAME=CARDIAC SOURCE=bSSFP TARGET=LGE bash scripts/analyze_dstab_cgsd.sh

DATA_NAME="${DATA_NAME:-CARDIAC}"
SOURCE="${SOURCE:-bSSFP}"
TARGET="${TARGET:-}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-2026}"
CKPT_DIR="${CKPT_DIR:-./ckpts}"
EPOCHS="${EPOCHS:-25:300:25}"
NUM_VIEWS="${NUM_VIEWS:-4}"
MAX_SLICES="${MAX_SLICES:-0}"
SPLIT="${SPLIT:-target_test}"

if [[ "${DATA_NAME}" == "CARDIAC" ]]; then
  if [[ -z "${TARGET}" ]]; then
    if [[ "${SOURCE}" == "LGE" ]]; then TARGET="bSSFP"; else TARGET="LGE"; fi
  fi
elif [[ "${DATA_NAME}" == "ABDOMINAL" ]]; then
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
OUT_DIR="${OUT_DIR:-results_dstab_analysis}"
SAVE_CSV="${SAVE_CSV:-${OUT_DIR}/${PAIR_TAG}_dstab.csv}"

TAU_ARGS=()
if [[ -n "${DSTAB_TAU:-}" ]]; then
  TAU_ARGS+=(--dstab_tau "${DSTAB_TAU}")
elif [[ -n "${DSTAB_TAU_QUANTILE:-}" ]]; then
  TAU_ARGS+=(--dstab_tau_quantile "${DSTAB_TAU_QUANTILE}")
else
  TAU_ARGS+=(--dstab_tau_quantile 0.70)
fi

python tools/analyze_dstab.py \
  --ckpt_template "${CKPT_DIR}/${SOURCE}/${FULL_EXP}/snapshots/{epoch}_net_Seg.pth" \
  --ckpt_template "${CKPT_DIR}/${SOURCE}/${WO_EXP}/snapshots/{epoch}_net_Seg.pth" \
  --variant_name "w_CGSD" \
  --variant_name "w/o_CGSD" \
  --use_cgsd 1 \
  --use_cgsd 0 \
  --epochs "${EPOCHS}" \
  --data_name "${DATA_NAME}" \
  --tr_domain "${SOURCE}" \
  --target_domain "${TARGET}" \
  --split "${SPLIT}" \
  --save_csv "${SAVE_CSV}" \
  --num_views "${NUM_VIEWS}" \
  --seed "${SEED}" \
  --max_slices "${MAX_SLICES}" \
  --gpu_ids "${GPU_IDS}" \
  "${TAU_ARGS[@]}"
