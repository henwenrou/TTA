#!/usr/bin/env bash
set -euo pipefail

# Plot final-checkpoint SAAM d_stab distributions for w/ vs w/o CGSD.
#
# Example:
#   DATA_NAME=CARDIAC SOURCE=bSSFP TARGET=LGE bash scripts/plot_final_dstab_distribution.sh

DATA_NAME="${DATA_NAME:-CARDIAC}"
SOURCE="${SOURCE:-bSSFP}"
TARGET="${TARGET:-}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-2026}"
CKPT_DIR="${CKPT_DIR:-./ckpts}"
EPOCH="${EPOCH:-300}"
NUM_VIEWS="${NUM_VIEWS:-8}"
MAX_SLICES="${MAX_SLICES:-0}"
SPLIT="${SPLIT:-target_test}"
OUT_DIR="${OUT_DIR:-results_dstab_distribution}"

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

TAU_ARGS=()
if [[ -n "${DSTAB_TAU:-}" ]]; then
  TAU_ARGS+=(--dstab_tau "${DSTAB_TAU}")
else
  TAU_ARGS+=(--dstab_tau_percentile "${DSTAB_TAU_PERCENTILE:-75}")
fi

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

"${PYTHON:-python}" tools/plot_final_dstab_distribution.py \
  --full_ckpt_template "${CKPT_DIR}/${SOURCE}/${FULL_EXP}/snapshots/{epoch}_net_Seg.pth" \
  --wo_cgsd_ckpt_template "${CKPT_DIR}/${SOURCE}/${WO_EXP}/snapshots/{epoch}_net_Seg.pth" \
  --epoch "${EPOCH}" \
  --data_name "${DATA_NAME}" \
  --tr_domain "${SOURCE}" \
  --target_domain "${TARGET}" \
  --split "${SPLIT}" \
  --out_dir "${OUT_DIR}" \
  --gpu_ids "${GPU_IDS}" \
  --seed "${SEED}" \
  --num_views "${NUM_VIEWS}" \
  --max_slices "${MAX_SLICES}" \
  "${TAU_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"
