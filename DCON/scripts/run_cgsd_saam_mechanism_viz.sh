#!/usr/bin/env bash
set -euo pipefail

# Generate CGSD/SAAM mechanism dynamics curves from training checkpoints.
#
# Example:
#   DATA_NAME=CARDIAC SOURCE=bSSFP TARGET=LGE bash scripts/run_cgsd_saam_mechanism_viz.sh

DATA_NAME="${DATA_NAME:-CARDIAC}"
SOURCE="${SOURCE:-bSSFP}"
TARGET="${TARGET:-}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-2026}"
CKPT_DIR="${CKPT_DIR:-./ckpts}"
EPOCHS="${EPOCHS:-50:300:25}"
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
OUT_DIR="${OUT_DIR:-results_mechanism_visualization}"

TAU_ARGS=()
if [[ -n "${UNSTABLE_TAU:-}" ]]; then
  TAU_ARGS+=(--unstable_tau "${UNSTABLE_TAU}")
else
  TAU_ARGS+=(--unstable_tau_quantile "${UNSTABLE_TAU_QUANTILE:-0.75}")
fi

python tools/analyze_cgsd_saam_training_dynamics.py \
  --full_ckpt_template "${CKPT_DIR}/${SOURCE}/${FULL_EXP}/snapshots/{epoch}_net_Seg.pth" \
  --wo_cgsd_ckpt_template "${CKPT_DIR}/${SOURCE}/${WO_EXP}/snapshots/{epoch}_net_Seg.pth" \
  --epochs "${EPOCHS}" \
  --data_name "${DATA_NAME}" \
  --tr_domain "${SOURCE}" \
  --target_domain "${TARGET}" \
  --split "${SPLIT}" \
  --out_dir "${OUT_DIR}" \
  --gpu_ids "${GPU_IDS}" \
  --seed "${SEED}" \
  --max_slices "${MAX_SLICES}" \
  --num_views_list "${NUM_VIEWS_LIST:-4,8}" \
  --distance_list "${DISTANCE_LIST:-cosine,l2}" \
  --stat_list "${STAT_LIST:-mean,median}" \
  --smooth_list "${SMOOTH_LIST:-1,2}" \
  --tau_source "${TAU_SOURCE:-combined}" \
  "${TAU_ARGS[@]}"
