#!/bin/bash
# Run SM-PPM on DCON source-only checkpoints.
#
# full/source_ce_only/ppm_ce are source-dependent ablations. source_free_proto
# passes no source loader to the adapter.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

NUM_WORKERS=${NUM_WORKERS:-4}
GPU_IDS=${GPU_IDS:-0}
if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi
RESULTS_DIR=${RESULTS_DIR:-results}

SMPPM_LR=${SMPPM_LR:-2.5e-4}
SMPPM_MOMENTUM=${SMPPM_MOMENTUM:-0.9}
SMPPM_WD=${SMPPM_WD:-5e-4}
SMPPM_STEPS=${SMPPM_STEPS:-1}
SMPPM_SRC_BATCH_SIZE=${SMPPM_SRC_BATCH_SIZE:-2}
SMPPM_PATCH_SIZE=${SMPPM_PATCH_SIZE:-8}
SMPPM_FEATURE_SIZE=${SMPPM_FEATURE_SIZE:-32}
SMPPM_EPISODIC=${SMPPM_EPISODIC:-false}
SMPPM_SOURCE_FREE_TAU=${SMPPM_SOURCE_FREE_TAU:-0.7}
SMPPM_SOURCE_FREE_ENTROPY_THRESHOLD=${SMPPM_SOURCE_FREE_ENTROPY_THRESHOLD:-}
SMPPM_SOURCE_FREE_ENTROPY_WEIGHT=${SMPPM_SOURCE_FREE_ENTROPY_WEIGHT:-1.0}
SMPPM_SOURCE_FREE_LAMBDA_PROTO=${SMPPM_SOURCE_FREE_LAMBDA_PROTO:-1.0}

SMPPM_ABLATION_MODE=${SMPPM_ABLATION_MODE:-full}
if [ "${SMPPM_ABLATION_MODE}" = "all" ]; then
  SMPPM_ABLATION_MODES=(full source_ce_only ppm_ce source_free_proto)
else
  read -r -a SMPPM_ABLATION_MODES <<< "${SMPPM_ABLATION_MODE}"
fi

run_smppm() {
  local expname_base=$1
  local data_name=$2
  local nclass=$3
  local source=$4
  local target=$5
  local ckpt=$6
  local ablation_mode=$7
  local expname="smppm_${ablation_mode}_${expname_base}"

  case "${ablation_mode}" in
    full|source_ce_only|ppm_ce|source_free_proto) ;;
    sm_ce)
      echo "Skipping sm_ce: current DCON tta_smppm.py has no explicit SM style-mixing implementation."
      return 0
      ;;
    *)
      echo "Unknown SMPPM_ABLATION_MODE: ${ablation_mode}" >&2
      return 2
      ;;
  esac

  echo "=========================================="
  echo "SM-PPM ${ablation_mode}: ${data_name}, source=${source}, target=${target}, ckpt=${ckpt}"
  echo "=========================================="

  local entropy_threshold_args=()
  if [ -n "${SMPPM_SOURCE_FREE_ENTROPY_THRESHOLD}" ]; then
    entropy_threshold_args=(--smppm_source_free_entropy_threshold "${SMPPM_SOURCE_FREE_ENTROPY_THRESHOLD}")
  fi

  "${PYTHON_BIN}" train.py \
    --phase test \
    --expname "${expname}" \
    --ckpt_dir "${RESULTS_DIR}" \
    --dataset "${data_name}" \
    --nclass "${nclass}" \
    --source "${source}" \
    --target "${target}" \
    --restore_from "${ckpt}" \
    --gpu_ids "${GPU_IDS}" \
    --num_workers "${NUM_WORKERS}" \
    --tta sm_ppm \
    --smppm_lr "${SMPPM_LR}" \
    --smppm_momentum "${SMPPM_MOMENTUM}" \
    --smppm_wd "${SMPPM_WD}" \
    --smppm_steps "${SMPPM_STEPS}" \
    --smppm_src_batch_size "${SMPPM_SRC_BATCH_SIZE}" \
    --smppm_patch_size "${SMPPM_PATCH_SIZE}" \
    --smppm_feature_size "${SMPPM_FEATURE_SIZE}" \
    --smppm_episodic "${SMPPM_EPISODIC}" \
    --smppm_ablation_mode "${ablation_mode}" \
    --smppm_source_free_tau "${SMPPM_SOURCE_FREE_TAU}" \
    "${entropy_threshold_args[@]}" \
    --smppm_source_free_entropy_weight "${SMPPM_SOURCE_FREE_ENTROPY_WEIGHT}" \
    --smppm_source_free_lambda_proto "${SMPPM_SOURCE_FREE_LAMBDA_PROTO}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0

  echo ""
}

for ablation_mode in "${SMPPM_ABLATION_MODES[@]}"; do
  run_smppm "dcon_sc_chaost2" "ABDOMINAL" 5 "SABSCT" "CHAOST2" "../ckpts/dcon-sc-300.pth" "${ablation_mode}"
  run_smppm "dcon_cs_sabsct" "ABDOMINAL" 5 "CHAOST2" "SABSCT" "../ckpts/dcon-cs-200.pth" "${ablation_mode}"
  run_smppm "dcon_bl_lge" "CARDIAC" 4 "bSSFP" "LGE" "../ckpts/dcon-bl-1200.pth" "${ablation_mode}"
  run_smppm "dcon_lb_bssfp" "CARDIAC" 4 "LGE" "bSSFP" "../ckpts/dcon-lb-500.pth" "${ablation_mode}"
done

echo "=========================================="
echo "SM-PPM DCON TTA evaluations completed!"
echo "=========================================="
