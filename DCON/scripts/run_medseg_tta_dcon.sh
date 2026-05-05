#!/usr/bin/env bash
# Run the DCON-adapted MedSeg-TTA baselines on all four DCON shifts.
#
# Examples:
#   bash scripts/run_medseg_tta_dcon.sh
#   METHODS="tent dg_tta gold" bash scripts/run_medseg_tta_dcon.sh
#   PYTHON_BIN=/path/to/env/bin/python SAA_DATA_ROOT=/path/to/data bash scripts/run_medseg_tta_dcon.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "No python executable found. Set PYTHON_BIN=/path/to/env/bin/python." >&2
    exit 1
  fi
fi

if [ -z "${SAA_DATA_ROOT:-}" ]; then
  if [ -d "/Users/RexRyder/PycharmProjects/Dataset" ]; then
    export SAA_DATA_ROOT="/Users/RexRyder/PycharmProjects/Dataset"
  else
    export SAA_DATA_ROOT="${PROJECT_DIR}/data"
  fi
fi
RESULTS_DIR="${RESULTS_DIR:-results_medseg_tta}"
CKPT_ROOT="${CKPT_ROOT:-../ckpts}"
GPU_IDS="${GPU_IDS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_PREDICTION="${SAVE_PREDICTION:-false}"
EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-false}"
DRY_RUN="${DRY_RUN:-0}"

METHODS="${METHODS:-none norm_test norm_alpha norm_ema tent dg_tta cotta memo asm sm_ppm gtta gold}"

BN_ALPHA="${BN_ALPHA:-0.1}"

TENT_LR="${TENT_LR:-1e-4}"
TENT_STEPS="${TENT_STEPS:-1}"

DGTTA_LR="${DGTTA_LR:-1e-4}"
DGTTA_STEPS="${DGTTA_STEPS:-1}"
DGTTA_TRANSFORM_STRENGTH="${DGTTA_TRANSFORM_STRENGTH:-1.0}"
DGTTA_ENTROPY_WEIGHT="${DGTTA_ENTROPY_WEIGHT:-0.05}"
DGTTA_BN_L2_REG="${DGTTA_BN_L2_REG:-1e-4}"
DGTTA_EPISODIC="${DGTTA_EPISODIC:-false}"

COTTA_LR="${COTTA_LR:-1e-4}"
COTTA_STEPS="${COTTA_STEPS:-1}"
COTTA_MT="${COTTA_MT:-0.999}"
COTTA_RST="${COTTA_RST:-0.01}"
COTTA_AP="${COTTA_AP:-0.9}"

MEMO_LR="${MEMO_LR:-1e-5}"
MEMO_STEPS="${MEMO_STEPS:-1}"
MEMO_N_AUGMENTATIONS="${MEMO_N_AUGMENTATIONS:-8}"
MEMO_INCLUDE_IDENTITY="${MEMO_INCLUDE_IDENTITY:-1}"
MEMO_HFLIP_P="${MEMO_HFLIP_P:-0.0}"
MEMO_UPDATE_SCOPE="${MEMO_UPDATE_SCOPE:-all}"

ASM_LR="${ASM_LR:-1e-4}"
ASM_STEPS="${ASM_STEPS:-1}"
ASM_INNER_STEPS="${ASM_INNER_STEPS:-2}"
ASM_LAMBDA_REG="${ASM_LAMBDA_REG:-2e-4}"
ASM_SAMPLING_STEP="${ASM_SAMPLING_STEP:-20.0}"
ASM_SRC_BATCH_SIZE="${ASM_SRC_BATCH_SIZE:-4}"
ASM_STYLE_BACKEND="${ASM_STYLE_BACKEND:-medical_adain}"
ASM_EPISODIC="${ASM_EPISODIC:-false}"

SMPPM_LR="${SMPPM_LR:-2.5e-4}"
SMPPM_MOMENTUM="${SMPPM_MOMENTUM:-0.9}"
SMPPM_WD="${SMPPM_WD:-5e-4}"
SMPPM_STEPS="${SMPPM_STEPS:-1}"
SMPPM_SRC_BATCH_SIZE="${SMPPM_SRC_BATCH_SIZE:-2}"
SMPPM_PATCH_SIZE="${SMPPM_PATCH_SIZE:-8}"
SMPPM_FEATURE_SIZE="${SMPPM_FEATURE_SIZE:-32}"
SMPPM_EPISODIC="${SMPPM_EPISODIC:-false}"

GTTA_LR="${GTTA_LR:-2.5e-4}"
GTTA_MOMENTUM="${GTTA_MOMENTUM:-0.9}"
GTTA_WD="${GTTA_WD:-5e-4}"
GTTA_STEPS="${GTTA_STEPS:-1}"
GTTA_SRC_BATCH_SIZE="${GTTA_SRC_BATCH_SIZE:-2}"
GTTA_LAMBDA_CE_TRG="${GTTA_LAMBDA_CE_TRG:-0.1}"
GTTA_PSEUDO_MOMENTUM="${GTTA_PSEUDO_MOMENTUM:-0.9}"
GTTA_STYLE_ALPHA="${GTTA_STYLE_ALPHA:-1.0}"
GTTA_INCLUDE_ORIGINAL="${GTTA_INCLUDE_ORIGINAL:-1}"
GTTA_EPISODIC="${GTTA_EPISODIC:-false}"

GOLD_LR="${GOLD_LR:-2.5e-4}"
GOLD_MOMENTUM="${GOLD_MOMENTUM:-0.9}"
GOLD_WD="${GOLD_WD:-5e-4}"
GOLD_STEPS="${GOLD_STEPS:-1}"
GOLD_RANK="${GOLD_RANK:-128}"
GOLD_TAU="${GOLD_TAU:-0.95}"
GOLD_ALPHA="${GOLD_ALPHA:-0.02}"
GOLD_T_EIG="${GOLD_T_EIG:-10}"
GOLD_MT="${GOLD_MT:-0.999}"
GOLD_S_LR="${GOLD_S_LR:-5e-3}"
GOLD_S_INIT_SCALE="${GOLD_S_INIT_SCALE:-0.0}"
GOLD_S_CLIP="${GOLD_S_CLIP:-0.5}"
GOLD_ADAPTER_SCALE="${GOLD_ADAPTER_SCALE:-0.05}"
GOLD_MAX_PIXELS_PER_BATCH="${GOLD_MAX_PIXELS_PER_BATCH:-512}"
GOLD_MIN_PIXELS_PER_BATCH="${GOLD_MIN_PIXELS_PER_BATCH:-64}"
GOLD_N_AUGMENTATIONS="${GOLD_N_AUGMENTATIONS:-6}"
GOLD_RST="${GOLD_RST:-0.01}"
GOLD_AP="${GOLD_AP:-0.9}"
GOLD_EPISODIC="${GOLD_EPISODIC:-false}"

PASS_OPTIMIZER="${PASS_OPTIMIZER:-Adam}"
PASS_LR="${PASS_LR:-5e-3}"
PASS_MOMENTUM="${PASS_MOMENTUM:-0.99}"
PASS_BETA1="${PASS_BETA1:-0.9}"
PASS_BETA2="${PASS_BETA2:-0.999}"
PASS_WEIGHT_DECAY="${PASS_WEIGHT_DECAY:-0.0}"
PASS_STEPS="${PASS_STEPS:-1}"
PASS_BN_ALPHA="${PASS_BN_ALPHA:-0.01}"
PASS_BN_LAYERS="${PASS_BN_LAYERS:-0}"
PASS_ENTROPY_WEIGHT="${PASS_ENTROPY_WEIGHT:-0.0}"
PASS_EMA_DECAY="${PASS_EMA_DECAY:-0.94}"
PASS_MIN_MOMENTUM_CONSTANT="${PASS_MIN_MOMENTUM_CONSTANT:-0.01}"
PASS_EPISODIC="${PASS_EPISODIC:-false}"
PASS_USE_SOURCE_FALLBACK="${PASS_USE_SOURCE_FALLBACK:-true}"
PASS_IMAGE_SIZE="${PASS_IMAGE_SIZE:-192}"
PASS_PROMPT_SIZE="${PASS_PROMPT_SIZE:-}"
PASS_ADAPTOR_HIDDEN="${PASS_ADAPTOR_HIDDEN:-64}"
PASS_PERTURB_SCALE="${PASS_PERTURB_SCALE:-1.0}"
PASS_PROMPT_SCALE="${PASS_PROMPT_SCALE:-1.0}"
PASS_PROMPT_SPARSITY="${PASS_PROMPT_SPARSITY:-0.1}"

if [ ! -d "${SAA_DATA_ROOT}" ]; then
  echo "SAA_DATA_ROOT does not exist: ${SAA_DATA_ROOT}" >&2
  exit 1
fi

require_ckpt() {
  local ckpt="$1"
  if [ ! -f "${ckpt}" ]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
}

method_args() {
  local method="$1"
  case "${method}" in
    none|norm_test|norm_ema)
      ;;
    norm_alpha)
      METHOD_EXTRA=(--bn_alpha "${BN_ALPHA}")
      ;;
    tent)
      METHOD_EXTRA=(--tent_lr "${TENT_LR}" --tent_steps "${TENT_STEPS}")
      ;;
    dg_tta)
      METHOD_EXTRA=(
        --dgtta_lr "${DGTTA_LR}"
        --dgtta_steps "${DGTTA_STEPS}"
        --dgtta_transform_strength "${DGTTA_TRANSFORM_STRENGTH}"
        --dgtta_entropy_weight "${DGTTA_ENTROPY_WEIGHT}"
        --dgtta_bn_l2_reg "${DGTTA_BN_L2_REG}"
        --dgtta_episodic "${DGTTA_EPISODIC}"
      )
      ;;
    cotta)
      METHOD_EXTRA=(
        --cotta_lr "${COTTA_LR}"
        --cotta_steps "${COTTA_STEPS}"
        --cotta_mt "${COTTA_MT}"
        --cotta_rst "${COTTA_RST}"
        --cotta_ap "${COTTA_AP}"
      )
      ;;
    memo)
      METHOD_EXTRA=(
        --memo_lr "${MEMO_LR}"
        --memo_steps "${MEMO_STEPS}"
        --memo_n_augmentations "${MEMO_N_AUGMENTATIONS}"
        --memo_include_identity "${MEMO_INCLUDE_IDENTITY}"
        --memo_hflip_p "${MEMO_HFLIP_P}"
        --memo_update_scope "${MEMO_UPDATE_SCOPE}"
      )
      ;;
    asm)
      METHOD_EXTRA=(
        --asm_lr "${ASM_LR}"
        --asm_steps "${ASM_STEPS}"
        --asm_inner_steps "${ASM_INNER_STEPS}"
        --asm_lambda_reg "${ASM_LAMBDA_REG}"
        --asm_sampling_step "${ASM_SAMPLING_STEP}"
        --asm_src_batch_size "${ASM_SRC_BATCH_SIZE}"
        --asm_style_backend "${ASM_STYLE_BACKEND}"
        --asm_episodic "${ASM_EPISODIC}"
      )
      ;;
    sm_ppm)
      METHOD_EXTRA=(
        --smppm_lr "${SMPPM_LR}"
        --smppm_momentum "${SMPPM_MOMENTUM}"
        --smppm_wd "${SMPPM_WD}"
        --smppm_steps "${SMPPM_STEPS}"
        --smppm_src_batch_size "${SMPPM_SRC_BATCH_SIZE}"
        --smppm_patch_size "${SMPPM_PATCH_SIZE}"
        --smppm_feature_size "${SMPPM_FEATURE_SIZE}"
        --smppm_episodic "${SMPPM_EPISODIC}"
      )
      ;;
    gtta)
      METHOD_EXTRA=(
        --gtta_lr "${GTTA_LR}"
        --gtta_momentum "${GTTA_MOMENTUM}"
        --gtta_wd "${GTTA_WD}"
        --gtta_steps "${GTTA_STEPS}"
        --gtta_src_batch_size "${GTTA_SRC_BATCH_SIZE}"
        --gtta_lambda_ce_trg "${GTTA_LAMBDA_CE_TRG}"
        --gtta_pseudo_momentum "${GTTA_PSEUDO_MOMENTUM}"
        --gtta_style_alpha "${GTTA_STYLE_ALPHA}"
        --gtta_include_original "${GTTA_INCLUDE_ORIGINAL}"
        --gtta_episodic "${GTTA_EPISODIC}"
      )
      ;;
    gold)
      METHOD_EXTRA=(
        --gold_lr "${GOLD_LR}"
        --gold_momentum "${GOLD_MOMENTUM}"
        --gold_wd "${GOLD_WD}"
        --gold_steps "${GOLD_STEPS}"
        --gold_rank "${GOLD_RANK}"
        --gold_tau "${GOLD_TAU}"
        --gold_alpha "${GOLD_ALPHA}"
        --gold_t_eig "${GOLD_T_EIG}"
        --gold_mt "${GOLD_MT}"
        --gold_s_lr "${GOLD_S_LR}"
        --gold_s_init_scale "${GOLD_S_INIT_SCALE}"
        --gold_s_clip "${GOLD_S_CLIP}"
        --gold_adapter_scale "${GOLD_ADAPTER_SCALE}"
        --gold_max_pixels_per_batch "${GOLD_MAX_PIXELS_PER_BATCH}"
        --gold_min_pixels_per_batch "${GOLD_MIN_PIXELS_PER_BATCH}"
        --gold_n_augmentations "${GOLD_N_AUGMENTATIONS}"
        --gold_rst "${GOLD_RST}"
        --gold_ap "${GOLD_AP}"
        --gold_episodic "${GOLD_EPISODIC}"
      )
      ;;
    pass)
      METHOD_EXTRA=(
        --pass_optimizer "${PASS_OPTIMIZER}"
        --pass_lr "${PASS_LR}"
        --pass_momentum "${PASS_MOMENTUM}"
        --pass_beta1 "${PASS_BETA1}"
        --pass_beta2 "${PASS_BETA2}"
        --pass_weight_decay "${PASS_WEIGHT_DECAY}"
        --pass_steps "${PASS_STEPS}"
        --pass_bn_alpha "${PASS_BN_ALPHA}"
        --pass_bn_layers "${PASS_BN_LAYERS}"
        --pass_entropy_weight "${PASS_ENTROPY_WEIGHT}"
        --pass_ema_decay "${PASS_EMA_DECAY}"
        --pass_min_momentum_constant "${PASS_MIN_MOMENTUM_CONSTANT}"
        --pass_episodic "${PASS_EPISODIC}"
        --pass_use_source_fallback "${PASS_USE_SOURCE_FALLBACK}"
        --pass_image_size "${PASS_IMAGE_SIZE}"
        --pass_adaptor_hidden "${PASS_ADAPTOR_HIDDEN}"
        --pass_perturb_scale "${PASS_PERTURB_SCALE}"
        --pass_prompt_scale "${PASS_PROMPT_SCALE}"
        --pass_prompt_sparsity "${PASS_PROMPT_SPARSITY}"
      )
      if [ -n "${PASS_PROMPT_SIZE}" ]; then
        METHOD_EXTRA+=(--pass_prompt_size "${PASS_PROMPT_SIZE}")
      fi
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 1
      ;;
  esac
}

run_one() {
  local method="$1"
  local data_name="$2"
  local nclass="$3"
  local source="$4"
  local target="$5"
  local ckpt="$6"
  local suffix="$7"
  local expname="${method}_dcon_${suffix}"

  require_ckpt "${ckpt}"
  METHOD_EXTRA=()
  method_args "${method}"

  local cmd=(
    "${PYTHON_BIN}" train.py
    --phase test
    --expname "${expname}"
    --ckpt_dir "${RESULTS_DIR}"
    --dataset "${data_name}"
    --nclass "${nclass}"
    --source "${source}"
    --target "${target}"
    --restore_from "${ckpt}"
    --gpu_ids "${GPU_IDS}"
    --num_workers "${NUM_WORKERS}"
    --save_prediction "${SAVE_PREDICTION}"
    --eval_source_domain "${EVAL_SOURCE_DOMAIN}"
    --tta "${method}"
    --use_cgsd 0
    --use_projector 0
    --use_saam 0
    --use_rccs 0
    "${METHOD_EXTRA[@]}"
  )

  echo "=========================================="
  echo "${method}: ${data_name} ${source}->${target}"
  echo "checkpoint: ${ckpt}"
  echo "expname: ${expname}"
  echo "=========================================="
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
  fi
  echo
}

echo "Project: ${PROJECT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "SAA_DATA_ROOT: ${SAA_DATA_ROOT}"
echo "Results: ${RESULTS_DIR}"
echo "Methods: ${METHODS}"
echo

for method in ${METHODS}; do
  run_one "${method}" "ABDOMINAL" 5 "SABSCT" "CHAOST2" "${CKPT_ROOT}/dcon-sc-300.pth" "sabsct_to_chaost2"
  run_one "${method}" "ABDOMINAL" 5 "CHAOST2" "SABSCT" "${CKPT_ROOT}/dcon-cs-200.pth" "chaost2_to_sabsct"
  run_one "${method}" "CARDIAC" 4 "bSSFP" "LGE" "${CKPT_ROOT}/dcon-bl-1200.pth" "bssfp_to_lge"
  run_one "${method}" "CARDIAC" 4 "LGE" "bSSFP" "${CKPT_ROOT}/dcon-lb-500.pth" "lge_to_bssfp"
done

if [ "${DRY_RUN}" != "1" ]; then
  echo "=========================================="
  echo "Runs completed. Summary:"
  echo "=========================================="
  "${PYTHON_BIN}" scripts/summarize_medseg_tta_dcon.py --roots "${RESULTS_DIR}" --methods ${METHODS} || true
fi
