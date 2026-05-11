#!/bin/bash
# Run SAAM-SPMM ablations on DCON source-only checkpoints.
#
# Ablation modes:
#   baseline: existing SM-PPM full path, unchanged
#   saam: SAAM stability estimation only
#   stable: SAAM + stable-region weighted adaptation
#   anchor: SAAM + stable mask + source prototype anchor
#   full: SAAM + stable mask + source prototype anchor + shape consistency

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi

GPU_IDS=${GPU_IDS:-0}
NUM_WORKERS=${NUM_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-1}
RESULTS_DIR=${RESULTS_DIR:-results}
PROTO_ROOT=${PROTO_ROOT:-ckpts}
SAVE_PREDICTION=${SAVE_PREDICTION:-false}
EVAL_SOURCE_DOMAIN=${EVAL_SOURCE_DOMAIN:-false}
QUIET_CONSOLE=${QUIET_CONSOLE:-true}
EXPORT_PROTOTYPES=${EXPORT_PROTOTYPES:-true}

SAAM_SPMM_LR=${SAAM_SPMM_LR:-1e-4}
SAAM_SPMM_STEPS=${SAAM_SPMM_STEPS:-1}
SAAM_SPMM_UPDATE_SCOPE=${SAAM_SPMM_UPDATE_SCOPE:-bn_affine}
NUM_VIEWS=${NUM_VIEWS:-5}
SAAM_METRIC=${SAAM_METRIC:-variance}
STABLE_TOPK_PERCENT=${STABLE_TOPK_PERCENT:-0.3}
UNSTABLE_WEIGHT=${UNSTABLE_WEIGHT:-0.1}
LAMBDA_ENT=${LAMBDA_ENT:-1.0}
LAMBDA_PROTO=${LAMBDA_PROTO:-1.0}
LAMBDA_SHAPE=${LAMBDA_SHAPE:-0.1}
LAMBDA_CONS=${LAMBDA_CONS:-1.0}
PROTO_MOMENTUM=${PROTO_MOMENTUM:-0.9}
PROTO_LOSS=${PROTO_LOSS:-cosine}
LOG_INTERVAL=${LOG_INTERVAL:-1}

SAAM_SPMM_ABLATION_MODE=${SAAM_SPMM_ABLATION_MODE:-all}
if [ "${SAAM_SPMM_ABLATION_MODE}" = "all" ]; then
  ABLATION_MODES=(baseline saam stable anchor full)
else
  read -r -a ABLATION_MODES <<< "${SAAM_SPMM_ABLATION_MODE}"
fi

prototype_path() {
  local data_name=$1
  local source=$2
  echo "${PROTO_ROOT}/${source}/source_prototypes_${data_name}_${source}.pt"
}

export_prototypes() {
  local data_name=$1
  local nclass=$2
  local source=$3
  local ckpt=$4
  local proto_path
  proto_path="$(prototype_path "${data_name}" "${source}")"

  if [ "${EXPORT_PROTOTYPES}" != "true" ] && [ -f "${proto_path}" ]; then
    return 0
  fi

  echo "=========================================="
  echo "Export source prototypes: ${data_name}, source=${source}, ckpt=${ckpt}"
  echo "Output: ${proto_path}"
  echo "=========================================="

  "${PYTHON_BIN}" train.py \
    --phase export_source_prototypes \
    --expname "export_proto_${data_name}_${source}" \
    --ckpt_dir "${RESULTS_DIR}" \
    --dataset "${data_name}" \
    --nclass "${nclass}" \
    --source "${source}" \
    --restore_from "${ckpt}" \
    --source_prototype_path "${proto_path}" \
    --gpu_ids "${GPU_IDS}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor "${PREFETCH_FACTOR}" \
    --smppm_src_batch_size 2 \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0
}

run_one() {
  local mode=$1
  local expname_base=$2
  local data_name=$3
  local nclass=$4
  local source=$5
  local target=$6
  local ckpt=$7
  local proto_path
  proto_path="$(prototype_path "${data_name}" "${source}")"

  echo "=========================================="
  echo "SAAM-SPMM mode=${mode}: ${data_name}, source=${source}, target=${target}, ckpt=${ckpt}"
  echo "=========================================="

  if [ "${mode}" = "baseline" ]; then
    "${PYTHON_BIN}" train.py \
      --phase test \
      --expname "baseline_smppm_${expname_base}" \
      --ckpt_dir "${RESULTS_DIR}" \
      --dataset "${data_name}" \
      --nclass "${nclass}" \
      --source "${source}" \
      --target "${target}" \
      --restore_from "${ckpt}" \
      --gpu_ids "${GPU_IDS}" \
      --num_workers "${NUM_WORKERS}" \
      --prefetch_factor "${PREFETCH_FACTOR}" \
      --save_prediction "${SAVE_PREDICTION}" \
      --eval_source_domain "${EVAL_SOURCE_DOMAIN}" \
      --quiet_console "${QUIET_CONSOLE}" \
      --tta sm_ppm \
      --smppm_ablation_mode full \
      --smppm_steps "${SAAM_SPMM_STEPS}" \
      --smppm_log_interval "${LOG_INTERVAL}" \
      --use_cgsd 0 \
      --use_projector 0 \
      --use_saam 0 \
      --use_rccs 0
    return 0
  fi

  local use_stable_mask=0
  local use_source_anchor=0
  local use_shape_consistency=0
  case "${mode}" in
    saam)
      ;;
    stable)
      use_stable_mask=1
      ;;
    anchor)
      use_stable_mask=1
      use_source_anchor=1
      ;;
    full)
      use_stable_mask=1
      use_source_anchor=1
      use_shape_consistency=1
      ;;
    *)
      echo "Unknown SAAM_SPMM_ABLATION_MODE: ${mode}" >&2
      return 2
      ;;
  esac

  if [ "${use_source_anchor}" = "1" ] && [ ! -f "${proto_path}" ]; then
    echo "Missing source prototype file: ${proto_path}" >&2
    echo "Run with EXPORT_PROTOTYPES=true or export it manually first." >&2
    return 2
  fi

  "${PYTHON_BIN}" train.py \
    --phase test \
    --expname "saam_spmm_${mode}_${expname_base}" \
    --ckpt_dir "${RESULTS_DIR}" \
    --dataset "${data_name}" \
    --nclass "${nclass}" \
    --source "${source}" \
    --target "${target}" \
    --restore_from "${ckpt}" \
    --source_prototype_path "${proto_path}" \
    --gpu_ids "${GPU_IDS}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor "${PREFETCH_FACTOR}" \
    --save_prediction "${SAVE_PREDICTION}" \
    --eval_source_domain "${EVAL_SOURCE_DOMAIN}" \
    --quiet_console "${QUIET_CONSOLE}" \
    --tta saam_spmm \
    --saam_spmm_lr "${SAAM_SPMM_LR}" \
    --saam_spmm_update_scope "${SAAM_SPMM_UPDATE_SCOPE}" \
    --smppm_steps "${SAAM_SPMM_STEPS}" \
    --num_views "${NUM_VIEWS}" \
    --use_saam 1 \
    --use_stable_mask "${use_stable_mask}" \
    --use_source_anchor "${use_source_anchor}" \
    --use_shape_consistency "${use_shape_consistency}" \
    --saam_metric "${SAAM_METRIC}" \
    --stable_topk_percent "${STABLE_TOPK_PERCENT}" \
    --unstable_weight "${UNSTABLE_WEIGHT}" \
    --lambda_ent "${LAMBDA_ENT}" \
    --lambda_proto "${LAMBDA_PROTO}" \
    --lambda_shape "${LAMBDA_SHAPE}" \
    --lambda_cons "${LAMBDA_CONS}" \
    --proto_momentum "${PROTO_MOMENTUM}" \
    --proto_loss "${PROTO_LOSS}" \
    --smppm_log_interval "${LOG_INTERVAL}" \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_rccs 0
}

pairs=(
  "dcon_sc_chaost2 ABDOMINAL 5 SABSCT CHAOST2 ../ckpts/dcon-sc-300.pth"
  "dcon_cs_sabsct ABDOMINAL 5 CHAOST2 SABSCT ../ckpts/dcon-cs-200.pth"
  "dcon_bl_lge CARDIAC 4 bSSFP LGE ../ckpts/dcon-bl-1200.pth"
  "dcon_lb_bssfp CARDIAC 4 LGE bSSFP ../ckpts/dcon-lb-500.pth"
)

if [ "${EXPORT_PROTOTYPES}" = "true" ]; then
  for pair in "${pairs[@]}"; do
    read -r expname data_name nclass source target ckpt <<< "${pair}"
    export_prototypes "${data_name}" "${nclass}" "${source}" "${ckpt}"
  done
fi

for mode in "${ABLATION_MODES[@]}"; do
  for pair in "${pairs[@]}"; do
    read -r expname data_name nclass source target ckpt <<< "${pair}"
    run_one "${mode}" "${expname}" "${data_name}" "${nclass}" "${source}" "${target}" "${ckpt}"
  done
done

echo "=========================================="
echo "SAAM-SPMM DCON evaluations completed."
echo "=========================================="
