#!/usr/bin/env bash
set -euo pipefail

# Compare original SLAug local augmentation (LLA: Bezier location-scale)
# against affine CLP in the same SLAug training pipeline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/results/compare_slaug_affine_clp/ckpts}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/results/compare_slaug_affine_clp/logs}"
METHODS="${METHODS:-lla clp}"

if [[ -z "${SAA_DATA_ROOT:-}" && -d "${ROOT_DIR}/../SLAug/data" ]]; then
  export SAA_DATA_ROOT="${ROOT_DIR}/../SLAug/data"
fi

mkdir -p "${CKPT_DIR}" "${LOG_ROOT}"

run_one() {
  local task="$1"
  local local_aug_type="$2"
  local dataset nclass source target epochs grid_size display_freq save_freq method_tag expname log_file

  case "${task}" in
    bl)
      dataset="CARDIAC"
      nclass="4"
      source="bSSFP"
      target="LGE"
      epochs="600"
      grid_size="18"
      display_freq="5000"
      save_freq="100"
      ;;
    cs)
      dataset="ABDOMINAL"
      nclass="5"
      source="CHAOST2"
      target="SABSCT"
      epochs="800"
      grid_size="3"
      display_freq="2000"
      save_freq="100"
      ;;
    *)
      echo "Unknown task: ${task}" >&2
      exit 1
      ;;
  esac

  if [[ "${local_aug_type}" == "lla" ]]; then
    method_tag="lla_bezier"
  elif [[ "${local_aug_type}" == "clp" ]]; then
    method_tag="affine_clp"
  else
    echo "Unknown local_aug_type: ${local_aug_type}" >&2
    exit 1
  fi

  expname="slaug_${method_tag}_${task}_${source}_to_${target}_${epochs}ep_seed${SEED}"
  log_file="${LOG_ROOT}/${expname}.log"

  echo "================================================================"
  echo "SLAug compare: task=${task} dataset=${dataset} ${source}->${target}"
  echo "local_aug_type=${local_aug_type} epochs=${epochs} expname=${expname}"
  echo "================================================================"

  "${PYTHON_BIN}" train.py \
    --local_aug_type "${local_aug_type}" \
    --expname "${expname}" \
    --phase train \
    --ckpt_dir "${CKPT_DIR}" \
    --gpu_ids "${GPU}" \
    --f_seed "${SEED}" \
    --lr 0.0005 \
    --model unet \
    --batchSize "${BATCH_SIZE}" \
    --all_epoch "${epochs}" \
    --validation_freq 50 \
    --display_freq "${display_freq}" \
    --save_freq "${save_freq}" \
    --data_name "${dataset}" \
    --nclass "${nclass}" \
    --tr_domain "${source}" \
    --target_domain "${target}" \
    --save_prediction False \
    --eval_source_domain False \
    --num_workers "${NUM_WORKERS}" \
    --use_sgf 1 \
    --sgf_grid_size "${grid_size}" \
    --use_cgsd 1 \
    --cgsd_layer 1 \
    --use_projector 1 \
    --use_separate_cgsd_optimizer 1 \
    --lambda_str 0.3 \
    --lambda_sty 0.3 \
    --use_saam 1 \
    --saam_tau 0.5 \
    --saam_topk 0.3 \
    --saam_stability_mode mean \
    --lambda_01 1.0 \
    --lambda_02 1.0 \
    --saam_warmup_epochs 50 \
    --saam_rampup_epochs 100 \
    --anchor_seg_alpha 0.0 \
    --strong_seg_alpha 1.0 \
    --use_rccs 1 \
    --p_rccs 0.3 \
    --rccs_candidates 4 \
    --rccs_metric cos \
    --rccs_embed_dim 128 \
    2>&1 | tee "${log_file}"
}

cd "${ROOT_DIR}"

for task in bl cs; do
  for method in ${METHODS}; do
    run_one "${task}" "${method}"
  done
done

echo "Done. Logs: ${LOG_ROOT}"
echo "Checkpoints/results: ${CKPT_DIR}"
