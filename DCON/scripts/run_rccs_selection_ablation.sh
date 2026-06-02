#!/usr/bin/env bash
set -euo pipefail

# Compact RCCS selection-strategy ablation for SAA.
#
# Default representative task:
#   CARDIAC bSSFP -> LGE
#
# To run all four transfer tasks:
#   TASKS=all bash scripts/run_rccs_selection_ablation.sh
#
# Required in this repo layout:
#   export SAA_DATA_ROOT=/Users/RexRyder/PycharmProjects/Dataset
#   PYTHON_BIN=/path/to/cuda/python bash scripts/run_rccs_selection_ablation.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
SAA_DATA_ROOT="${SAA_DATA_ROOT:-/Users/RexRyder/PycharmProjects/Dataset}"
GPU_IDS="${GPU_IDS:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TASKS="${TASKS:-cardiac_bl}"
SELECTS="${SELECTS:-none random min max}"
CKPT_DIR="${CKPT_DIR:-./ckpts}"
LOG_ROOT="${LOG_ROOT:-./run_logs/rccs_selection_ablation}"
SEED="${SEED:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DCON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DCON_DIR}"

mkdir -p "${LOG_ROOT}"

task_list=()
case "${TASKS}" in
  cardiac_bl)
    task_list=("CARDIAC:bSSFP:LGE:4:18:1800:50:5000:100")
    ;;
  cardiac_lb)
    task_list=("CARDIAC:LGE:bSSFP:4:18:1800:50:5000:100")
    ;;
  abdominal_sc)
    task_list=("ABDOMINAL:SABSCT:CHAOST2:5:3:1500:50:2000:100")
    ;;
  abdominal_cs)
    task_list=("ABDOMINAL:CHAOST2:SABSCT:5:3:1500:50:2000:100")
    ;;
  all)
    task_list=(
      "ABDOMINAL:SABSCT:CHAOST2:5:3:1500:50:2000:100"
      "ABDOMINAL:CHAOST2:SABSCT:5:3:1500:50:2000:100"
      "CARDIAC:bSSFP:LGE:4:18:1800:50:5000:100"
      "CARDIAC:LGE:bSSFP:4:18:1800:50:5000:100"
    )
    ;;
  *)
    echo "Unknown TASKS=${TASKS}. Use cardiac_bl, cardiac_lb, abdominal_sc, abdominal_cs, or all." >&2
    exit 2
    ;;
esac

for task in "${task_list[@]}"; do
  IFS=: read -r data_name source target nclass sgf_grid all_epoch val_freq display_freq save_freq <<< "${task}"

  for select in ${SELECTS}; do
    expname="saa_rccs_select_${select}_${source}_to_${target}"
    stdout_log="${LOG_ROOT}/${expname}.stdout.log"

    echo "================================================================"
    echo "Task: ${data_name} ${source}->${target}"
    echo "RCCS select: ${select}"
    echo "Experiment: ${expname}"
    echo "stdout log: ${stdout_log}"
    echo "train log: ${CKPT_DIR}/${source}/${expname}/log.txt"
    echo "dice log: ${CKPT_DIR}/${source}/${expname}/log/out.csv"
    echo "================================================================"

    SAA_DATA_ROOT="${SAA_DATA_ROOT}" "${PYTHON_BIN}" train.py \
      --use_sgf 1 \
      --sgf_grid_size "${sgf_grid}" \
      --num_workers "${NUM_WORKERS}" \
      --expname "${expname}" \
      --phase train \
      --ckpt_dir "${CKPT_DIR}" \
      --gpu_ids "${GPU_IDS}" \
      --f_seed "${SEED}" \
      --lr 0.0005 \
      --model unet \
      --batchSize 20 \
      --all_epoch "${all_epoch}" \
      --validation_freq "${val_freq}" \
      --display_freq "${display_freq}" \
      --save_freq "${save_freq}" \
      --data_name "${data_name}" \
      --nclass "${nclass}" \
      --tr_domain "${source}" \
      --target_domain "${target}" \
      --save_prediction True \
      --w_ce 1.0 \
      --w_dice 1.0 \
      --w_seg 1.0 \
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
      --rccs_select "${select}" \
      --p_rccs 0.3 \
      --rccs_candidates 4 \
      --rccs_metric cos \
      --rccs_embed_dim 128 \
      2>&1 | tee "${stdout_log}"
  done
done
