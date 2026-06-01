#!/usr/bin/env bash
# Run the foreground-prior isolation ablation for SAAM:
#   Uniform align: A=1
#   M only:        A=M
#   W only:        A=W
#   W x M:         A=W*M
#
# Common overrides:
#   PYTHON_BIN=/path/to/python bash scripts/run_saam_mask_ablation.sh
#   ONLY_TASKS="CARDIAC:bSSFP ABDOMINAL:SABSCT" bash scripts/run_saam_mask_ablation.sh
#   ONLY_VARIANTS="m_only w_only" bash scripts/run_saam_mask_ablation.sh
#   DRY_RUN=1 bash scripts/run_saam_mask_ablation.sh
#   FORCE=1 bash scripts/run_saam_mask_ablation.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DCON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DCON_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No python interpreter found. Set PYTHON_BIN=/path/to/python." >&2
    exit 2
  fi
fi

GPU_IDS="${GPU_IDS:-0}"
CKPT_DIR="${CKPT_DIR:-./ckpts}"
F_SEED="${F_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_PREDICTION="${SAVE_PREDICTION:-True}"
SKIP_FINISHED="${SKIP_FINISHED:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_PREFIX="${RUN_PREFIX:-saam_mask_ablation}"

DEFAULT_VARIANTS=(uniform_align m_only w_only w_times_m)
if [[ -n "${ONLY_VARIANTS:-}" ]]; then
  read -r -a VARIANTS <<< "${ONLY_VARIANTS}"
else
  VARIANTS=("${DEFAULT_VARIANTS[@]}")
fi

ALL_TASKS=(
  "CARDIAC 4 bSSFP 1800 18 5000"
  "CARDIAC 4 LGE 1800 18 5000"
  "ABDOMINAL 5 SABSCT 1500 3 2000"
  "ABDOMINAL 5 CHAOST2 1500 3 2000"
)

task_selected() {
  local data_name="$1"
  local source="$2"
  if [[ -z "${ONLY_TASKS:-}" ]]; then
    return 0
  fi
  local token
  for token in ${ONLY_TASKS}; do
    if [[ "${token}" == "${data_name}:${source}" || "${token}" == "${source}" || "${token}" == "${data_name}" ]]; then
      return 0
    fi
  done
  return 1
}

variant_label() {
  case "$1" in
    uniform_align) echo "Uniform align" ;;
    m_only) echo "M only" ;;
    w_only) echo "W only" ;;
    w_times_m) echo "W x M" ;;
    *) echo "$1" ;;
  esac
}

check_runtime() {
  echo "Runtime: ${PYTHON_BIN}"
  "${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY
}

run_one() {
  local data_name="$1"
  local nclass="$2"
  local source="$3"
  local all_epoch="$4"
  local sgf_grid_size="$5"
  local display_freq="$6"
  local variant="$7"

  local expname="${RUN_PREFIX}_${variant}_${source}"
  local final_snapshot="${CKPT_DIR}/${source}/${expname}/snapshots/${all_epoch}_net_Seg.pth"
  if [[ "${FORCE}" != "1" && "${SKIP_FINISHED}" == "1" && -f "${final_snapshot}" ]]; then
    echo "[skip] ${expname}: found ${final_snapshot}"
    return 0
  fi

  local -a cmd=(
    "${PYTHON_BIN}" train.py
    --use_sgf 1
    --sgf_grid_size "${sgf_grid_size}"
    --num_workers "${NUM_WORKERS}"
    --quiet_console True
    --expname "${expname}"
    --phase train
    --ckpt_dir "${CKPT_DIR}"
    --gpu_ids "${GPU_IDS}"
    --f_seed "${F_SEED}"
    --lr 0.0005
    --model unet
    --batchSize "${BATCH_SIZE}"
    --all_epoch "${all_epoch}"
    --validation_freq 50
    --display_freq "${display_freq}"
    --save_freq 100
    --data_name "${data_name}"
    --nclass "${nclass}"
    --tr_domain "${source}"
    --save_prediction "${SAVE_PREDICTION}"
    --w_ce 1.0
    --w_dice 1.0
    --w_seg 1.0
    --use_cgsd 1
    --cgsd_layer 1
    --use_projector 1
    --use_separate_cgsd_optimizer 1
    --lambda_str 0.3
    --lambda_sty 0.3
    --use_saam 1
    --saam_tau 0.5
    --saam_topk 0.3
    --saam_stability_mode mean
    --saam_weight_type stability
    --saam_mask_ablation "${variant}"
    --lambda_01 1.0
    --lambda_02 1.0
    --saam_warmup_epochs 50
    --saam_rampup_epochs 100
    --anchor_seg_alpha 0.0
    --strong_seg_alpha 1.0
    --use_rccs 1
    --p_rccs 0.3
    --rccs_candidates 4
    --rccs_metric cos
    --rccs_embed_dim 128
  )

  echo "============================================================"
  echo "Run: ${expname}"
  echo "Table row: $(variant_label "${variant}")"
  echo "Task: ${data_name} source=${source} epochs=${all_epoch}"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  echo
  echo "============================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

if [[ "${DRY_RUN}" != "1" && -z "${SAA_DATA_ROOT:-}" ]]; then
  echo "SAA_DATA_ROOT is not set. Export it before running full experiments." >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  check_runtime
fi

for task in "${ALL_TASKS[@]}"; do
  read -r data_name nclass source all_epoch sgf_grid_size display_freq <<< "${task}"
  if ! task_selected "${data_name}" "${source}"; then
    continue
  fi
  for variant in "${VARIANTS[@]}"; do
    run_one "${data_name}" "${nclass}" "${source}" "${all_epoch}" "${sgf_grid_size}" "${display_freq}" "${variant}"
  done
done

echo "SAAM foreground-prior isolation ablation completed."
