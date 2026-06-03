#!/usr/bin/env bash
# Run the FSDA-DG-style disagreement uncertainty weighting ablation only.
#
# This keeps the existing SAA/SAAM feature alignment setup unchanged and
# replaces only the spatial alignment reliability weight:
#   A = W_fsda_uncertainty * M
#
# Common overrides:
#   DRY_RUN=1 bash scripts/run_fsda_uncertainty_ablation.sh
#   ONLY_TASKS="sabsct_to_chaost2 bssfp_to_lge" bash scripts/run_fsda_uncertainty_ablation.sh
#   PYTHON_BIN=/path/to/python bash scripts/run_fsda_uncertainty_ablation.sh
#   FORCE=1 bash scripts/run_fsda_uncertainty_ablation.sh

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

RESULTS_DIR="${RESULTS_DIR:-./results_fsda_uncertainty_ablation}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
GPU_IDS="${GPU_IDS:-0}"
F_SEED="${F_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_PREDICTION="${SAVE_PREDICTION:-True}"
SKIP_FINISHED="${SKIP_FINISHED:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_PREFIX="${RUN_PREFIX:-saam_fsda_uncertainty}"

ALL_TASKS=(
  "ABDOMINAL 5 SABSCT CHAOST2 1500 3 2000 sabsct_to_chaost2"
  "ABDOMINAL 5 CHAOST2 SABSCT 1500 3 2000 chaost2_to_sabsct"
  "CARDIAC 4 bSSFP LGE 1800 18 5000 bssfp_to_lge"
  "CARDIAC 4 LGE bSSFP 1800 18 5000 lge_to_bssfp"
)

task_selected() {
  local suffix="$1"
  local source="$2"
  local target="$3"
  if [[ -z "${ONLY_TASKS:-}" ]]; then
    return 0
  fi
  local token
  for token in ${ONLY_TASKS}; do
    if [[ "${token}" == "${suffix}" || "${token}" == "${source}" || "${token}" == "${source}_to_${target}" ]]; then
      return 0
    fi
  done
  return 1
}

check_runtime() {
  echo "Runtime: ${PYTHON_BIN}"
  "${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this interpreter; this training code calls .cuda().")
PY
}

analyze_args() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import re
import sys
from pathlib import Path

argv = sys.argv[1:]
train_py = Path("train.py").read_text()
declared = set()
for match in re.finditer(r"parser\.add_argument\((.*?)\)", train_py, flags=re.S):
    declared.update(re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", match.group(1)))

passed = []
unknown = []
for token in argv[1:]:
    if not token.startswith("--"):
        continue
    option = token.split("=", 1)[0]
    if option in declared:
        passed.append(option)
    else:
        unknown.append(option)

required = {"--align_weight_type", "--saam_weight_type", "--target_domain"}
missing = sorted(required - set(passed))
if unknown or missing:
    if unknown:
        print("Unknown options: " + ", ".join(sorted(set(unknown))), file=sys.stderr)
    if missing:
        print("Missing required FSDA options: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(2)
PY
}

run_one() {
  local data_name="$1"
  local nclass="$2"
  local source="$3"
  local target="$4"
  local all_epoch="$5"
  local sgf_grid_size="$6"
  local display_freq="$7"
  local suffix="$8"

  local expname="${RUN_PREFIX}_${suffix}"
  local final_snapshot="${RESULTS_DIR}/${source}/${expname}/snapshots/${all_epoch}_net_Seg.pth"
  local log_file="${LOG_DIR}/fsda_uncertainty_${suffix}.log"

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
    --ckpt_dir "${RESULTS_DIR}"
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
    --target_domain "${target}"
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
    --align_weight_type fsda_uncertainty
    --saam_weight_type fsda_uncertainty
    --saam_mask_ablation w_times_m
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
  echo "Task: ${data_name} ${source} -> ${target}"
  echo "Log: ${log_file}"
  echo "Command:"
  printf ' %q' "${cmd[@]}"
  echo
  echo "============================================================"

  analyze_args "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  mkdir -p "${LOG_DIR}"
  "${cmd[@]}" 2>&1 | tee "${log_file}"
}

if [[ "${DRY_RUN}" != "1" && -z "${SAA_DATA_ROOT:-}" ]]; then
  echo "SAA_DATA_ROOT is not set. Export it before running full experiments." >&2
  echo "Example: export SAA_DATA_ROOT=/path/to/data" >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  check_runtime
else
  echo "DRY_RUN=1: checking argument parsing only; skipping dataset and torch runtime checks."
fi

for task in "${ALL_TASKS[@]}"; do
  read -r data_name nclass source target all_epoch sgf_grid_size display_freq suffix <<< "${task}"
  if ! task_selected "${suffix}" "${source}" "${target}"; then
    continue
  fi
  run_one "${data_name}" "${nclass}" "${source}" "${target}" "${all_epoch}" "${sgf_grid_size}" "${display_freq}" "${suffix}"
done

echo "FSDA uncertainty ablation script completed."
