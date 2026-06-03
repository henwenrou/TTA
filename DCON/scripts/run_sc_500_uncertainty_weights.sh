#!/usr/bin/env bash
# Run SABSCT -> CHAOST2 500-epoch uncertainty-weight baselines only:
#   1. entropy(anchor_only)
#   2. confidence(anchor_only)
#   3. FSDA-DG-style disagreement uncertainty
#
# Common overrides:
#   DRY_RUN=1 bash scripts/run_sc_500_uncertainty_weights.sh
#   PYTHON_BIN=/path/to/cuda/python bash scripts/run_sc_500_uncertainty_weights.sh
#   SAA_DATA_ROOT=/path/to/data bash scripts/run_sc_500_uncertainty_weights.sh
#   ONLY_VARIANTS="entropy confidence fsda_uncertainty" bash scripts/run_sc_500_uncertainty_weights.sh
#   FORCE=1 bash scripts/run_sc_500_uncertainty_weights.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DCON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DCON_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/tpsdg/bin/python}"
export SAA_DATA_ROOT="${SAA_DATA_ROOT:-data}"

RESULTS_DIR="${RESULTS_DIR:-./results_sc_500_uncertainty_weights}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
GPU_IDS="${GPU_IDS:-0}"
F_SEED="${F_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_PREDICTION="${SAVE_PREDICTION:-True}"
SKIP_FINISHED="${SKIP_FINISHED:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-500}"

DEFAULT_VARIANTS=(entropy confidence fsda_uncertainty)
if [[ -n "${ONLY_VARIANTS:-}" ]]; then
  read -r -a VARIANTS <<< "${ONLY_VARIANTS}"
else
  VARIANTS=("${DEFAULT_VARIANTS[@]}")
fi

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

unknown = []
passed = []
for token in argv[1:]:
    if not token.startswith("--"):
        continue
    option = token.split("=", 1)[0]
    passed.append(option)
    if option not in declared:
        unknown.append(option)

required = {"--align_weight_type", "--saam_weight_type", "--target_domain", "--all_epoch"}
missing = sorted(required - set(passed))
if unknown or missing:
    if unknown:
        print("Unknown options: " + ", ".join(sorted(set(unknown))), file=sys.stderr)
    if missing:
        print("Missing required options: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(2)
PY
}

run_one() {
  local variant="$1"
  local expname=""
  local log_file=""
  local view_mode="anchor_only"

  case "${variant}" in
    entropy)
      expname="sc_500_entropy_anchor_only"
      log_file="${LOG_DIR}/sc_500_entropy_anchor_only.log"
      ;;
    confidence)
      expname="sc_500_confidence_anchor_only"
      log_file="${LOG_DIR}/sc_500_confidence_anchor_only.log"
      ;;
    fsda_uncertainty)
      expname="sc_500_fsda_uncertainty"
      log_file="${LOG_DIR}/sc_500_fsda_uncertainty.log"
      ;;
    *)
      echo "Unknown variant: ${variant}. Use entropy, confidence, or fsda_uncertainty." >&2
      exit 2
      ;;
  esac

  local final_snapshot="${RESULTS_DIR}/SABSCT/${expname}/snapshots/${EPOCHS}_net_Seg.pth"
  if [[ "${FORCE}" != "1" && "${SKIP_FINISHED}" == "1" && -f "${final_snapshot}" ]]; then
    echo "[skip] ${expname}: found ${final_snapshot}"
    return 0
  fi

  local -a cmd=(
    "${PYTHON_BIN}" train.py
    --use_sgf 1
    --sgf_grid_size 3
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
    --all_epoch "${EPOCHS}"
    --validation_freq 50
    --display_freq 2000
    --save_freq 100
    --data_name ABDOMINAL
    --nclass 5
    --tr_domain SABSCT
    --target_domain CHAOST2
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
    --align_weight_type "${variant}"
    --saam_weight_type "${variant}"
    --saam_mask_ablation w_times_m
    --uncertainty_tau 0.5
    --uncertainty_view_mode "${view_mode}"
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
  echo "Task: ABDOMINAL SABSCT -> CHAOST2"
  echo "Variant: ${variant}"
  echo "Epochs: ${EPOCHS}"
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

if [[ "${DRY_RUN}" != "1" ]]; then
  check_runtime
else
  echo "DRY_RUN=1: checking argument parsing only; skipping dataset and torch runtime checks."
fi

for variant in "${VARIANTS[@]}"; do
  run_one "${variant}"
done

echo "SABSCT -> CHAOST2 500-epoch uncertainty-weight runs completed."
