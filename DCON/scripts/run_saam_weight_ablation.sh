#!/usr/bin/env bash
# Run SAAM weight-source ablations with the same feature-alignment setup.
#
# Defaults:
# - skips the already-run stability baseline
# - runs entropy/confidence with anchor_only and tri_view_mean
# - runs uniform and foreground_only once each
# - skips a run when the final snapshot already exists
#
# Common overrides:
#   PYTHON_BIN=/path/to/python bash scripts/run_saam_weight_ablation.sh
#   ONLY_TASKS="CARDIAC:bSSFP ABDOMINAL:SABSCT" bash scripts/run_saam_weight_ablation.sh
#   ONLY_VARIANTS="entropy:anchor_only confidence:anchor_only foreground_only:anchor_only" bash scripts/run_saam_weight_ablation.sh
#   INCLUDE_STABILITY=1 bash scripts/run_saam_weight_ablation.sh
#   DRY_RUN=1 bash scripts/run_saam_weight_ablation.sh
#   FORCE=1 bash scripts/run_saam_weight_ablation.sh

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
INCLUDE_STABILITY="${INCLUDE_STABILITY:-0}"
RUN_PREFIX="${RUN_PREFIX:-saam_weight_ablation}"

DEFAULT_VARIANTS=(
  "entropy:anchor_only"
  "entropy:tri_view_mean"
  "confidence:anchor_only"
  "confidence:tri_view_mean"
  "uniform:anchor_only"
  "foreground_only:anchor_only"
)

if [[ "${INCLUDE_STABILITY}" == "1" ]]; then
  DEFAULT_VARIANTS=("stability:anchor_only" "${DEFAULT_VARIANTS[@]}")
fi

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

check_runtime() {
  echo "Runtime: ${PYTHON_BIN}"
  "${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"Cannot import torch in this interpreter: {exc}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY
}

analyze_args() {
  local weight_type="$1"
  local view_mode="$2"
  shift 2

  "${PYTHON_BIN}" - "${weight_type}" "${view_mode}" "$@" <<'PY'
import re
import sys
from pathlib import Path

weight_type = sys.argv[1]
view_mode = sys.argv[2]
argv = sys.argv[3:]
train_py = Path("train.py").read_text()
code_text = "\n".join(
    path.read_text(errors="ignore")
    for path in Path(".").rglob("*.py")
    if "__pycache__" not in path.parts
)

declared = {}
for match in re.finditer(r"parser\.add_argument\((.*?)\)", train_py, flags=re.S):
    body = match.group(1)
    opts = re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", body)
    if not opts:
        continue
    dest = re.search(r"dest\s*=\s*['\"]([A-Za-z0-9_]+)['\"]", body)
    name = dest.group(1) if dest else opts[0][2:].replace("-", "_")
    for opt in opts:
        declared[opt] = name

passed = []
unknown = []
for token in argv:
    if not token.startswith("--"):
        continue
    opt = token.split("=", 1)[0]
    if opt in declared:
        passed.append((opt, declared[opt]))
    else:
        unknown.append(opt)

used_after_parse = set(re.findall(r"\b(?:opt|args)\.([A-Za-z0-9_]+)\b", code_text))
used_after_parse.update(re.findall(r"\bself\.opt\.([A-Za-z0-9_]+)\b", code_text))
used_after_parse.update(
    re.findall(r"\b(?:getattr|hasattr)\(\s*(?:self\.opt|opt|args)\s*,\s*['\"]([A-Za-z0-9_]+)['\"]", code_text)
)
declared_but_unreferenced = sorted({name for name in declared.values() if name not in used_after_parse})
passed_but_unreferenced = sorted({name for _, name in passed if name not in used_after_parse})

effective_unused = []
if weight_type == "stability":
    effective_unused += ["uncertainty_tau", "uncertainty_view_mode"]
else:
    effective_unused += ["saam_tau", "saam_topk", "saam_stability_mode"]
if weight_type == "confidence":
    effective_unused += ["uncertainty_tau"]
if weight_type in {"uniform", "foreground_only"}:
    effective_unused += ["uncertainty_tau", "uncertainty_view_mode"]
if view_mode == "anchor_only":
    effective_unused += ["base/strong logits for uncertainty weighting"]

print("Parameter analysis:")
print(f"  saam_weight_type={weight_type}")
print(f"  uncertainty_view_mode={view_mode}")
print(f"  passed_options={len(passed)} unknown_options={len(unknown)}")
if unknown:
    print("  unknown_options: " + ", ".join(sorted(set(unknown))))
if passed_but_unreferenced:
    print("  passed_but_not_referenced_by_train_py: " + ", ".join(passed_but_unreferenced))
print("  effectively_unused_for_this_variant: " + ", ".join(dict.fromkeys(effective_unused)))
if declared_but_unreferenced:
    print("  declared_but_not_referenced_by_train_py: " + ", ".join(declared_but_unreferenced))
PY
}

run_one() {
  local data_name="$1"
  local nclass="$2"
  local source="$3"
  local all_epoch="$4"
  local sgf_grid_size="$5"
  local display_freq="$6"
  local weight_type="$7"
  local view_mode="$8"

  local expname="${RUN_PREFIX}_${weight_type}_${view_mode}_${source}"
  if [[ "${weight_type}" == "uniform" || "${weight_type}" == "foreground_only" ]]; then
    expname="${RUN_PREFIX}_${weight_type}_${source}"
  fi

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
    --saam_weight_type "${weight_type}"
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
  echo "Task: ${data_name} source=${source} epochs=${all_epoch}"
  analyze_args "${weight_type}" "${view_mode}" "${cmd[@]:1}"
  echo "Command:"
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
  echo "Example: export SAA_DATA_ROOT=/path/to/data" >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  check_runtime
else
  echo "DRY_RUN=1: skipping dataset and torch runtime checks."
fi

for task in "${ALL_TASKS[@]}"; do
  read -r data_name nclass source all_epoch sgf_grid_size display_freq <<< "${task}"
  if ! task_selected "${data_name}" "${source}"; then
    continue
  fi
  for variant in "${VARIANTS[@]}"; do
    IFS=: read -r weight_type view_mode <<< "${variant}"
    view_mode="${view_mode:-anchor_only}"
    run_one "${data_name}" "${nclass}" "${source}" "${all_epoch}" "${sgf_grid_size}" "${display_freq}" "${weight_type}" "${view_mode}"
  done
done

echo "All requested SAAM weight-source ablations finished."
