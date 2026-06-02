#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/compare_lla_clp}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET="${DATASET:-ABDOMINAL}"
SOURCE="${SOURCE:-CHAOST2}"
TARGET="${TARGET:-SABSCT}"
NCLASS="${NCLASS:-5}"
GPU="${GPU:-0}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_ITERS="${MAX_ITERS:-0}"

if [[ -z "${SAA_DATA_ROOT:-}" && -d "${ROOT_DIR}/../SLAug/data" ]]; then
  export SAA_DATA_ROOT="${ROOT_DIR}/../SLAug/data"
fi

mkdir -p "${OUT_DIR}/logs"
rm -f "${OUT_DIR}/aug_time_per_batch.csv" \
      "${OUT_DIR}/train_time_per_epoch.csv" \
      "${OUT_DIR}/summary.md"

run_one() {
  local aug_type="$1"
  local method="$2"
  local expname="compare_${aug_type}_${DATASET}_${SOURCE}_to_${TARGET}_seed${SEED}"
  local log_file="${OUT_DIR}/logs/${aug_type}.log"

  echo "[compare] method=${method} local_aug_type=${aug_type}"
  "${PYTHON_BIN}" train.py \
    --profile_cost 1 \
    --profile_method "${method}" \
    --profile_output_dir "${OUT_DIR}" \
    --profile_max_iters "${MAX_ITERS}" \
    --local_aug_type "${aug_type}" \
    --expname "${expname}" \
    --phase train \
    --ckpt_dir "${OUT_DIR}/ckpts" \
    --gpu_ids "${GPU}" \
    --f_seed "${SEED}" \
    --lr 0.0005 \
    --model unet \
    --batchSize "${BATCH_SIZE}" \
    --all_epoch "${EPOCHS}" \
    --validation_freq 999999 \
    --display_freq 999999 \
    --save_freq 999999 \
    --data_name "${DATASET}" \
    --nclass "${NCLASS}" \
    --tr_domain "${SOURCE}" \
    --target_domain "${TARGET}" \
    --save_prediction False \
    --eval_source_domain False \
    --num_workers "${NUM_WORKERS}" \
    --use_sgf 0 \
    --use_cgsd 0 \
    --use_projector 0 \
    --use_saam 0 \
    --use_rccs 0 \
    --quiet_console True \
    2>&1 | tee "${log_file}"
}

cd "${ROOT_DIR}"
run_one "lla" "LLA"
run_one "clp" "CLP"
run_one "none" "none"

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import csv
import math
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
aug_path = out_dir / "aug_time_per_batch.csv"
train_path = out_dir / "train_time_per_epoch.csv"
summary_path = out_dir / "summary.md"
methods = ["none", "LLA", "CLP"]

def mean(values):
    vals = [float(v) for v in values if v not in ("", "nan", "NaN", None)]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")

aug = {m: [] for m in methods}
if aug_path.exists():
    with aug_path.open() as f:
        for row in csv.DictReader(f):
            if row["method"] in aug:
                aug[row["method"]].append(row["aug_time_ms"])

train = {m: {"time": [], "mem": [], "dice": []} for m in methods}
if train_path.exists():
    with train_path.open() as f:
        for row in csv.DictReader(f):
            m = row["method"]
            if m in train:
                train[m]["time"].append(row["train_time_sec"])
                train[m]["mem"].append(row["peak_gpu_mem_gb"])
                train[m]["dice"].append(row["dice"])

def fmt(value, digits=4):
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"

lines = [
    "# LLA vs CLP Profiling Summary",
    "",
    "| Method | Aug time / batch | Train time / epoch | Peak memory | Dice |",
    "| ------ | ---------------: | -----------------: | ----------: | ---: |",
]
for m in methods:
    lines.append(
        f"| {m} | {fmt(mean(aug[m]), 3)} ms | "
        f"{fmt(mean(train[m]['time']), 3)} sec | "
        f"{fmt(mean(train[m]['mem']), 3)} GB | "
        f"{fmt(mean(train[m]['dice']), 4)} |"
    )
lines.extend([
    "",
    f"- Raw augmentation timing: `{aug_path}`",
    f"- Raw epoch timing: `{train_path}`",
])
summary_path.write_text("\n".join(lines) + "\n")
print(f"[compare] summary written to {summary_path}")
PY
