#!/bin/bash
# Conservative SAAM-SPMM run settings.
#
# This profile is intended for stability-first comparison against SM-PPM/source
# baselines. It keeps the method source-prototype based, but relaxes stable-mask
# gating and weakens prototype/consistency constraints to reduce harmful
# adaptation on target slices.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SAAM_SPMM_ABLATION_MODE="${SAAM_SPMM_ABLATION_MODE:-full}"
export NUM_VIEWS="${NUM_VIEWS:-2}"
export SAAM_SPMM_STEPS="${SAAM_SPMM_STEPS:-1}"
export SAAM_SPMM_UPDATE_SCOPE="${SAAM_SPMM_UPDATE_SCOPE:-bn_affine}"

# Less rigid stable-region selection than the original full setting.
export STABLE_TOPK_PERCENT="${STABLE_TOPK_PERCENT:-0.7}"
export UNSTABLE_WEIGHT="${UNSTABLE_WEIGHT:-0.3}"

# Softer auxiliary constraints; entropy remains the main target signal.
export LAMBDA_ENT="${LAMBDA_ENT:-1.0}"
export LAMBDA_PROTO="${LAMBDA_PROTO:-0.3}"
export LAMBDA_SHAPE="${LAMBDA_SHAPE:-0.05}"
export LAMBDA_CONS="${LAMBDA_CONS:-0.2}"

export NUM_WORKERS="${NUM_WORKERS:-0}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
export SAVE_PREDICTION="${SAVE_PREDICTION:-false}"
export EVAL_SOURCE_DOMAIN="${EVAL_SOURCE_DOMAIN:-false}"
export QUIET_CONSOLE="${QUIET_CONSOLE:-true}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export SKIP_FINISHED="${SKIP_FINISHED:-false}"

bash "${SCRIPT_DIR}/run_saam_spmm.sh"
