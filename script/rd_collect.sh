#!/bin/bash
# Incremental RD data collection for SPCAC
# Usage:
#   bash script/rd_collect.sh                         # all models × datasets × q0-5
#   bash script/rd_collect.sh elpcac                   # one model, all datasets
#   bash script/rd_collect.sh elpcac j8ivfbv2-longdress10
#   bash script/rd_collect.sh elpcac j8ivfbv2-longdress10 3
#   FORCE=1 bash script/rd_collect.sh elpcac          # overwrite existing points
#
# Env:
#   TRAIN_DATASET   default coco3d
#   EPOCH           default las
#   OUT_DIR         default result
#   FORCE           set to 1 to overwrite
#   SPCAC_HF_CHECKPOINT_REPO / HF_TOKEN  optional Hub fallback

set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

TRAIN_DATASET="${TRAIN_DATASET:-coco3d}"
EPOCH="${EPOCH:-las}"
OUT_DIR="${OUT_DIR:-result}"

EXTRA=()
if [ "${FORCE:-0}" = "1" ]; then
    EXTRA+=(--force)
fi

MODEL="${1:-}"
DATASET="${2:-}"
QUALITY="${3:-}"

ARGS=(
    --train_dataset "${TRAIN_DATASET}"
    --epoch "${EPOCH}"
    --out_dir "${OUT_DIR}"
    --verbose
    "${EXTRA[@]}"
)

if [ -n "${MODEL}" ]; then
    ARGS+=(--model "${MODEL}")
fi
if [ -n "${DATASET}" ]; then
    ARGS+=(--dataset "${DATASET}")
fi
if [ -n "${QUALITY}" ]; then
    ARGS+=(--qualities "${QUALITY}")
fi

echo "=============================================="
echo "RD collect: ${ARGS[*]}"
echo "=============================================="

uv run python -m evaluation.rd_collect "${ARGS[@]}"
