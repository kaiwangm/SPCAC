#!/bin/bash
# Export original / reconstructed clouds for preview/app.py (full resolution by default)
# Usage:
#   bash script/export_previews.sh                         # default 8iVFBv2 × all models
#   bash script/export_previews.sh j8ivfbv2-longdress10
#   bash script/export_previews.sh j8ivfbv2-longdress10 elpcac 3
#   SKIP_INFER=1 bash script/export_previews.sh            # original only
#
# Env:
#   TRAIN_DATASET   default coco3d
#   EPOCH           default las
#   MAX_POINTS      default 0 (full cloud; set e.g. 60000 to downsample)
#   FRAME           default 0
#   FRAME_END       inclusive end index, e.g. FRAME_END=69 exports 0–69
#   SKIP_INFER      set to 1 to skip GPU reconstruction

set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TRAIN_DATASET="${TRAIN_DATASET:-coco3d}"
EPOCH="${EPOCH:-las}"
MAX_POINTS="${MAX_POINTS:-0}"
FRAME="${FRAME:-0}"
FRAME_END="${FRAME_END:-}"

DATASET="${1:-}"
MODEL="${2:-}"
QUALITY="${3:-}"

ARGS=(
    --train_dataset "${TRAIN_DATASET}"
    --epoch "${EPOCH}"
    --max_points "${MAX_POINTS}"
    --frame "${FRAME}"
)
if [ -n "${FRAME_END}" ]; then
    ARGS+=(--frame_end "${FRAME_END}")
fi

if [ "${SKIP_INFER:-0}" = "1" ]; then
    ARGS+=(--skip_infer)
fi
if [ "${FORCE:-0}" = "1" ]; then
    ARGS+=(--force)
fi
if [ -n "${DATASET}" ]; then
    ARGS+=(--dataset "${DATASET}")
fi
if [ -n "${MODEL}" ]; then
    ARGS+=(--model "${MODEL}")
fi
if [ -n "${QUALITY}" ]; then
    ARGS+=(--qualities "${QUALITY}")
fi

echo "=============================================="
echo "Export previews: ${ARGS[*]}"
echo "=============================================="

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="uv run --no-sync python"
fi
${PYTHON} -m evaluation.export_previews "${ARGS[@]}"
