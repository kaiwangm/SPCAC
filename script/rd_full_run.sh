#!/bin/bash
# Full RD collect + plot outside Cursor sandbox.
set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure no sandbox package-manager cache overrides.
unset UV_CACHE_DIR PNPM_STORE_PATH PLAYWRIGHT_BROWSERS_PATH \
      HOMEBREW_CACHE BUNDLE_PATH COMPOSER_HOME GRADLE_USER_HOME CP_HOME_DIR || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p result
LOG=result/full_collect.log
exec > >(tee "$LOG") 2>&1

echo "===== START $(date -Is) ====="
echo "pwd=$(pwd)"
echo "UV_CACHE_DIR=${UV_CACHE_DIR:-<unset>}"
echo "uv=$(command -v uv) ($(uv --version))"
echo "python via: uv run --no-sync"

MODELS=(
  baseline_factorized
  baseline_mean
  grouping
  elpcac_l
  elpcac
)
QUALITIES=(0 1 2 3 4 5)
OUT_DIR=result
EPOCH=las

echo "----- coco3d → 8iVFBv2 -----"
uv run --no-sync python -m evaluation.rd_collect \
  --train_dataset coco3d \
  --epoch "${EPOCH}" \
  --out_dir "${OUT_DIR}" \
  --models "${MODELS[@]}" \
  --datasets \
    j8ivfbv2-longdress10 \
    j8ivfbv2-loot10 \
    j8ivfbv2-redandblack10 \
    j8ivfbv2-soldier10 \
  --qualities "${QUALITIES[@]}" \
  --verbose

echo "----- coco3d → owlii -----"
uv run --no-sync python -m evaluation.rd_collect \
  --train_dataset coco3d \
  --epoch "${EPOCH}" \
  --out_dir "${OUT_DIR}" \
  --models "${MODELS[@]}" \
  --datasets \
    owlii-basketball_player11 \
    owlii-dancer11 \
    owlii-exercise11 \
    owlii-model11 \
  --qualities "${QUALITIES[@]}" \
  --verbose

echo "----- scannet → scannet -----"
uv run --no-sync python -m evaluation.rd_collect \
  --train_dataset scannet \
  --epoch "${EPOCH}" \
  --out_dir "${OUT_DIR}" \
  --models "${MODELS[@]}" \
  --datasets scannet \
  --qualities "${QUALITIES[@]}" \
  --verbose

echo "----- sensat_urban → sensat_urban -----"
uv run --no-sync python -m evaluation.rd_collect \
  --train_dataset sensat_urban \
  --epoch "${EPOCH}" \
  --out_dir "${OUT_DIR}" \
  --models "${MODELS[@]}" \
  --datasets sensat_urban \
  --qualities "${QUALITIES[@]}" \
  --verbose

echo "----- plot all -----"
uv run --no-sync python -m evaluation.rd_plot \
  --in_dir "${OUT_DIR}"

echo "===== DONE $(date -Is) ====="
