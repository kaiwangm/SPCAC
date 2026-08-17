#!/bin/bash
# Evaluation script for SPCAC models
# Usage: bash script/evaluation_all.sh [model] [dataset] [quality]
#   If no arguments: run all models on all default datasets
#   If model specified: run only that model
#   If model+dataset: run that combination
#   If model+dataset+quality: run only that specific quality
#
# Env:
#   TRAIN_DATASET   default coco3d (use scannet / sensat_urban for those sets)
#   EPOCH           default las
#   CUDA_VISIBLE_DEVICES  default 0
#   OMP_NUM_THREADS       default 8
#
# Hugging Face Hub (optional):
#   export SPCAC_HF_CHECKPOINT_REPO="kaiwangm/SPCAC-checkpoints"
#   export SPCAC_HF_DATASET_REPO="kaiwangm/SPCAC-datasets"
#   export HF_TOKEN="hf_..."   # only for gated repos / uploads

set -euo pipefail
cd "$(dirname "$0")/.."

# Avoid Cursor sandbox package-manager cache overrides hanging `uv run`.
unset UV_CACHE_DIR PNPM_STORE_PATH PLAYWRIGHT_BROWSERS_PATH \
      HOMEBREW_CACHE BUNDLE_PATH COMPOSER_HOME GRADLE_USER_HOME CP_HOME_DIR || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ -x ".venv/bin/python" ]; then
    PYTHON=(.venv/bin/python)
else
    PYTHON=(uv run --no-sync python)
fi

ALL_MODELS=(
    baseline_factorized
    baseline_mean
    grouping
    elpcac_l
    elpcac
)

# Default eval sets for coco3d-trained checkpoints (sequence PLYs, not shared HDF5).
ALL_DATASETS=(
    j8ivfbv2-longdress10
    j8ivfbv2-loot10
    j8ivfbv2-redandblack10
    j8ivfbv2-soldier10
    owlii-basketball_player11
    owlii-dancer11
    owlii-exercise11
    owlii-model11
)

QUALITIES=(0 1 2 3 4 5)

TRAIN_DATASET="${TRAIN_DATASET:-coco3d}"
EPOCH="${EPOCH:-las}"

model_class_of() {
    case "$1" in
        baseline_factorized) echo factorized_prior ;;
        baseline_mean) echo mean_scale_hyperprior ;;
        grouping) echo grouping ;;
        elpcac_l) echo elpcac_l ;;
        elpcac) echo elpcac ;;
        *) return 1 ;;
    esac
}

run_single() {
    local model=$1
    local dataset=$2
    local quality=$3
    local model_class
    local ckpt
    local out_dir

    model_class="$(model_class_of "${model}")" || {
        echo "Unknown model: ${model}"
        return 1
    }

    ckpt="checkpoints/${TRAIN_DATASET}/${model_class}/quality_${quality}/eb_${EPOCH}.pth"
    if [ ! -f "${ckpt}" ] && [ -z "${SPCAC_HF_CHECKPOINT_REPO:-}" ]; then
        echo "WARNING: Checkpoint not found: ${ckpt} (and SPCAC_HF_CHECKPOINT_REPO unset), skip"
        return 0
    fi

    # Logs under the test-dataset folder (same layout as RD metrics).
    out_dir="result/${dataset}"
    mkdir -p "${out_dir}"

    echo "=============================================="
    echo "Evaluating: model=${model}, dataset=${dataset}, quality=${quality}"
    echo "train_dataset=${TRAIN_DATASET}, epoch=${EPOCH}"
    echo "=============================================="

    "${PYTHON[@]}" -m evaluation.evaluation \
        --model "${model}" \
        --quality "${quality}" \
        --epoch "${EPOCH}" \
        --train_dataset "${TRAIN_DATASET}" \
        --dataset "${dataset}" \
        --verbose 2>&1 | tee "${out_dir}/${model}_q${quality}.log"

    echo "Done: ${model} ${dataset} q${quality}"
    echo ""
}

run_model_dataset() {
    local model=$1
    local dataset=$2
    local q

    for q in "${QUALITIES[@]}"; do
        run_single "${model}" "${dataset}" "${q}"
    done
}

# Main
if [ $# -ge 3 ]; then
    run_single "$1" "$2" "$3"
elif [ $# -ge 2 ]; then
    run_model_dataset "$1" "$2"
elif [ $# -ge 1 ]; then
    for dataset in "${ALL_DATASETS[@]}"; do
        run_model_dataset "$1" "${dataset}"
    done
else
    for model in "${ALL_MODELS[@]}"; do
        for dataset in "${ALL_DATASETS[@]}"; do
            run_model_dataset "${model}" "${dataset}"
        done
    done
fi

echo "=============================================="
echo "All evaluations completed!"
echo "Results saved under: result/{dataset}/"
echo "=============================================="
