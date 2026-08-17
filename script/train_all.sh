#!/bin/bash
# Training script for SPCAC models (uv version)
# Usage: bash script/train_all.sh <model> <start_gpu> <num_gpu>
#   model     : model config name, e.g. elpcac (see configs/model/)
#   start_gpu : first GPU device index
#   num_gpu   : number of GPUs to use; quality i runs on GPU (i % num_gpu) + start_gpu
#
# Trains quality levels q0-q5 for the given model, one tmux session per quality.
#
# Example:
#   bash script/train_all.sh elpcac 0 4    # train elpcac q0-q5 on GPUs 0-3
#
# Checkpoints are saved to:
#   checkpoints/{dataset}/{model}/quality_{quality}/eb_las.pth

set -e

cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

method=$1
start=$2
num_gpu=$3

if [ -z "$method" ] || [ -z "$start" ] || [ -z "$num_gpu" ]; then
    echo "Usage: bash script/train_all.sh <model> <start_gpu> <num_gpu>"
    exit 1
fi

for i in {0..5}
do
    devidx=$(((i%num_gpu)+start))
    muxname=${method}_tq${i}
    echo "Starting $muxname on GPU $devidx"
    tmux kill-session -t $muxname 2>/dev/null
    tmux new -s $muxname -d
    tmux send-keys -t $muxname "cd $(pwd)" C-m
    tmux send-keys -t $muxname 'CUDA_VISIBLE_DEVICES='$devidx' uv run python -m train.train --model='$method' --quality='$i C-m
    sleep 0.03
done
echo 'Finished'
