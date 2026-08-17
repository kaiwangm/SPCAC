#!/bin/bash
# Plot RD curves from collected JSON (no GPU needed)
# Usage:
#   bash script/rd_plot.sh                            # all datasets under result/
#   bash script/rd_plot.sh j8ivfbv2-longdress10       # one dataset folder
#
# Env:
#   IN_DIR          default result
#   DPI             default 150

set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

IN_DIR="${IN_DIR:-result}"
DPI="${DPI:-150}"
DATASET="${1:-}"

ARGS=(
    --in_dir "${IN_DIR}"
    --dpi "${DPI}"
)

if [ -n "${DATASET}" ]; then
    ARGS+=(--dataset "${DATASET}")
fi

echo "=============================================="
echo "RD plot: ${ARGS[*]}"
echo "=============================================="

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="uv run --no-sync python"
fi
${PYTHON} -m evaluation.rd_plot "${ARGS[@]}"
