#!/bin/bash
# Launch the local Gradio evaluation viewer.
# Usage:
#   bash script/launch_demo.sh
#   PORT=7861 bash script/launch_demo.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export PORT="${PORT:-7860}"
export SPCAC_RESULT_DIR="${SPCAC_RESULT_DIR:-result}"
export SPCAC_PREVIEW_DIR="${SPCAC_PREVIEW_DIR:-result}"

echo "SPCAC viewer  http://127.0.0.1:${PORT}"
echo "metrics: ${SPCAC_RESULT_DIR}   previews: ${SPCAC_PREVIEW_DIR}"

if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import gradio, plotly" 2>/dev/null; then
    .venv/bin/python preview/app.py
else
    uv run --with gradio --with plotly python preview/app.py
fi
