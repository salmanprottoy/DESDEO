#!/bin/bash
# One-time setup for running the XNIMBUS grounded-chat benchmark on CSC Roihu.
# Run on a Roihu-GPU login node after unpacking the DESDEO code into $CODE_DIR:
#
#   export PROJECT=project_<number>
#   bash $CODE_DIR/scripts/xnimbus_eval/roihu/setup.sh
#
# 1. installs uv (user space) and the benchmark's Python environment into the code directory;
# 2. downloads the open-weight models into the project's scratch Hugging Face cache.
# Model weights are public, ungated downloads. No API keys are needed or used.
set -euo pipefail

: "${PROJECT:?Set PROJECT first, for example: export PROJECT=project_2001234}"
CODE_DIR=${CODE_DIR:-/projappl/$PROJECT/$USER/DESDEO}
SCRATCH_DIR=${SCRATCH_DIR:-/scratch/$PROJECT/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH_DIR/hf-cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH_DIR/uv-cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/projappl/$PROJECT/$USER/uv-python}

# Same list as run_benchmark.sbatch: <served name>=<Hugging Face repository>.
MODELS=${MODELS:-"qwen3.5-9b=Qwen/Qwen3.5-9B gemma4-12b=google/gemma-4-12B-it ministral-3-8b=mistralai/Ministral-3-8B-Instruct-2512 llama3.1-8b=unsloth/Llama-3.1-8B-Instruct"}

if [[ ! -f "$CODE_DIR/pyproject.toml" ]]; then
    echo "DESDEO code not found in $CODE_DIR. Unpack it there or set CODE_DIR." >&2
    exit 1
fi
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

echo "== 1/2 Python environment for the benchmark client"
UV_BIN_DIR=${UV_BIN_DIR:-$HOME/.local/bin}
export PATH="$UV_BIN_DIR:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    # Pin the install directory: on Roihu the installer's default resolves to ~/.local/share/bin.
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_BIN_DIR" UV_NO_MODIFY_PATH=1 sh
fi
cd "$CODE_DIR"
# coco-experiment and tsp-solver have no ARM wheels and are not used by the chat benchmark.
uv sync --frozen --no-dev --no-install-package coco-experiment --no-install-package tsp-solver
.venv/bin/python -c "import desdeo.api.evaluation, desdeo.api.llm_config; print('benchmark client OK')"

echo "== 2/2 Model weights into $HF_HOME"
module purge
module load python-vllm/0.29.0
for entry in $MODELS; do
    repo=${entry#*=}
    echo "-- $repo"
    python - "$repo" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    sys.argv[1],
    # Hugging Face-format weights only; skip duplicate original/consolidated checkpoints.
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt", "tokenizer*"],
    ignore_patterns=["original/*", "consolidated*"],
)
PY
done
du -sh "$HF_HOME"
echo "Setup finished. Submit the benchmark with run_benchmark.sbatch."
