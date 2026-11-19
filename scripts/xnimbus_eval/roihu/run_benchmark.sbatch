#!/bin/bash
# XNIMBUS grounded-chat benchmark on CSC Roihu: one GH200 GPU, vLLM server, models run one after another.
#
# Laptop-size models (downloaded to scratch by setup.sh), 3 repetitions:
#   sbatch --account=$PROJECT scripts/xnimbus_eval/roihu/run_benchmark.sbatch
# Larger models (each downloaded to the node's local disk during the job and deleted afterwards):
#   sbatch --account=$PROJECT --export=ALL,MODEL_SET=large scripts/xnimbus_eval/roihu/run_benchmark.sbatch
# 15-minute smoke test (one model, one repetition):
#   sbatch --account=$PROJECT --partition=gputest --time=00:15:00 \
#     --export=ALL,REPETITIONS=1,MODELS=qwen3.5-9b=Qwen/Qwen3.5-9B scripts/xnimbus_eval/roihu/run_benchmark.sbatch
#
# Run setup.sh first. The vLLM server listens on 127.0.0.1 only and requires a per-job random API key.
#SBATCH --job-name=xnimbus-bench
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --output=xnimbus-bench-%j.out
set -euo pipefail

PROJECT=${PROJECT:-$SLURM_JOB_ACCOUNT}
CODE_DIR=${CODE_DIR:-/projappl/$PROJECT/$USER/DESDEO}
SCRATCH_DIR=${SCRATCH_DIR:-/scratch/$PROJECT/$USER}
OUT_DIR=${OUT_DIR:-$SCRATCH_DIR/xnimbus-results/$SLURM_JOB_ID}
REPETITIONS=${REPETITIONS:-3}
SCRATCH_HF_HOME=${SCRATCH_HF_HOME:-$SCRATCH_DIR/hf-cache}
LOCAL_HF_HOME=$TMPDIR/hf-cache
PY=$CODE_DIR/.venv/bin/python

# <served name>=<Hugging Face repository>. Served names keep the family prefixes the client keys on.
SMALL_MODELS="qwen3.5-9b=Qwen/Qwen3.5-9B gemma4-12b=google/gemma-4-12B-it ministral-3-8b=mistralai/Ministral-3-8B-Instruct-2512 llama3.1-8b=unsloth/Llama-3.1-8B-Instruct"
LARGE_MODELS="qwen3.5-27b=Qwen/Qwen3.5-27B gemma4-31b=google/gemma-4-31B-it mistral-small-3.2-24b=mistralai/Mistral-Small-3.2-24B-Instruct-2506 gpt-oss-120b=openai/gpt-oss-120b"
case "${MODEL_SET:-small}" in
    small) DEFAULT_MODELS=$SMALL_MODELS ;;
    large) DEFAULT_MODELS=$LARGE_MODELS ;;
    all) DEFAULT_MODELS="$SMALL_MODELS $LARGE_MODELS" ;;
    *) echo "MODEL_SET must be small, large or all" >&2; exit 1 ;;
esac
MODELS=${MODELS:-$DEFAULT_MODELS}

export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torch-cache

# Benchmark client: local profiles only, pointed at this job's vLLM server.
unset OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY GEMINI_API_KEY DEEPSEEK_API_KEY GROQ_API_KEY
PORT=$((20000 + SLURM_JOB_ID % 20000))
VLLM_API_KEY=$(openssl rand -hex 24)
export VLLM_API_KEY
export XNIMBUS_LOCAL_API_KEY=$VLLM_API_KEY
export XNIMBUS_LOCAL_BASE_URL=http://127.0.0.1:$PORT/v1
export XNIMBUS_LOCAL_THINKING_CONTROL=chat_template
export XNIMBUS_LOCAL_TIMEOUT_SECONDS=180

mkdir -p "$OUT_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$LOCAL_HF_HOME"
module purge
module load python-vllm/0.29.0
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT_DIR/gpu.csv"

server_pid=""
stop_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}
trap stop_server EXIT

# Extra vLLM arguments per model family.
model_args() {
    case "$1" in
        mistral-small-*) echo "--tokenizer-mode mistral" ;;  # repository ships tekken.json, no tokenizer.json
        gpt-oss-*) echo "--reasoning-parser openai_gptoss" ;;
        *) echo "" ;;
    esac
}

for entry in $MODELS; do
    name=${entry%%=*}
    repo=${entry#*=}
    echo "== $name ($repo) $(date -Is)"

    # Use the scratch copy from setup.sh when present; otherwise download to the node's local disk.
    if [[ -d "$SCRATCH_HF_HOME/hub/models--${repo//\//--}" ]]; then
        export HF_HOME=$SCRATCH_HF_HOME HF_HUB_OFFLINE=1
    else
        export HF_HOME=$LOCAL_HF_HOME HF_HUB_OFFLINE=0
        echo "downloading to local disk $(date -Is)"
        python - "$repo" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    sys.argv[1],
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt", "tokenizer*"],
    ignore_patterns=["original/*", "consolidated*", "metal/*"],
)
PY
        export HF_HUB_OFFLINE=1
    fi

    # shellcheck disable=SC2046  # model_args intentionally splits into separate arguments
    apptainer exec --bind="$(csc-common-bind)" "$SIF" vllm serve "$repo" \
        --served-model-name "$name" \
        --host 127.0.0.1 --port "$PORT" \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.85 \
        --generation-config vllm \
        $(model_args "$name") \
        >"$OUT_DIR/vllm-$name.log" 2>&1 &
    server_pid=$!

    ready=0
    for _ in $(seq 1 240); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            break
        fi
        if curl -sf -H "Authorization: Bearer $VLLM_API_KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
            ready=1
            break
        fi
        sleep 5
    done
    if [[ $ready -ne 1 ]]; then
        echo "vLLM did not start for $name; see $OUT_DIR/vllm-$name.log" >&2
        tail -40 "$OUT_DIR/vllm-$name.log" >&2 || true
    else
        echo "server ready $(date -Is)"
        XNIMBUS_LOCAL_MODELS=$name "$PY" "$CODE_DIR/scripts/xnimbus_eval/run_benchmark.py" \
            --providers local \
            --repetitions "$REPETITIONS" \
            --out-dir "$OUT_DIR/$name" || echo "benchmark failed for $name" >&2
    fi

    stop_server
    # Free the node's local disk (150 GiB) before the next large model.
    if [[ "$HF_HOME" == "$LOCAL_HF_HOME" ]]; then
        rm -rf "${LOCAL_HF_HOME:?}/hub"
    fi
done

echo "Results in $OUT_DIR"
ls -R "$OUT_DIR"
