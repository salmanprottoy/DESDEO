#!/bin/bash
# XNIMBUS grounded-chat benchmark on CSC Roihu: one GH200 GPU, vLLM server, models run one after another.
#
# Full run (all models, 3 repetitions):
#   sbatch --account=$PROJECT scripts/xnimbus_eval/roihu/run_benchmark.sbatch
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
#SBATCH --time=04:00:00
#SBATCH --output=xnimbus-bench-%j.out
set -euo pipefail

PROJECT=${PROJECT:-$SLURM_JOB_ACCOUNT}
CODE_DIR=${CODE_DIR:-/projappl/$PROJECT/$USER/DESDEO}
SCRATCH_DIR=${SCRATCH_DIR:-/scratch/$PROJECT/$USER}
OUT_DIR=${OUT_DIR:-$SCRATCH_DIR/xnimbus-results/$SLURM_JOB_ID}
REPETITIONS=${REPETITIONS:-3}
MODELS=${MODELS:-"qwen3.5-9b=Qwen/Qwen3.5-9B gemma4-12b=google/gemma-4-12B-it ministral-3-8b=mistralai/Ministral-3-8B-Instruct-2512 llama3.1-8b=unsloth/Llama-3.1-8B-Instruct"}
PY=$CODE_DIR/.venv/bin/python

export HF_HOME=${HF_HOME:-$SCRATCH_DIR/hf-cache}
export HF_HUB_OFFLINE=1  # weights were downloaded by setup.sh
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torch-cache

# Benchmark client: local profiles only, pointed at this job's vLLM server.
unset OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY GEMINI_API_KEY DEEPSEEK_API_KEY GROQ_API_KEY
PORT=$((20000 + SLURM_JOB_ID % 20000))
VLLM_API_KEY=$(openssl rand -hex 24)
export VLLM_API_KEY
export XNIMBUS_LOCAL_API_KEY=$VLLM_API_KEY
export XNIMBUS_LOCAL_BASE_URL=http://127.0.0.1:$PORT/v1
export XNIMBUS_LOCAL_THINKING_CONTROL=chat_template
export XNIMBUS_LOCAL_TIMEOUT_SECONDS=120

mkdir -p "$OUT_DIR" "$TORCHINDUCTOR_CACHE_DIR"
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

for entry in $MODELS; do
    name=${entry%%=*}
    repo=${entry#*=}
    echo "== $name ($repo) $(date -Is)"

    apptainer exec --bind="$(csc-common-bind)" "$SIF" vllm serve "$repo" \
        --served-model-name "$name" \
        --host 127.0.0.1 --port "$PORT" \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.85 \
        --generation-config vllm \
        >"$OUT_DIR/vllm-$name.log" 2>&1 &
    server_pid=$!

    ready=0
    for _ in $(seq 1 180); do
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
        stop_server
        continue
    fi
    echo "server ready $(date -Is)"

    XNIMBUS_LOCAL_MODELS=$name "$PY" "$CODE_DIR/scripts/xnimbus_eval/run_benchmark.py" \
        --providers local \
        --repetitions "$REPETITIONS" \
        --out-dir "$OUT_DIR/$name" || echo "benchmark failed for $name" >&2

    stop_server
done

echo "Results in $OUT_DIR"
ls -R "$OUT_DIR"
