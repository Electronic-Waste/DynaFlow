#!/usr/bin/env bash
# Profile ONE transformer forward step of NanoFlow with Nsight Systems, comparing
# the fused AR+RMSNorm kernel (use_ar_norm_fusion=true) vs no fusion (false).
#
# Uses nsys capture-range tied to cudaProfilerStart/Stop, which the NanoFlow
# scheduler fires around the Nth overlap step (profile_step). nsys records
# NOTHING until that point, so warmup/compile/other steps are excluded -> a tiny
# trace of a single forward step (the per-layer kernel pattern repeats inside).
#
# Usage: bash profile_nanoflow.sh [fusion|nofusion|both]
set -u

AE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV=/home/shaow/DynaFlow/examples/ae/.venv
SCHED="$AE_DIR/scheduler/vllm/nanoflow.py:NanoFlowScheduler"
OUTDIR="$AE_DIR/results/nsys"
PROFILE_STEP=3   # capture the 3rd overlap step (after dynamo compile settles)
mkdir -p "$OUTDIR"

export PATH="$VENV/bin:$PATH"
export HF_HOME=/raid/catalyst/models
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# nsys (CUPTI) slows the uncaptured compile forward; raise the worker RPC timeout.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900

# Small prefill-heavy workload: each prefill step packs to 16384 > 2*4096 tokens
# so the 2-nano-batch overlap path (and the fused kernel) is exercised. A few
# prefill steps is enough to reach profile_step.
COMMON_BENCH=(vllm bench throughput
  --model meta-llama/Meta-Llama-3.1-8B-Instruct
  --tensor-parallel-size 2
  --num-prompts 64 --input-len 1024 --output-len 8 --n 1
  --compilation-config '{"cudagraph_mode": "NONE"}')

run_one() {
  local tag="$1" fusion="$2"
  local cfg="{\"scheduler_path\":\"$SCHED\",\"use_inductor\": false,\"min_nano_split_tokens\": 4096,\"max_num_splits\": 2,\"use_ar_norm_fusion\": $fusion,\"profile_step\": $PROFILE_STEP}"
  echo "=== profiling nanoflow_$tag (use_ar_norm_fusion=$fusion), capturing overlap step $PROFILE_STEP ==="
  nsys profile \
    --output "$OUTDIR/nanoflow_$tag" \
    --trace=cuda,nvtx \
    --sample=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --trace-fork-before-exec=true \
    --force-overwrite=true \
    "${COMMON_BENCH[@]}" --dynaflow-config "$cfg"
  echo "=== wrote $OUTDIR/nanoflow_$tag.nsys-rep ==="
}

case "${1:-both}" in
  fusion)   run_one fusion true ;;
  nofusion) run_one nofusion false ;;
  both)     run_one fusion true; run_one nofusion false ;;
esac
