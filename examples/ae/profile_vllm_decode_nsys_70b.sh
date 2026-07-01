#!/usr/bin/env bash
# Capture ONE batch-N DECODE step of Llama-3.1-70B with Nsight Systems, to measure the
# share of the "CUDA-bound" auxiliary kernels (RMSNorm/SiLU/KV-store/RoPE) at small batch.
# Decode sibling of profile_vllm_prefill_nsys_70b.sh; trigger token count = DECODE_BATCH.
#
# Usage: CUDA_VISIBLE_DEVICES=1,2 DECODE_BATCH=64 CTX=512 bash profile_vllm_decode_nsys_70b.sh
set -u
AE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$AE_DIR/.venv"
OUTDIR="$AE_DIR/results/nsys"
TP="${TP:-2}"
DECODE_BATCH="${DECODE_BATCH:-64}"
CTX="${CTX:-512}"
WARMUP_BATCH="${WARMUP_BATCH:-32}"
GPU_METRICS="${GPU_METRICS:-0}"
METRIC_SET="${METRIC_SET:-gb10x}"
TAG="llama70b_decode_tp${TP}_b${DECODE_BATCH}_ctx${CTX}"
mkdir -p "$OUTDIR"

export TMPDIR="${TMPDIR:-$OUTDIR/.nsys_tmp}"; mkdir -p "$TMPDIR"
export PATH="$VENV/bin:$PATH"
export HF_HOME=/raid/catalyst/models
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
export VLLM_NSYS_PROFILE_TOKENS="$DECODE_BATCH"   # decode step has DECODE_BATCH positions -> triggers
export DECODE_BATCH CTX WARMUP_BATCH TP
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900

metrics_flags=()
if [ "$GPU_METRICS" = "1" ]; then
  TAG="${TAG}_metrics"
  metrics_flags=(--gpu-metrics-devices="$CUDA_VISIBLE_DEVICES" --gpu-metrics-set="$METRIC_SET")
fi

echo "=== profiling $TAG on GPUs [$CUDA_VISIBLE_DEVICES] (decode batch=$DECODE_BATCH, ctx=$CTX, warmup_batch=$WARMUP_BATCH) ==="
nsys profile \
  --output "$OUTDIR/$TAG" \
  --trace=cuda,nvtx \
  --sample=none \
  "${metrics_flags[@]}" \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --trace-fork-before-exec=true \
  --force-overwrite=true \
  "$VENV/bin/python" "$AE_DIR/profile_vllm_decode_nsys_70b.py"
echo "=== wrote $OUTDIR/$TAG.nsys-rep ==="
