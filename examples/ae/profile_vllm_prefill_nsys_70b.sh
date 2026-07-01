#!/usr/bin/env bash
# Capture ONE baseline-vLLM prefill forward of Llama-3.1-70B with Nsight Systems.
#
# Baseline = plain vLLM (no DynaFlow scheduler). The capture window lives in
# LlamaModel.forward and fires cudaProfilerStart/Stop only on the forward whose token
# count == VLLM_NSYS_PROFILE_TOKENS, so nsys (capture-range=cudaProfilerApi) records
# nothing until that one 80-layer prefill -> a tiny trace, no engine-init/warmup noise.
#
# 70B has no vLLM-loadable checkpoint locally (the only complete weights are Megatron
# MP4-presharded), so the driver uses load_format="dummy": random weights of the correct
# shapes, no checkpoint read, no download. Kernel timings are unaffected by weight values.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1,2 bash profile_vllm_prefill_nsys_70b.sh
# Env (with defaults): TP=2  VLLM_NSYS_PROFILE_TOKENS=2048  WARMUP_LEN=2176
set -u

AE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$AE_DIR/.venv"
OUTDIR="$AE_DIR/results/nsys"
TP="${TP:-2}"
TRIG="${VLLM_NSYS_PROFILE_TOKENS:-2048}"
WARMUP_LEN="${WARMUP_LEN:-2176}"
GPU_METRICS="${GPU_METRICS:-0}"        # 1 -> add per-GPU hardware-counter sampling (SM/warp occ, tensor active, DRAM/NVLink BW)
METRIC_SET="${METRIC_SET:-gb10x}"      # B200 = GB100 die -> gb10x (see OVERLAP.md); gb10x-ct for compute-triage counters
ABLATE="${VLLM_ABLATE_KERNELS:-}"     # comma list: rmsnorm,rope,kvstore,silu (env-gated skips in vllm src)
TAG="llama70b_prefill_tp${TP}_${TRIG}"
[ -n "$ABLATE" ] && TAG="${TAG}_ablate_${ABLATE//,/-}"
mkdir -p "$OUTDIR"

# nsys/ncu write temp files under $TMPDIR; the default /tmp/nvidia is owned by another
# user on this shared box and makes nsys fail with "Unknown error on device 0". Point it
# at a dir we own.
export TMPDIR="${TMPDIR:-$AE_DIR/results/nsys/.nsys_tmp}"
mkdir -p "$TMPDIR"

export PATH="$VENV/bin:$PATH"
export HF_HOME=/raid/catalyst/models
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0       # plain NCCL all-reduce (clean baseline, like the 8B trace)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_NVTX_SCOPES_FOR_PROFILING=1    # per-layer NVTX ranges Layer_0..Layer_79
export VLLM_NSYS_PROFILE_TOKENS="$TRIG"
export WARMUP_LEN="$WARMUP_LEN"
export TP="$TP"
export VLLM_ABLATE_KERNELS="$ABLATE"   # skip RMSNorm/RoPE/KV-store/SiLU kernels (timing ablation)
[ -n "$ABLATE" ] && echo "=== KERNEL ABLATION: $ABLATE ==="
# CUPTI slows the uncaptured warmup/init; raise the worker RPC timeout so it isn't
# mistaken for a hang.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900

# Optional per-GPU hardware-counter sampling. GPU metrics are system-wide on the
# PHYSICAL device, so sample exactly the (free) GPUs we pinned -> no other user's work
# pollutes the counters. Sampling is gated by the cudaProfilerApi capture range, so the
# metrics timeline covers just the one prefill (averages still include the launch gaps
# inside that window -- read SM/tensor activity over the kernel-busy region, see OVERLAP.md).
metrics_flags=()
if [ "$GPU_METRICS" = "1" ]; then
  TAG="${TAG}_metrics"
  metrics_flags=(--gpu-metrics-devices="$CUDA_VISIBLE_DEVICES" --gpu-metrics-set="$METRIC_SET")
  echo "=== GPU metrics ON: set=$METRIC_SET devices=$CUDA_VISIBLE_DEVICES ==="
fi

echo "=== profiling $TAG on GPUs [$CUDA_VISIBLE_DEVICES], trigger=${TRIG} warmup=${WARMUP_LEN} ==="
nsys profile \
  --output "$OUTDIR/$TAG" \
  --trace=cuda,nvtx \
  --sample=none \
  "${metrics_flags[@]}" \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --trace-fork-before-exec=true \
  --force-overwrite=true \
  "$VENV/bin/python" "$AE_DIR/profile_vllm_prefill_nsys_70b.py"
echo "=== wrote $OUTDIR/$TAG.nsys-rep ==="
