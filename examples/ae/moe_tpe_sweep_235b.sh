#!/usr/bin/env bash
# token-per-expert sweep on the BIGGER MoE: Qwen3-235B-A22B, TP4 + EP4 (GPUs 1-4).
# Same metric as the 30B sweep: Tensor Active DURING fused_moe vs tokens/expert (= L/16).
set -u
AE=/home/shaow/DynaFlow/examples/ae
OUT=$AE/results/nsys/sweep235
mkdir -p "$OUT" "$AE/results/nsys/.nsys_tmp"
export TMPDIR="$AE/results/nsys/.nsys_tmp"
export PATH="$AE/.venv/bin:$PATH" HF_HOME=/raid/catalyst/models
export CUDA_VISIBLE_DEVICES=1,2,3,4 VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_NVTX_SCOPES_FOR_PROFILING=1 TP=4 ENFORCE_EAGER=1 ENABLE_EP=1
export MODEL_PATH=/raid/catalyst/models/Qwen3-235B-A22B
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
me=$(whoami)
cleanup(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do
  [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" = "$me" ] && kill -9 $p 2>/dev/null; done; sleep 5; }

for L in 256 512 1024 2048 4096 8192 16384; do
  W=$((L+64))
  echo "===== L=$L (tok/expert ~= $((L/16))), warmup=$W ====="
  VLLM_NSYS_PROFILE_TOKENS=$L WARMUP_LEN=$W \
  nsys profile --output "$OUT/moe235_ep_L${L}" \
    --trace=cuda,nvtx --sample=none \
    --gpu-metrics-devices=1,2,3,4 --gpu-metrics-set=gb10x \
    --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
    --trace-fork-before-exec=true --force-overwrite=true \
    "$AE/.venv/bin/python" "$AE/profile_vllm_prefill_nsys_70b.py" > "$OUT/run_L${L}.log" 2>&1
  echo "  exit=$? -> $(ls -la $OUT/moe235_ep_L${L}.nsys-rep 2>/dev/null | awk '{print $5}') bytes"
  cleanup
done
echo "===== 235B sweep done ====="