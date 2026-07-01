"""Drive ONE baseline-vLLM prefill forward of Llama-3.1-70B for an Nsight Systems capture.

70B sibling of profile_vllm_prefill_nsys.py. Same mechanism: the cudaProfilerStart/Stop
window in LlamaModel.forward (gated by VLLM_NSYS_PROFILE_TOKENS) plus
`nsys --capture-range=cudaProfilerApi` mean nsys records nothing until the one prefill
forward whose token count == VLLM_NSYS_PROFILE_TOKENS. The trace is then just that
prefill's 80-layer stack (per-layer NVTX ranges Layer_0..Layer_79 via
VLLM_NVTX_SCOPES_FOR_PROFILING=1). enforce_eager=True so the Python forward (hence the
NVTX/profiler hooks) actually runs.

Two 70B-specific changes vs the 8B script:
  * MODEL points at the local Llama-3.1-70B config/tokenizer dir, and load_format="dummy"
    so vLLM builds the 80-layer architecture with RANDOM weights and never reads a
    checkpoint. The only complete local 70B weights are Megatron MP4-presharded
    (model{0..3}-mp4.safetensors, no index.json) which vLLM's HF loader cannot read.
    Weight *values* don't affect kernel timings, so dummy weights are the correct,
    download-free choice for a kernel-timing trace.
  * 80 layers (vs 32) -> the captured window simply contains more repeats of the same
    per-layer kernel pattern.

Env:
  VLLM_NSYS_PROFILE_TOKENS  prefill length that triggers the capture (default 2048)
  WARMUP_LEN                warmup/dummy-run prefill length; MUST be > the trigger so
                            neither the warmup generates nor vLLM's init memory-profiling
                            dummy forward (sized at max_num_batched_tokens) accidentally
                            fires the capture (default 2176)
  TP                        tensor-parallel size (default 2, to match the 8B baseline)
  MODEL_PATH               override the model dir (default the local 70B dir)
"""
import os
import sys

# examples/ae/ contains a `vllm/` source dir that would shadow the installed
# (editable) vllm package; drop this file's dir from sys.path before importing.
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL_PATH", "/raid/catalyst/models/Llama-3.1-70B-Instruct-4gpus")
TRIG = int(os.environ.get("VLLM_NSYS_PROFILE_TOKENS", "2048"))
WARMUP_LEN = int(os.environ.get("WARMUP_LEN", "2176"))
TP = int(os.environ.get("TP", "2"))

# The capture fires on ANY forward with positions.shape[0] == TRIG. vLLM's init memory
# profiling does a dummy forward of max_num_batched_tokens (= CAP) tokens, and warmup
# uses WARMUP_LEN tokens. Keeping WARMUP_LEN > TRIG makes CAP = WARMUP_LEN != TRIG, so the
# ONLY forward that matches TRIG is the explicit profiled generate below.
assert WARMUP_LEN > TRIG, (
    f"WARMUP_LEN ({WARMUP_LEN}) must be > VLLM_NSYS_PROFILE_TOKENS ({TRIG}); otherwise "
    f"vLLM's init dummy forward (sized at max_num_batched_tokens) would fire the capture.")
CAP = max(TRIG, WARMUP_LEN)


def main():
    llm = LLM(
        model=MODEL,
        load_format="dummy",                # random weights; never read the MP4 shards / network
        tokenizer=MODEL,
        tensor_parallel_size=TP,
        enable_expert_parallel=os.environ.get("ENABLE_EP", "0") == "1",  # EP: partition experts across ranks (all-to-all) instead of TP-sharding them (all-reduce)
        # eager (default) -> in-forward NVTX/cudaProfiler hooks fire. Set ENFORCE_EAGER=0 to run
        # COMPILED (torch.compile/piecewise cudagraph) for a clean trace; then capture via the
        # model-runner-level window (VLLM_NSYS_CAPTURE_RUNNER_TOKENS), not the in-forward hook.
        enforce_eager=os.environ.get("ENFORCE_EAGER", "1") == "1",
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.85")),
        max_num_seqs=1,
        max_num_batched_tokens=CAP,         # whole prompt in one prefill forward (no chunking)
        max_model_len=CAP + 16,
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    # warmup at WARMUP_LEN (!= trigger -> not captured); 2x to settle allocator/kernels
    warm = {"prompt_token_ids": list(range(5, 5 + WARMUP_LEN))}
    llm.generate(warm, sp, use_tqdm=False)
    llm.generate(warm, sp, use_tqdm=False)

    # the profiled forward: exactly TRIG tokens -> fires cudaProfilerStart/Stop
    prof = {"prompt_token_ids": list(range(7, 7 + TRIG))}
    llm.generate(prof, sp, use_tqdm=False)
    print(f"Done. Captured the {TRIG}-token prefill forward (TP={TP}, 80-layer Llama-3.1-70B).")


if __name__ == "__main__":
    main()
