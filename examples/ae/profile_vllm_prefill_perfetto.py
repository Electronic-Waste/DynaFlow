"""Export a Perfetto/Chrome trace of ONE prefill forward — per-kernel timing, NO GPU metrics.

Uses vLLM's torch-profiler integration (VLLM_TORCH_PROFILER_DIR): start_profile() -> one
prefill generate -> stop_profile() writes a `*.pt.trace.json.gz` PER WORKER RANK, loadable
directly at https://ui.perfetto.dev. The trace has every CPU op + CUDA kernel with its
duration on the GPU timeline — exactly "how long each kernel ran" (no hardware-counter
sampling, so no CUPTI observer effect on timing).

Defaults to the Qwen3-30B-A3B MoE with dummy weights (load_format=dummy -> no checkpoint
read; kernel timings are value-independent). enforce_eager=True so the real fused CUDA
kernels (fused_add_rms_norm / silu_and_mul / fused_moe / flash / rotary ...) appear with
their own durations (compiled mode would decompose them into thousands of tiny aten ops).

Env:
  VLLM_TORCH_PROFILER_DIR  output dir (REQUIRED)
  MODEL_PATH               model dir (default Qwen3-30B-A3B)
  TP                       tensor-parallel size (default 2)
  INPUT_LEN                prefill length (default 2048)
  ENFORCE_EAGER            1 (default) -> real fused kernels; 0 -> compiled
  MAX_TOKENS               1 (default) -> prefill only; >1 adds decode steps
"""
import os
import sys

sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL_PATH", "/raid/catalyst/models/Qwen3-30B-A3B")
INPUT_LEN = int(os.environ.get("INPUT_LEN", "2048"))
TP = int(os.environ.get("TP", "2"))
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1"))
assert os.environ.get("VLLM_TORCH_PROFILER_DIR"), "set VLLM_TORCH_PROFILER_DIR (output dir)"


def main():
    llm = LLM(
        model=MODEL,
        load_format="dummy",
        tokenizer=MODEL,
        tensor_parallel_size=TP,
        enforce_eager=ENFORCE_EAGER,
        gpu_memory_utilization=0.85,
        max_num_seqs=1,
        max_num_batched_tokens=INPUT_LEN,   # whole prompt in one prefill forward
        max_model_len=INPUT_LEN + 16,
        enable_prefix_caching=False,        # else the profiled generate hits the warmup's cache -> empty forward
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    ids = list(range(10, 10 + INPUT_LEN))

    # warmup OUTSIDE profiling (allocator / kernels / any compile warm)
    llm.generate({"prompt_token_ids": ids}, sp, use_tqdm=False)
    llm.generate({"prompt_token_ids": ids}, sp, use_tqdm=False)

    # capture exactly one prefill forward
    llm.start_profile()
    llm.generate({"prompt_token_ids": ids}, sp, use_tqdm=False)
    llm.stop_profile()
    print(f"Done. Perfetto trace(s) (per rank) -> {os.environ['VLLM_TORCH_PROFILER_DIR']}")


if __name__ == "__main__":
    main()
