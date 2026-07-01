"""Export a Perfetto trace of baseline-vLLM transformer-layer prefill (+decode).

Baseline = plain vLLM (no DynaFlow scheduler), Llama-3.1-8B, TP2.

Strategy (mirrors PROFILE.md's "never profile torch.compile" rule):
  1. Build the engine and run a warmup generate -> populates the compile cache
     and captures CUDA graphs, so the profiled step records inference only.
  2. start_profile() -> a short generate -> stop_profile().

Env knobs:
  VLLM_TORCH_PROFILER_DIR   where the *.pt.trace.json.gz is written (required)
  CUDAGRAPH_MODE            NONE | PIECEWISE | FULL | FULL_AND_PIECEWISE  (default NONE)
  MAX_TOKENS                output tokens; 1 = prefill only, >1 adds decode steps (default 1)

A 2048-token prefill has a dynamic shape and runs eager/piecewise (NOT inside a
CUDA graph); only the fixed-size decode steps replay from a graph. So set
MAX_TOKENS>1 to capture decode steps and see graph replay (no launch bubbles)
next to the prefill block.

vLLM's profiler (enabled by VLLM_TORCH_PROFILER_DIR) writes a torch/Chrome trace
(`*.pt.trace.json.gz`) per worker rank, loadable directly at https://ui.perfetto.dev.
"""
import os
import sys

# This file lives in examples/ae/, which also contains a `vllm/` source dir.
# That dir shadows the installed `vllm` package (sys.path[0] == this file's dir),
# so drop it before importing vllm.
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]

from vllm import LLM, SamplingParams

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_LEN = int(os.environ.get("INPUT_LEN", "2048"))  # prefill length (single forward)
CUDAGRAPH_MODE = os.environ.get("CUDAGRAPH_MODE", "NONE")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1"))
# enforce_eager=True runs the Python model forward (no torch.compile), which is
# required for in-model record_function scopes (e.g. per-layer "Layer_N") to fire.
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0") == "1"

def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        compilation_config={"cudagraph_mode": CUDAGRAPH_MODE},
        enforce_eager=ENFORCE_EAGER,
        gpu_memory_utilization=0.85,
        max_num_seqs=1,
        # Process the whole prompt in ONE prefill forward (no chunked prefill),
        # so the AllReduce-vs-compute ratio reflects this exact token count.
        max_num_batched_tokens=INPUT_LEN,
        max_model_len=INPUT_LEN + 16,
        disable_log_stats=True,
    )

    prompt_ids = list(range(10, 10 + INPUT_LEN))
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # --- warmup: trigger torch.compile / autotune / graph capture OUTSIDE profiling ---
    llm.generate({"prompt_token_ids": prompt_ids}, sp, use_tqdm=False)
    llm.generate({"prompt_token_ids": prompt_ids}, sp, use_tqdm=False)

    # --- capture: 1 prefill step (+ MAX_TOKENS-1 decode steps) ---
    llm.start_profile()
    llm.generate({"prompt_token_ids": prompt_ids}, sp, use_tqdm=False)
    llm.stop_profile()

    print(f"Profiling done (cudagraph={CUDAGRAPH_MODE}, max_tokens={MAX_TOKENS}). "
          f"Trace(s) -> {os.environ.get('VLLM_TORCH_PROFILER_DIR')}")


if __name__ == "__main__":
    main()
