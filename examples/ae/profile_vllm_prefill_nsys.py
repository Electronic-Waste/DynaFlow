"""Drive ONE baseline-vLLM prefill forward for an Nsight Systems capture.

Pairs with the cudaProfilerStart/Stop window in LlamaModel.forward (gated by
VLLM_NSYS_PROFILE_TOKENS) and `nsys --capture-range=cudaProfilerApi`: nsys records
nothing until that exact forward, so the trace is just one prefill's 32-layer stack
(per-layer NVTX ranges via VLLM_NVTX_SCOPES_FOR_PROFILING=1). enforce_eager=True so
the Python forward (hence the NVTX/profiler hooks) actually runs.

Env:
  VLLM_NSYS_PROFILE_TOKENS  unique prefill length that triggers the capture (e.g. 2000)
  WARMUP_LEN                warmup prefill length, must differ from the trigger (default 2048)
"""
import os
import sys

sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]

from vllm import LLM, SamplingParams

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TRIG = int(os.environ["VLLM_NSYS_PROFILE_TOKENS"])
WARMUP_LEN = int(os.environ.get("WARMUP_LEN", "2048"))
CAP = max(TRIG, WARMUP_LEN)

def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        enforce_eager=True,                 # run the eager forward -> hooks fire
        gpu_memory_utilization=0.85,
        max_num_seqs=1,
        max_num_batched_tokens=CAP,         # whole prompt in one prefill forward
        max_model_len=CAP + 16,
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    # warmup at WARMUP_LEN (does NOT match the trigger -> not captured)
    warm = {"prompt_token_ids": list(range(5, 5 + WARMUP_LEN))}
    llm.generate(warm, sp, use_tqdm=False)
    llm.generate(warm, sp, use_tqdm=False)

    # the profiled forward: exactly TRIG tokens -> fires cudaProfilerStart/Stop
    prof = {"prompt_token_ids": list(range(7, 7 + TRIG))}
    llm.generate(prof, sp, use_tqdm=False)
    print(f"Done. Captured the {TRIG}-token prefill forward.")


if __name__ == "__main__":
    main()
