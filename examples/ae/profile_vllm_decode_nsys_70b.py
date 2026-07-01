"""Drive ONE batch-N DECODE forward of Llama-3.1-70B for an Nsight Systems capture.

Decode sibling of profile_vllm_prefill_nsys_70b.py. Goal: at a small decode batch
(default 64) measure what fraction of the decode step is the "CUDA-bound" auxiliary
kernels (RMSNorm / SiLU / KV-cache-store / RoPE) vs the weight-streaming GEMMs / attention.

Mechanism: the capture window in LlamaModel.forward fires cudaProfilerStart/Stop on the
forward whose token count == VLLM_NSYS_PROFILE_TOKENS. A decode step over B sequences has
positions.shape[0] == B (one new token per sequence), so we set the trigger to B. To keep
the trigger unique:
  * warmup uses a DIFFERENT batch (WARMUP_BATCH != B) -> its decode steps have != B tokens,
  * the prefills (B*CTX tokens) and vLLM's init dummy forward (max_num_batched_tokens) are
    all != B,
so the ONLY forward with exactly B positions is the profiled batch's first decode step.

Dummy weights (load_format="dummy") -> no checkpoint read; kernel timings are value-independent.
The (norm+silu+kv+rope)/total ratio is taken within one decode step, so it is clock-invariant.

Env:
  DECODE_BATCH              decode batch = trigger token count (default 64)
  CTX                       context length per sequence / prompt length (default 512)
  WARMUP_BATCH              warmup batch, MUST != DECODE_BATCH (default 32)
  TP                        tensor-parallel size (default 2)
  VLLM_NSYS_PROFILE_TOKENS  must equal DECODE_BATCH (set by the runner)
"""
import os
import sys

sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL_PATH", "/raid/catalyst/models/Llama-3.1-70B-Instruct-4gpus")
BATCH = int(os.environ.get("DECODE_BATCH", "64"))
CTX = int(os.environ.get("CTX", "512"))
WARMUP_BATCH = int(os.environ.get("WARMUP_BATCH", "32"))
TP = int(os.environ.get("TP", "2"))
# max_num_seqs MUST differ from BATCH: vLLM's sampler-profiling dummy run executes at
# exactly max_num_seqs tokens during init and (being a dummy) SKIPS attention/KV-store; if it
# equals BATCH it fires the capture on that fake forward instead of the real decode step.
MAX_SEQS = int(os.environ.get("MAX_SEQS", str(2 * BATCH)))

# the captured forward is the decode step (B positions); make sure nothing else has B tokens
PROF_TOK = int(os.environ.get("VLLM_NSYS_PROFILE_TOKENS", str(BATCH)))
assert PROF_TOK == BATCH, f"VLLM_NSYS_PROFILE_TOKENS ({PROF_TOK}) must equal DECODE_BATCH ({BATCH})"
assert WARMUP_BATCH != BATCH, "WARMUP_BATCH must differ from DECODE_BATCH so warmup decode != trigger"
assert MAX_SEQS != BATCH, "MAX_SEQS must differ from DECODE_BATCH (sampler dummy run = max_num_seqs tokens)"
CAP = BATCH * CTX     # one-shot prefill of the profiled batch; also the mem-profiling dummy size (!= B)


def prompts(n, base):
    # UNIQUE token ids per sequence (no shared prefix) -> a genuine decode, not a prefix-cache degenerate one
    return [{"prompt_token_ids": list(range(base + i * (CTX + 3), base + i * (CTX + 3) + CTX))}
            for i in range(n)]


def main():
    llm = LLM(
        model=MODEL,
        load_format="dummy",
        tokenizer=MODEL,
        tensor_parallel_size=TP,
        enforce_eager=True,                 # run the eager forward -> NVTX/profiler hooks fire
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.85")),
        max_num_seqs=MAX_SEQS,
        enable_prefix_caching=False,        # don't dedup -> every seq does real attention/KV-store
        max_num_batched_tokens=CAP,         # prefill all B seqs in one forward (positions=B*CTX != B)
        max_model_len=CTX + 8,
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=4)   # prefill + a few decode steps

    # warmup at WARMUP_BATCH seqs -> decode positions = WARMUP_BATCH != BATCH (no capture); warms decode kernels
    llm.generate(prompts(WARMUP_BATCH, 1), sp, use_tqdm=False)

    # profiled: BATCH seqs -> after prefill, the first decode step has exactly BATCH positions -> capture
    llm.generate(prompts(BATCH, 50000), sp, use_tqdm=False)
    print(f"Done. Captured a {BATCH}-sequence DECODE step (ctx={CTX}, TP={TP}, 80-layer Llama-3.1-70B).")


if __name__ == "__main__":
    main()
