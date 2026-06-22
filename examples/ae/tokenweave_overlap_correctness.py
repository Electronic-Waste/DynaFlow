"""Correctness check for the TokenWeave overlap-deadlock fix.

Runs greedy generation on a batch large enough to trigger DynaFlow's
2-nano-batch split (so the fused collective overlap path is exercised), under
two configs, and compares the generated token ids:
  - reference: plain vLLM TP2 (no dynaflow)
  - test:      tokenweave overlap (the patched kernel)
If the outputs match, the signal-pad-offset fix preserves correctness.

Run each config in a SEPARATE process (one model load each) to avoid engine
reuse issues; pass the config via argv.
"""
import sys
import json
import os

# This script lives in examples/ae, which contains a `vllm/` source-checkout
# subdir that would shadow the installed `vllm` package (Python puts the
# script's own dir on sys.path[0]). Drop it before importing vllm.
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]

CONFIG = sys.argv[1]  # "baseline" or "tokenweave"
OUT = sys.argv[2]

from vllm import LLM, SamplingParams

AE_DIR = os.path.dirname(os.path.abspath(__file__))
TW_CONFIG = {
    "scheduler_path": f"{AE_DIR}/scheduler/vllm/tokenweave.py:TokenWeaveScheduler",
    "use_inductor": False,
    "min_nano_split_tokens": 4096,
    "max_num_splits": 2,
    "hidden_dim": 4096,  # Llama-3.1-8B hidden size (for eager symm-mem rendezvous)
}

# Distinct prompts (unique leading token so prefix caching can't collapse them),
# each long enough that the combined prefill in one scheduler step exceeds
# 2*4096 tokens -> triggers DynaFlow's 2-nano-batch overlap path.
_TOPICS = [
    "tensor parallelism", "pipeline scheduling", "kv cache paging",
    "speculative decoding", "rotary embeddings", "flash attention",
    "quantization tradeoffs", "all-reduce collectives",
]
# 128 distinct long prompts -> prefill spans multiple 16384-token scheduler
# steps. The first big step initializes rendezvous in single-nano-batch mode
# (per the fix); LATER big steps then exercise the 2-nano-batch overlap path.
PROMPTS = [
    (f"Prompt {i}. " + " ".join(
        f"Sentence {i}-{k} discussing {_TOPICS[(i + k) % len(_TOPICS)]} and its "
        f"tradeoffs in distributed LLM inference."
        for k in range(20)))
    for i in range(128)
]

kwargs = dict(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,
    enforce_eager=False,
    compilation_config={"cudagraph_mode": "NONE"},
    gpu_memory_utilization=0.85,
    enable_prefix_caching=False,
    max_num_batched_tokens=16384,
)
if CONFIG == "tokenweave":
    kwargs["dynaflow_config"] = TW_CONFIG

llm = LLM(**kwargs)
sp = SamplingParams(temperature=0.0, max_tokens=64)
outs = llm.generate(PROMPTS, sp)
result = [list(o.outputs[0].token_ids) for o in outs]
with open(OUT, "w") as f:
    json.dump(result, f)
print(f"[{CONFIG}] wrote {len(result)} sequences to {OUT}")
