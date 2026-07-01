"""Drive a DP+EP prefill of a Qwen3-MoE model for an Nsight Systems capture.

Why DP and not just TP: vLLM only engages the all-to-all EP path
(`FusedMoEParallelConfig.use_all2all_kernels = dp_size > 1 and use_ep`) when
data-parallel size > 1. With pure TP+EP (dp=1) the MoE combine is an all-reduce
regardless of VLLM_ALL2ALL_BACKEND. So to compare all-reduce-EP vs all-to-all-EP
(DeepEP/pplx dispatch+combine) we must run with dp_size>1.

Mechanism mirrors profile_vllm_prefill_nsys_70b.py: each DP rank does warmup at
WARMUP_LEN (!= trigger) then ONE profiled prefill of exactly VLLM_NSYS_PROFILE_TOKENS
tokens. The cudaProfilerStart/Stop window in Qwen3MoeModel.forward (gated on
positions.shape[0] == trigger) fires only on that forward, on every rank
simultaneously, so the all-to-all collective has all ranks participating.

Launch one OS process per DP rank (the supported offline-DP pattern), coordinated
via VLLM_DP_* env + a master port. nsys traces the whole process tree; analyze dev0.

Env:
  DP_SIZE         data-parallel size (default 4)
  TP_SIZE         tensor-parallel size PER dp rank (default 1) -> world = DP*TP
  MODEL_PATH      model dir
  VLLM_NSYS_PROFILE_TOKENS  prefill length that triggers capture (default 2048)
  WARMUP_LEN      warmup prefill length, MUST be > trigger (default 2112)
  ENABLE_EP       1 -> enable_expert_parallel (default 1)
  GPU_MEM_UTIL    gpu_memory_utilization (default 0.45)
  ENFORCE_EAGER   1 -> eager so the in-forward NVTX/profiler hooks fire (default 1)
"""
import os
import sys

sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]

MODEL = os.environ.get("MODEL_PATH", "/raid/catalyst/models/Qwen3-30B-A3B")
TRIG = int(os.environ.get("VLLM_NSYS_PROFILE_TOKENS", "2048"))
WARMUP_LEN = int(os.environ.get("WARMUP_LEN", "2112"))
DP_SIZE = int(os.environ.get("DP_SIZE", "4"))
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
assert WARMUP_LEN > TRIG, "WARMUP_LEN must be > VLLM_NSYS_PROFILE_TOKENS"
CAP = max(TRIG, WARMUP_LEN)


def run_rank(global_dp_rank, dp_master_ip, dp_master_port):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(global_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(DP_SIZE)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    from vllm import LLM, SamplingParams
    # Real weights matter here: DeepEP all-to-all volume/balance depends on the
    # gating's token->expert routing. Dummy (random) weights route pathologically
    # (some experts get 0 tokens, some all), inflating the combine barrier wait.
    load_format = "dummy" if os.environ.get("LOAD_DUMMY", "1") == "1" else "auto"
    llm = LLM(
        model=MODEL,
        load_format=load_format,
        tokenizer=MODEL,
        tensor_parallel_size=TP_SIZE,
        enable_expert_parallel=os.environ.get("ENABLE_EP", "1") == "1",
        enforce_eager=os.environ.get("ENFORCE_EAGER", "1") == "1",
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.45")),
        max_num_seqs=1,
        max_num_batched_tokens=CAP,
        max_model_len=CAP + 16,
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    # distinct token ids per rank -> realistic (non-identical) expert routing spread
    base = 7 + global_dp_rank * 1000
    warm = {"prompt_token_ids": list(range(5, 5 + WARMUP_LEN))}
    llm.generate(warm, sp, use_tqdm=False)
    llm.generate(warm, sp, use_tqdm=False)
    prof = {"prompt_token_ids": list(range(base, base + TRIG))}
    llm.generate(prof, sp, use_tqdm=False)
    print(f"DP rank {global_dp_rank}: captured {TRIG}-token prefill (TP={TP_SIZE}).")


def main():
    from vllm.utils import get_open_port
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    from multiprocessing import Process
    procs = []
    for r in range(DP_SIZE):
        p = Process(target=run_rank, args=(r, dp_master_ip, dp_master_port))
        p.start()
        procs.append(p)
    exit_code = 0
    for p in procs:
        p.join(timeout=1800)
        if p.exitcode is None:
            p.kill(); exit_code = 1
        elif p.exitcode:
            exit_code = p.exitcode
    print(f"DP+EP run done (DP={DP_SIZE} TP={TP_SIZE} world={DP_SIZE*TP_SIZE}).")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
