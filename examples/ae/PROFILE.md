# Profiling NanoFlow with Nsight Systems

How to capture an Nsight Systems (nsys) trace of **one** NanoFlow forward step and
compare the fused AllReduce+RMSNorm kernel (`use_ar_norm_fusion=true`) against the
non-fused path (`false`). Written for Llama-3.1-8B, TP2, B200.

The whole point is to profile **only the steady-state overlap step**, NOT the
engine init / `torch.compile` / warmup. Profiling everything either produces a
giant useless trace or hangs (see Gotchas).

---

## ⚠️ Prerequisite: the fused path needs working NVLS multicast (IMEX)

NanoFlow's fused kernel is flashinfer `trtllm_allreduce_fusion`, whose IPC
workspace setup calls `torch.distributed._symmetric_memory.rendezvous()`, which
creates a **multicast (NVLS) object**. That needs IMEX channels
(`/dev/nvidia-caps-imex-channels/`, the `nvidia-imex` service).

Check before you start:

```bash
python -c "import ctypes;ctypes.CDLL('libcuda.so.1').cuDeviceGetAttribute" # multicast cap
ls /dev/nvidia-caps-imex-channels/ 2>/dev/null && echo "IMEX present" || echo "NO IMEX -> fusion will HANG"
```

If IMEX is missing, the **`use_ar_norm_fusion=true` run hangs in `rendezvous()`**
at the first overlap step (silent worker hang, then `VLLM_EXECUTE_MODEL_TIMEOUT`
RPC timeout). Only the `false` (no-fusion, plain NCCL all-reduce) path can be
profiled there. This is an environment problem, not a code bug — see
`memory/tokenweave-ae-findings.md`. The original fusion numbers were captured on
GPU 6,7 when multicast worked.

---

## Core principle: never profile during `torch.compile`

Two independent reasons the naive `nsys profile vllm bench ...` fails:

1. **Trace bloat / noise** — engine init + dynamo/inductor compile + warmup dwarf
   the few ms of inference you actually care about.
2. **CUPTI slows compile to a crawl** — nsys injects CUPTI from process start;
   inductor autotuning launches thousands of probe kernels, each intercepted by
   CUPTI. The compile forward can blow past the 300 s worker RPC timeout and look
   like a deadlock (GPU 0% util, worker CPU ~100% = it's compiling, not hung).

The fix is two-fold:

- **Warm the `torch.compile` cache first** with a plain (no-nsys) run so the nsys
  run hits the disk cache and skips compile/autotune entirely.
- **Use nsys capture-range tied to `cudaProfilerStart/Stop`** so nsys records
  nothing until a chosen *post-compile* overlap step.

---

## Step 1 — instrument the scheduler (one-time patch)

`scheduler/vllm/nanoflow.py` is reverted to upstream by default. Apply this small,
inert (opt-in via `profile_step`) patch to fire `cudaProfilerStart/Stop` around
the Nth overlap step.

In `NanoFlowScheduler.__init__`, after `self.use_ar_norm_fusion = ...`:

```python
        # nsys profiling window (opt-in via dynaflow config "profile_step": N).
        self._profile_step = nanoflow_config.get("profile_step", None)
        self._overlap_step_count = 0
```

In `schedule()`, right after the `assert isinstance(attn_metadata_list, list)`:

```python
        do_profile = False
        if self._profile_step is not None and num_batches >= 2:
            self._overlap_step_count += 1
            if self._overlap_step_count == self._profile_step:
                do_profile = True
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
```

And at the very end of `schedule()` (after the `while batch_indices:` loop):

```python
        if do_profile:
            torch.cuda.synchronize()   # drain the step's async comm/comp streams
            torch.cuda.profiler.stop()
```

Notes:
- Gated on `num_batches >= 2` so we capture an actual 2-nano-batch **overlap**
  step (the interesting one). With `profile_step` unset the patch is a no-op, so
  it's safe to leave in for normal benchmarking.
- The `torch.cuda.synchronize()` calls bracket the step so all async kernels on
  `comm_stream`/`comp_stream` land inside the captured window.

---

## Step 2 — warm the compile cache (no nsys)

Run the exact same workload once without nsys so vLLM populates
`~/.cache/vllm/torch_compile_cache`. Engine init is ~10 s; total a couple minutes.

```bash
cd examples/ae
SCHED="$PWD/scheduler/vllm/nanoflow.py:NanoFlowScheduler"
export PATH=$PWD/.venv/bin:$PATH HF_HOME=/raid/catalyst/models
export CUDA_VISIBLE_DEVICES=4,5 VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
vllm bench throughput --model meta-llama/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 2 \
  --num-prompts 64 --input-len 1024 --output-len 8 --n 1 \
  --compilation-config '{"cudagraph_mode": "NONE"}' \
  --dynaflow-config "{\"scheduler_path\":\"$SCHED\",\"use_inductor\": false,\"min_nano_split_tokens\": 4096,\"max_num_splits\": 2,\"use_ar_norm_fusion\": false}"
```

The compiled FX graph is independent of `use_ar_norm_fusion` (fusion is a
*scheduling*-time choice), so one warm run covers both configs.

---

## Step 3 — capture the trace

Use the helper (it sets capture-range + the workload that triggers overlap):

```bash
bash profile_nanoflow.sh both      # or: fusion | nofusion
# -> results/nsys/nanoflow_fusion.nsys-rep, nanoflow_nofusion.nsys-rep
```

What the nsys flags do (`profile_nanoflow.sh`):

| flag | why |
|---|---|
| `--capture-range=cudaProfilerApi` | record nothing until `cudaProfilerStart()` (fired at overlap step `PROFILE_STEP=3`, post-compile) |
| `--capture-range-end=stop-shutdown` | finalize the trace right after that one step; app keeps running |
| `--trace=cuda,nvtx` | CUDA kernels/API + NVTX ranges |
| `--sample=none` | no CPU sampling (smaller trace) |
| `--trace-fork-before-exec=true` | follow the TP worker subprocesses (GPU work lives there) |

Workload (`--num-prompts 64 --input-len 1024 --output-len 8`): each prefill step
packs to 16384 > 2×4096 tokens, so DynaFlow splits into 2 nano-batches and the
overlap + fused path runs. `output-len 8` keeps the trace prefill-focused.

---

## Reading the two traces

Open the `.nsys-rep` in the Nsight Systems GUI, or summarize kernels headlessly:

```bash
nsys stats --report cuda_gpu_kern_sum --format table results/nsys/nanoflow_fusion.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format table results/nsys/nanoflow_nofusion.nsys-rep
```

What to compare:

1. **Two-stream overlap** — `comm_stream` and `comp_stream` run concurrently
   (one does GEMM/attention while the other does the all-reduce). Present in both
   configs (overlap is scheduling, independent of fusion).
2. **The all-reduce region — the key difference:**
   - **fusion=true:** ONE fused kernel per layer per nano-batch
     (`trtllm_allreduce_fusion` / flashinfer) doing allreduce+residual+rmsnorm.
   - **fusion=false:** TWO separate kernels — a NCCL/custom all-reduce
     (`ncclDevKernel_AllReduce...`) then a separate `rms_norm`/`fused_add_rms_norm`,
     with an extra HBM round-trip and kernel-launch gap between them.
3. **Per-step wall time** of the all-reduce region: fused is shorter, no gap.

The captured step contains all 32 transformer layers, so the per-layer kernel
pattern simply repeats — pick any one layer block to compare.

---

## Gotchas

- **`nsys profile vllm ...` "hangs" at 0% GPU / ~100% worker CPU** = it's
  compiling under CUPTI, not deadlocked. Warm the cache (Step 2) and use
  capture-range (Step 3). If you must profile cold, bump
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900`.
- **`use_ar_norm_fusion=true` hangs at the first overlap step** with no GPU
  activity → check IMEX (Prerequisite). The hang is inside `rendezvous()`.
- **TP workers not in the trace** → ensure `--trace-fork-before-exec=true`.
- **Need a stack of a hung worker?** py-spy needs ptrace (blocked here,
  `ptrace_scope=1`, no sudo). Instead register a SIGUSR1 dumper in the scheduler:
  `import faulthandler, signal; faulthandler.register(signal.SIGUSR1, all_threads=True)`,
  then `kill -USR1 <worker_pid>` — stacks go to the worker's stderr/log. Stagger
  signals per rank to separate interleaved output.
- **`min_nano_split_tokens` (4096)** must be < half the packed step tokens or no
  split happens and `schedule()`'s overlap path (hence the fused kernel) never
  runs — single-nano-batch falls back to `execute_single_batch`, which bypasses
  the scheduler.
