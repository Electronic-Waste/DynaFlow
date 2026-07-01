# MoE prefill profiling — Qwen3-30B-A3B vs 235B-A22B (B200)

What we measured and what it means for **Tensor∥CUDA-core (SiLU∥GEMM) overlap** in MoE.
All runs: vLLM (this tree), **dummy weights** (`load_format=dummy` — kernel timings are
value-independent), `enforce_eager`, single 2048-token prefill captured via the
`cudaProfilerStart/Stop` window in `qwen3_moe.py` (gated on `VLLM_NSYS_PROFILE_TOKENS`),
B200. Two models, both `Qwen3MoeForCausalLM`, top-8 of 128 experts:

| model | active | layers | hidden | moe_intermediate | parallelism | weights/GPU |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 3B | 48 | 2048 | 768 | TP2 + EP2 | 28.5 GiB |
| Qwen3-235B-A22B | 22B | 94 | 4096 | 1536 | TP4 + EP4 | 109.6 GiB |

> **Definitions.** *Tensor-bound jobs* = the GEMMs that run on Tensor Cores (the grouped
> `fused_moe` expert-GEMM + the attention q/k/v/o-proj GEMMs). *CUDA-bound jobs* = everything
> that runs on CUDA cores / SFU / memory and not Tensor Cores: RMSNorm, SiLU (`act_and_mul`),
> RoPE, KV-store, and the MoE routing/combine machinery (`topk-gating`, `moe_align`,
> `count_and_sort`, the expert-output `reduce`, elementwise). *Comm* = the NCCL all-reduce.

---

## 1. Tensor-bound vs CUDA-bound share (single 2048-tok prefill, dev0, clean no-metrics)

% of **GPU-busy** time (the kernel-time composition):

| group | **30B-A3B** | **235B-A22B** |
|---|---|---|
| **Tensor-bound** (expert-GEMM + attn-proj GEMM) | **44.0 %** | **51.6 %** |
| **CUDA-bound** (norm/silu/rope/kv/route/combine) | **28.2 %** | **24.0 %** |
| Comm (AllReduce) | 14.7 % | 16.7 % |
| Attention (flash) | 13.1 % | 7.7 % |

CUDA-bound sub-breakdown (% of busy):

| | 30B | 235B |
|---|---|---|
| SiLU (`act_and_mul`) | 11.0 % | 11.3 % |
| RMSNorm | 5.6 % | 4.0 % |
| MoE-combine (`reduce`) | 3.5 % | 3.8 % |
| elementwise / other | 4.1 % | 2.4 % |
| MoE-route (topk/align/sort) | 2.5 % | 1.5 % |
| RoPE | 1.5 % | 0.9 % |
| KV-store | ~0.9 % | ~0.6 % |

**Read:** CUDA-bound work is ~**24–28 %** of busy time; SiLU alone is ~11 % (the single biggest
CUDA-bound item, and the natural SiLU∥GEMM target). The bigger model has a *larger* Tensor
share (52 % vs 44 %) and a *smaller* CUDA-bound share — the expert-GEMMs grew faster than the
per-token elementwise work.

---

## 2. GPU idle / launch bubble — and it's a SMALL-model problem

| model | wall | busy | **idle** |
|---|---|---|---|
| 30B-A3B (TP2+EP2) | 64.4 ms | 35.4 ms (55 %) | **29.0 ms (45 %)** |
| 235B-A22B (TP4+EP4) | 125.1 ms | 115.3 ms (92 %) | **9.8 ms (8 %)** |

The 30B prefill is **45 % idle**; the 235B is only **8 %**. Same architecture — the difference is
kernel size. In eager mode every kernel pays a per-op CPU launch (Python/PyTorch dispatch →
`cudaLaunchKernel`, ~tens of µs). The 30B's MoE/routing/norm kernels are only a few µs on the
GPU, so the GPU finishes and **waits ~50–90 µs for the next launch** — launch-bound. Where the
30B idle sits (gap predecessor→successor): AllReduce→RMSNorm 23 %, RMSNorm→GEMM 23 %,
gating→`moe_align` 17 %, →expert-GEMM 14 %, attention→GEMM 10 %. **No per-layer device-sync or
D2H copy** — pure launch/dispatch + (EP) all-reduce sync-wait. The 235B's kernels are ~4–14× bigger
(22B active, hidden 4096), so the fixed launch overhead is amortized → near-fully busy.

> The launch bubble is fixed by **CUDA graphs** (replay kernels back-to-back, no per-op launch)
> or a **megakernel / persistent kernel** (FlashMoE arXiv 2506.04667, Mirage-MPK 2512.22219,
> UCCL mKernel) — *not* by overlap. It shrinks on its own as the model/batch grows.

---

## 3. When does the expert-GEMM turn Tensor-bound? (tokens/expert sweep)

Knob = **tokens/expert** ≈ `L · top8 / 128 = L/16` (L = prefill length); it is the M dim
(weight-reuse factor) of the per-expert grouped GEMM. Metric = **Tensor Active averaged over the
`fused_moe` kernels** (gb10x, dev0).

Chart (30B vs 235B overlay): **https://claude.ai/code/artifact/92dc766a-4282-4053-a6ef-6ea6b9cd38ba**

| tokens/expert | 30B Tensor% | **235B Tensor%** | 30B DRAM% | 235B DRAM% |
|---|---|---|---|---|
| 16 | 13.3 | 17.9 | 30.1 | 41.0 |
| 32 | 13.8 | 16.0 | 31.1 | 37.5 |
| 64 | 14.8 | 16.7 | 24.0 | 29.6 |
| 128 | 18.7 | 24.4 | 19.2 | 27.1 |
| 256 | 25.4 | 29.4 | 16.0 | 18.6 |
| 512 | 30.5 | 31.3 | 11.7 | 11.2 |
| 1024 | 32.9 | 28.8 | 7.9 | 5.9 |

- **Memory→compute crossover** (Tensor% overtakes DRAM%) at **~130 tok/expert** for 30B,
  **~180** for 235B (235B reads more weight/expert → higher DRAM → later crossover).
- **Both plateau at the same ~31–33 % Tensor ceiling.** 235B does *not* go higher (it's even
  lower at 1024). A bigger MoE does **not** make the expert-GEMM tensor-bound — arithmetic
  intensity is set by tokens/expert (weight reuse), and bigger experts scale compute *and*
  weight-read together. The ~33 % wall is a **kernel property** (untuned Triton grouped
  `fused_moe` on B200, `default config, sub-optimal`), not model size.

---

## 4. Bottom line for Tensor∥CUDA-core (SiLU∥GEMM) overlap

SiLU∥GEMM co-residency (the 1.39× warp-spec result, `up-silu-overlap.cu`) wins only when the
GEMM is **genuinely Tensor-bound** so the CUDA-core SiLU rides idle Tensor-pipe capacity. In MoE:

1. Below ~130–180 tok/expert the expert-GEMM is **memory-bound** → SiLU contends for HBM →
   overlap loses (the 0.70× regime). Don't.
2. Above it, neither 30B nor 235B exceeds **~33 % Tensor Active** with the current kernel → the
   complementary win is **muted**. **The lever is kernel efficiency, not model scale**: a tuned /
   CUTLASS grouped-GEMM that actually saturates Tensor Cores must come first; only then is there a
   high-Tensor GEMM to overlap SiLU onto. Scaling the model up won't unlock it.
3. The big idle in small-MoE prefill (45 %) is a **launch bubble**, not a missing-overlap problem —
   it's solved by CUDA graphs / megakernels and disappears at scale (235B: 8 %).

---

## Reproduce

```bash
cd examples/ae
# single prefill (set MODEL_PATH, TP, ENABLE_EP, VLLM_NSYS_PROFILE_TOKENS via env on the driver):
#   profile_vllm_prefill_nsys_70b.py   (Qwen3MoeForCausalLM capture window is in vllm/.../qwen3_moe.py)
# tokens/expert sweeps:
bash moe_tpe_sweep.sh        # 30B-A3B  TP2+EP2  GPUs 2,3
bash moe_tpe_sweep_235b.sh   # 235B-A22B TP4+EP4 GPUs 1-4
# analysis: scratchpad analyze_sweep.py / analyze_sweep235.py  (Tensor Active during fused_moe)
```
Traces: `results/nsys/qwen3moe{30b,235b}_*`, `results/nsys/sweep/`, `results/nsys/sweep235/`.
Models from `HF_HOME=/raid/catalyst/models` (`Qwen3-30B-A3B`, `Qwen3-235B-A22B`), dummy weights.
Caveat: random dummy router ≈ uniform expert load; absolute % shift with a tuned MoE kernel —
read the shape, the shared ~33 % ceiling, and the crossover, not the exact values.

---

## DeepEP all-to-all vs all-reduce EP (the EP communication strategies)

**Built DeepEP for Blackwell.** `deep-ep==1.1.0+e3908bf` (the commit vLLM pins) compiled for
**sm_100** — DeepEP officially targets only Hopper sm_90. Recipe: `CUDA_HOME=/usr/local/cuda-12.8`
(match torch cu128, not system nvcc 13.2), system **NVSHMEM 3.2.5** (deb, `NVSHMEM_DIR`=symlink
prefix; no patched-from-source build needed — single-node NVLink intranode P2P, fabricmanager
active, IBGDA off is fine), `TORCH_CUDA_ARCH_LIST=10.0 DISABLE_AGGRESSIVE_PTX_INSTRS=1` (drops the
`.L1::no_allocate` Hopper cache hints; dispatch/combine are pure comm kernels, no MMA). Build via
`uv pip install --no-build-isolation -e .`. See `_deepep_build/`, `profile_dp_ep.py`.

**vLLM only uses all-to-all EP when `dp_size > 1`** (`FusedMoEParallelConfig.use_all2all_kernels =
dp_size>1 and use_ep`). With pure **TP+EP (dp=1)** every rank already holds all tokens, so the MoE
combine is **always all-reduce** and `VLLM_ALL2ALL_BACKEND` is silently ignored (confirmed: TP4+EP4
deepep_high_throughput trace still showed only `two_shot_all_reduce`, zero dispatch/combine). So
"all-reduce EP vs all-to-all EP" == **TP+EP vs DP+EP** — a different parallelization, not a flag.

**Comm cost, Qwen3-30B-A3B, 2048 tok/rank, eager prefill, REAL weights, per-layer-per-rank (median per-call):**

| EP comm strategy | parallelism | per-layer comm kernels | per-layer comm |
|---|---|---|---:|
| all-reduce        | TP4+EP4 (dp1) | 2× all-reduce (1 attn + 1 MoE) @122µs        | 245µs |
| AllGather-ReduceScatter | DP4+EP4 | AllGather 100µs + Reduce 201µs              | 301µs |
| DeepEP all-to-all | DP4+EP4       | layout 17 + notify_dispatch 104 + dispatch 571 + combine 294 + notify_combine 118 | 1104µs |

Rigorous same-conditions back-to-back (both DP+EP): **DeepEP exposes 3.7× the comm-kernel time of
NCCL AllGather-ReduceScatter** (1104 vs 301µs/layer).

**Why DeepEP is the *most* expensive here (counterintuitive):** isolated **intranode + eager +
single-batch** defeats DeepEP's design — (1) eager = no compute to overlap dispatch/combine behind
(its whole point), (2) single-node NVLink = no internode RDMA scale where its low-SM async design
wins, (3) BF16 not FP8 dispatch, (4) 5-kernel NVSHMEM machinery has high fixed overhead. Simple NCCL
collectives beat it intranode. **DP also removes the attention all-reduce**, but the MoE all-to-all
dwarfs that saving. Takeaway: all-to-all EP comm is NOT free — it only pays off when overlapped with
compute (continuous batching / micro-batch overlap) or at internode scale. Ties to the
microbatch-overlap research gap.

**Dummy-weight trap (again):** dummy random router → pathological expert imbalance → combine barrier
spins (one `cached_notify_combine` waited **1.47s**); wall 1620ms (dummy) → 275ms (real). Always use
real weights for any routing/comm measurement. Traces: `results/nsys/deepep/q30dpR_{deepep,agrs}`.

**At 235B scale (Qwen3-235B-A22B DP4+EP4, real weights, hidden 4096, 94L, 2048 tok/rank), per-layer-per-rank comm (median):**

| EP comm strategy | per-layer comm kernels | per-layer comm |
|---|---|---:|
| AllGather-ReduceScatter | AllGather 134µs + ncclReduce_Sum 633µs | 767µs |
| DeepEP all-to-all | layout 17 + notify_dispatch 27 + dispatch 436 + combine 432 + notify_combine 545 | 1458µs |

DeepEP/AgRs ratio: **1.9× at 235B (hidden 4096)** vs 3.7× at 30B (hidden 2048) — **DeepEP's relative
penalty shrinks as the model widens** (per-call overhead amortizes over a larger payload; dispatch
436µs & combine 432µs are clean/symmetric with real-weight balanced routing, p90≈median).
Extrapolation: with wider hidden + FP8 dispatch + internode + compute overlap, DeepEP closes the gap
or wins — but in this single-node eager prefill it stays the heavier option. (235B needs
GPU_MEM_UTIL≤0.80: ~130GB/rank weights, and DeepEP's NVSHMEM symmetric buffer is allocated *outside*
vLLM's budget so a lower util leaves it room.) Traces `results/nsys/deepep/q235dpR_{deepep,agrs}`.
