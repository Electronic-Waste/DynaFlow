# Overlapping the up-projection GEMM with SiLU on B200

**Question:** can batch A's `SiLU(gate)*up` (CUDA-core / SFU / HBM-bound) be *hidden* behind
batch B's up/gate-projection GEMM (Tensor-Core bound) by running them concurrently **on the
same SMs** — a two-batch "vertical" pipeline (NanoFlow-style), *not* folding SiLU into the
GEMM epilogue?

**Answer (B200, Llama-3.1-8B): yes — a within-SM warp-spec kernel hits 1.39×, and a plain two-stream
launch (scheduler co-residency) gets 1.24× for free.** Against a **fair** SiLU baseline (vLLM's exact
`silu_and_mul`, 44µs — see below), the within-SM warp-spec kernel reaches **1.39×** at its optimum
(**2 MMA + 16 SiLU warps, WPB=18**), where the fused time (~112µs) ≈ the GEMM-only time (~110µs) —
i.e. the SiLU is **fully hidden**. There are **two knobs**, not one:

1. **MMA-warp count.** GEMM-only saturates the tensor core with a *single* MMA warp, but *inside the
   fused kernel* that lone warp shares its SMSP with hungry SiLU warps and can only issue
   `tcgen05.mma` at ~¼ rate → the tensor core starves and the GEMM stretches (1 MMA → 1.27×). A
   **2nd MMA warp** (landing in another SMSP) ~doubles MMA issue bandwidth under contention and keeps
   the GEMM fed (2 MMA → **1.39×**). 4 MMA warps is past the knee (CTA bloat).
2. **SiLU-warp count.** Enough to finish under the ~110µs GEMM (~16). Too few starves SiLU (7 warps /
   8-warp CTA → 192µs ≫ GEMM → **0.70×**); too many just bloats the CTA.

The fused kernel beats two-stream because SiLU/GEMM co-residency is *guaranteed* rather than left to
the block scheduler, **and** because the 2nd MMA warp is something two separate kernels can't express.
Kernel: [`up-silu-overlap.cu`](up-silu-overlap.cu).

```
M=2048, mma_warps=2, nacc=2, WPB=18 (the new default; stable across runs):
  GEMM(tcgen05)-only         : 110.3 us   (Tensor Active 87-89%, SM Issue ~7%; roofline ideal 107us)
  SiLU-only (vLLM, M blocks) :  44.2 us   <- FAIR baseline: vLLM-exact, optimal 1-block-per-token launch
  serial (GEMM + vLLM SiLU)  : 154.5 us
  2-stream (green-ctx iso, GEMM=96||SiLU=52): 221.9 us  speedup 0.70x  <- NanoFlow SM-isolation (LOSES; see below)
  FUSED warp-spec (2mma+16silu): 111.7 us  speedup 1.38x  <- WINS: SiLU fully hidden, fused ≈ GEMM-only
```

(The binary's "2-stream" line now runs the **green-ctx SM-isolated** variant — see the SM-isolation
section. The plain *scheduler co-residency* two-stream, measured separately, gives **1.24×**; that
result is documented below and is unchanged — it was removed from the binary when the 2-stream slot
was repurposed for the NanoFlow-style isolation baseline.)

**Two knobs: MMA-warp count and SiLU-warp count.** Holding the SiLU-warp count fixed, **2 MMA warps
beats 1 at every count** — so it is the 2nd MMA warp doing the work, not merely a wider CTA (FUSED
time / speedup, M=2048, nacc chosen so mma×nacc=4 accumulators in every cell):

| SiLU warps | 1 MMA warp | 2 MMA warps |
|---|---|---|
| 15 | WPB=16 → 122 µs · 1.27× | WPB=17 → 115 µs · 1.34× |
| 16 | WPB=17 → 128 µs · 1.21× | **WPB=18 → 112 µs · 1.39× ← optimum** |
| 17 | WPB=18 → 127 µs · 1.22× | WPB=19 → 112 µs · 1.39× |

**Why the 2nd MMA warp matters** — GEMM-only is ~110µs for 1/2/4 MMA warps alike (a single warp
saturates the tensor core *in isolation*; total accumulators = mma×nacc = 4 in all of these, so the
ILP budget is identical). But inside the fused kernel the MMA warp competes with the SiLU warps for
warp-scheduler issue slots: with ~16 warps over 4 SMSPs the lone MMA warp issues `tcgen05.mma` at ~¼
rate and the tensor core intermittently starves. A 2nd MMA warp lands in a different SMSP → ~2× MMA
issue bandwidth under contention → the GEMM portion compresses back to its ~110µs floor
(FUSED 112µs ≈ GEMM-only 110µs ⇒ SiLU fully hidden).

**SiLU-warp sweep, 1 MMA warp** (the original study; nacc=4) — its optimum is WPB=16, but every cell
is dominated by the matching 2-MMA config above:

| WPB | warp split | FUSED | speedup |
|---|---|---|---|
| 8  | 1mma + 7silu  | 220 µs | 0.70× (SiLU starved) |
| 14 | 1mma + 13silu | 129 µs | 1.20× |
| 16 | 1mma + 15silu | 122 µs | 1.27× |
| 32 | 1mma + 31silu | 135 µs | 1.15× (CTA bloated; steals issue from the lone MMA warp) |

With 1 MMA warp, adding SiLU warps past 15 even *regresses* (more warps further dilute the single MMA
warp's issue share). Past the knee 2-MMA also bloats: WPB 18→24→32 = 1.39→1.37→1.28×. **Hard cap
WPB≤32** (1024 threads/block), so "2 MMA + 32 SiLU" (34 warps = 1088 threads) won't launch. (Raising
residency via more blocks — 2 CTAs/SM, GRID=296 — also backfires: per-CTA TMEM-alloc overhead doubles.
Widening the CTA and adding the 2nd MMA warp is what works.)

---

## ⚠️ This corrects an earlier (wrong) conclusion

An earlier version of this study reported **1.74× for the within-SM fused kernel** and **1.0× for
two-stream**. Both were **artifacts of an unfair SiLU**:

- The old SiLU used a **rational sigmoid** `0.5+0.5·x/(1+|x|)` (1 fp32 divide, no `expf`) — *not*
  the function vLLM computes — and was measured **confined to 148 blocks** in *both* the baseline
  and the fused payload. That crippled baseline (~95µs) made the fused kernel look like a big win,
  and the equally-crippled 148-block SiLU in the 2-stream test had too few blocks to fill the
  GEMM's gaps, so 2-stream looked like 1.0×.
- **The fix:** make SiLU match vLLM exactly — `silu(x)=x/(1+expf(-x))`, reading the packed
  `gate_up[M, 14336]` tensor (gate=`[:, :7168]`, up=`[:, 7168:]`) — and use vLLM's launch geometry
  (one block per token) for the baseline. That baseline is **43µs**, matching the real
  `act_and_mul` (~45µs).

With the fair baseline the picture changed twice. First it looked like two-stream wins (1.23×) and
within-SM warp-spec **loses** (0.70×) — but that fused number was measured with an **8-warp CTA
(only 7 SiLU warps)**, which starves the SiLU. The fused kernel's SiLU parallelism is
`(CTAs/SM) × (SiLU warps/CTA) × 148`, and an 8-warp CTA gives far too few warps. **Widening the CTA
to 16 warps** (1 MMA + 15 SiLU) un-starves it and the fused kernel **wins (1.27×)**. So the real
lesson is: *both* mechanisms overlap; the fused one needs its SiLU-warp budget tuned.

---

## Both mechanisms overlap — here is the hardware proof, and the fused knob

- **SiLU is SFU(`expf`)-bound, not bandwidth-bound.** It moves only ~88MB (read gate_up 58.7MB +
  write out 29.4MB); even the optimal 44µs run is just **2.0 TB/s = 26% of HBM**. The cost is the
  `expf`, so SiLU throughput scales with how many warps/SFUs you keep busy — i.e. with
  **parallelism**, not bytes. This is why the fused kernel's SiLU-warp budget matters so much.
- **Two streams** launch SiLU with **2048 blocks × 1024 threads** (vLLM geometry). The
  tensor-saturated GEMM occupies only 148 blocks (1/SM) and leaves the warp schedulers / CUDA
  cores ~93% idle (SM Issue ~7%); the block scheduler drops the abundant small SiLU blocks onto
  that idle capacity on the same SMs. ~30µs of the 44µs SiLU is hidden → 125µs.
- **The fused kernel** keeps SiLU warps in the *same CTA* as the MMA warp, so co-residency is
  guaranteed. With 15 SiLU warps the confined SiLU is ~102µs — balanced against the ~110µs GEMM —
  and hides almost entirely → 122µs. It edges out two-stream because the GEMM stays **more**
  saturated (Tensor 87% vs 79%): no separate SiLU CTAs compete for SM residency.

**Hardware-counter proof of same-SM co-execution** (nsys gb10x GPU metrics, averaged over the
kernel-execution window only — *not* the whole capture, see Bug 2):

| regime | Tensor Active | SM Issue | Compute Warps | SMs Active | DRAM rd |
|---|---|---|---|---|---|
| GEMM-only | **88%** | 7% | 12% | 98% | 0.2% |
| SiLU-only | 0% | **70%** | **82%** | 92% | 3.1% |
| 2-stream | **79%** | **31%** | 37% | 99% | 1.6% |
| **FUSED (1mma+15silu)** | **87%** | **39%** | 24% | 97% | 1.0% |

In both overlap rows the **Tensor pipe runs the GEMM (79–87%, ≈ its solo 88%) WHILE SM-issue sits
at 31–39%** — far above GEMM-solo's 7%. Both pipe classes are lit *simultaneously* on the same 148
SMs: Tensor cores do the GEMM, CUDA cores / SFUs do the SiLU, concurrently. The fused kernel holds
Tensor at **87%** (vs two-stream's 79%) — guaranteed co-residency keeps the GEMM better fed, which
is the ~3µs edge. (DRAM <2% confirms SiLU rides idle *compute*, not spare bandwidth.)

**The fused knob is warps-per-CTA**, not blocks-per-SM. Raising CTAs/SM via more blocks (296+)
backfires — per-CTA TMEM-alloc overhead × more GEMM waves slows everything (GRID=296 → 0.66×).
Widening the CTA is what works; see the WPB table above (8→0.70×, 16→1.27×, 32→1.15×).

---

## SM isolation via green contexts (the NanoFlow/TokenWeave mechanism) — wrong for SiLU∥GEMM

NanoFlow/TokenWeave do **not** rely on the scheduler's co-residency. They **hard-partition the
SMs**: `scheduler/vllm/tokenweave.py` calls `green_ctx.split_device_green_ctx_by_sm_count(dev,[48])`
to reserve 48 SMs for the comm stream and leave the rest for compute (plus a `MAX_CTAS=48` grid cap
in the fused AR kernel). We replicated exactly that with the CUDA Green Context driver API
(`cuDevSmResourceSplitByCount` → `cuGreenCtxCreate` → `cuGreenCtxStreamCreate`): GEMM on one SM
pool, SiLU on the other (in [`up-silu-overlap.cu`](up-silu-overlap.cu), `bench_iso`):

| split | time | speedup |
|---|---|---|
| GEMM 120 SMs ∥ SiLU 28 | 227 µs | 0.68× |
| GEMM 96 ∥ SiLU 52 | 220 µs | 0.70× |
| GEMM 72 ∥ SiLU 76 | 328 µs | 0.47× |
| GEMM 48 ∥ SiLU 100 | 437 µs | 0.35× |

**Every split loses — the best (0.70×) is worse than plain serial (1.00×).** Why: SiLU and GEMM
bottleneck on *complementary* units, so co-residency lets **both kernels use all 148 SMs at once**
(GEMM the tensor cores, SiLU the CUDA cores). Isolation instead **throttles each kernel to a
fraction of the machine** — the GEMM gets only K SMs' tensor cores. Since the GEMM dominates, you
must give it ~all the SMs to keep it fast, leaving the SiLU almost nothing. The theoretical best
split (balance the two: `110/K = 44/(148−K)` → K≈106) only *ties* serial (~153µs); wave imbalance +
green-ctx overhead make every real split worse.

## The decision rule

This is the flip side of the comm∥GEMM story. Isolation is the **right** tool when the two kernels
**contend for the same units** (all-reduce's copy/reduce/issue/LSU/L2/NVLink vs the GEMM's): there
co-residency causes destructive interference, so you must separate them spatially — exactly the
NanoFlow/TokenWeave case. So the mechanism follows the resource profile:

| the two kernels use… | best mechanism | example (here) |
|---|---|---|
| **complementary** units (tensor ⟂ CUDA/SFU) | co-residency (shared SMs) or in-CTA warp-spec | **SiLU∥GEMM → up to 1.39×** (isolation → 0.7×) |
| **the same** units (issue/LSU/L2/NVLink) | **SM isolation** (green contexts) | comm∥GEMM (NanoFlow/TokenWeave) |

For SiLU∥GEMM the fused warp-spec (**1.39×** at 2 MMA + 16 SiLU) clearly beats the co-residency
2-stream (1.24×) — the 2nd MMA warp, which two separate kernels can't express, is what opens the gap
(two streams still gets the easy ~1.24× for none of the code). Hard SM isolation, NanoFlow's actual
mechanism, is a net **loss** here because it is solving the wrong problem.

---

## Setup / reference numbers

- GPU: **NVIDIA B200 (GB100, 148 SMs)**, bf16 in / fp32 accumulate, peak ~2250 TFLOP/s, HBM ~8 TB/s.
- Workload: Llama-3.1-8B FFN, TP2 **per-rank** `gate_up` `[M,4096]@[4096,14336]` (14336=2×7168),
  then SwiGLU `silu(gate)*up : [M,14336]→[M,7168]`.
- Real per-layer cost (nsys, M≈2000): real CUTLASS gate_up GEMM ≈ **157µs @ 74.8% Tensor Active**;
  SiLU (`act_and_mul`) ≈ **45µs** — our 44µs baseline matches it.
- SiLU here is a **byte-for-byte port of vLLM** `csrc/activation_kernels.cu`
  (`act_and_mul_kernel<silu_kernel, act_first=true>`): `out[t,j] = (g/(1+expf(-g)))·u`,
  `g=gate_up[t,j]`, `u=gate_up[t,D+j]`, loads via `__ldg`, launch `grid=num_tokens, block=min(d,1024)`.
  **vLLM's kernel is per-element, NOT vectorized** — so its 44µs is *not* clever code, it is purely
  **launch parallelism**: 2048 blocks × 1024 threads ≈ 2.1M threads feed all 148 SMs' SFUs (each
  thread does only 7 `expf`s). Confine that *same code* to the fused kernel's 148 blocks × **7**
  warps and it is 192µs (4.4× slower from geometry alone); give it **15** warps (WPB=16) and it
  drops to ~102µs — fast enough to hide. This warp budget is the whole ballgame for the fused kernel.

---

## Two bugs that nearly sank this (both real, both fixed)

**Bug 1 — the hang (warp-convergent alloc).** `tcgen05.alloc.cta_group::1.sync.aligned` (and
`dealloc`) are **warp-convergent**: all 32 lanes must execute them. Guarding with
`if (threadIdx.x==0)` runs it on one lane → the `.sync` waits forever → deadlock. Bisected with a
minimal single-warp test + global progress markers. **Fix:** `if (threadIdx.x < 32) alloc.allocate(...)`.

**Bug 2 — the fake "5% Tensor Active" (measurement error).** Averaging GPU-metric samples over the
*whole* nsys capture includes CUDA-init / first-launch idle, diluting the GEMM's Tensor Active to
~5%. **Fix:** average only over **kernel-execution windows** (CUPTI kernel intervals). Properly
measured the GEMM is **63–89% Tensor Active**; roofline confirms (120.6µs for 0.24 TFLOP ≈ 1995
TFLOP/s ≈ 89% of peak).

Other must-haves for a non-hanging tcgen05 MMA: valid SMEM descriptors via
`cute::UMMA::make_umma_desc<Major::K>` on a `tile_to_shape(Layout_K_SW128_Atom<bf16>,…)` tensor;
`make_runtime_instr_desc<bf16,bf16,float,M,N,K,K>()`; completion handshake `tcgen05.commit …
mbarrier::arrive::one` → `mbarrier.try_wait.parity` → then `dealloc`; build for **`sm_100a`**.

---

## Making the in-CTA GEMM actually saturate the Tensor Core

(Still useful: the GEMM payload is genuinely tensor-saturated, which is what makes "the SiLU rides
the idle SM capacity" a fair test.) The GEMM is a synthetic loop (zeroed SMEM tiles, looped
`tcgen05.mma`; output garbage, **timing valid**). To saturate it needs **bigger N tiles + ILP**
(independent TMEM accumulators to hide MMA latency):

| tile N | ILP (nacc) | #MMA warps | % of peak |
|---|---|---|---|
| 64  | 1 | 2 | ~63% (original) |
| 128 | 1 | 2 | 79% |
| 128 | 4 | 1 | 96% |
| **128** | **2** | **2** | **96%** ← used (2 MMA warps × 2 acc = 4 acc; the other 16 of an 18-warp CTA do SiLU) |
| 256 | 2 | 1 | 93% |

GEMM-only saturation depends only on the **total** accumulator count (mma_warps × nacc = 4 here), not
on how it is split across warps — so 1×4 and 2×2 both hit ~96%. The split to **2 MMA warps** is for
the *fused* case (MMA issue bandwidth under SiLU contention), not for standalone GEMM throughput. In
the full kernel it measures **87–89% Tensor Active**. Constraint: `mma_warps × nacc × N ≤ 512` TMEM
columns.

---

## Caveats (honest)

- **The GEMM is synthetic** (zeroed tiles, looped MMA). It is genuinely tensor-saturated, so the
  *idle-capacity* argument holds — but a **real** CUTLASS GEMM uses more SMEM + warps for its TMA
  load pipeline. For **two-stream** that means fewer resident slots for co-resident SiLU blocks
  (1.24× headroom could shrink); for the **fused** kernel it means SiLU shares register/SMEM budget
  with a heavier MMA path (the 1.27× could shrink too, and the WPB sweet spot would move). Both
  numbers are upper-ish bounds; the *qualitative* result (both overlap, warp budget is the knob)
  is robust.
- **SiLU∥GEMM is the easy case.** SiLU is cheap (44µs) and even a plain two-stream launch overlaps it
  nearly for free (1.24×). The hand-written kernel's extra reach to **1.39×** comes from the 2nd MMA
  warp keeping the GEMM fed under issue contention — a real gain here, but the warp-spec/SM-partition
  machinery earns its complexity most on jobs that contend for SM *residency* (all-reduce) — the
  **communication∥GEMM** case NanoFlow/TokenWeave target.

## Build & run

```bash
INC=.venv/lib/python3.10/site-packages/flashinfer/data/cutlass/include   # CuTe headers
nvcc -gencode=arch=compute_100a,code=sm_100a -O3 -std=c++17 -I$INC up-silu-overlap.cu -o up_silu -lcuda
CUDA_VISIBLE_DEVICES=<gpu> ./up_silu <input_len_M=2048> <mma_warps=1> <nacc=4>   # default WPB=16
# -lcuda is for the green-context (driver API) SM-isolation sweep printed at the end.
WARPS=<n> CUDA_VISIBLE_DEVICES=<gpu> ./up_silu 2048 1 4   # warps/CTA sweep (the fused knob; 16 is best)
GRID=<n>  CUDA_VISIBLE_DEVICES=<gpu> ./up_silu 2048 1 4   # override #blocks (CTAs/SM sweep)
```
