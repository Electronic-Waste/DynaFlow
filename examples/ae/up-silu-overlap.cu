// up-silu-overlap.cu — Can a SwiGLU SiLU (CUDA-core/SFU op) fully overlap the up-projection GEMM
// (Tensor-Core op) on the SAME SMs of a B200? Yes -- and this file shows HOW to make it hold at ANY
// batch size. Two mechanisms are compared:
//   (A) WITHIN-SM warp specialization (one fused kernel, async tcgen05 MMA):  <- the winner
//         warps 0..mma_warps-1                : issue tcgen05.mma into per-warp TMEM   (Tensor Core)
//         warps mma_warps..mma_warps+silu-1   : SiLU(gate)*up                          (CUDA cores/SFU)
//         (optional) dedicated loader warps   : stream the GEMM weights from HBM (env STREAMW=1)
//         ...all in ONE CTA -> same SM, co-residency GUARANTEED.
//   (B) TWO-STREAM via SM ISOLATION (NanoFlow/TokenWeave style): green contexts hard-partition the SMs
//         into a GEMM pool (env SMG) + a SiLU pool; each kernel runs on its own slice. Always LOSES here
//         (0.5-0.7x): SiLU and GEMM use COMPLEMENTARY units (Tensor ∥ CUDA cores), so splitting throttles
//         both, AND the 148-block GEMM confined to <148 SMs takes ceil(148/SMG) waves.
//
// ===================== THE KEY FINDING (why this file exists) =====================
// A NAIVE scalar SiLU (1 bf16 per __ldg) shows a deceptive 1.39x at M=2048 -- but that is an L2-RESIDENCY
// ARTIFACT: repeated-launch timing keeps gate_up warm in the 126MB L2. At large batch (M>=4096) gate_up
// (>=112MB) spills L2, the SiLU reads COLD from HBM, and -- being LATENCY-bound at the fused kernel's low
// occupancy (~28%) -- it leaks past the GEMM window => overlap collapses to ~1.04x.
//   (Proof: cold-L2 flush drops M=2048 from 1.39x->1.08x; ncu shows L2 hit-rate ~0% once gate_up > L2.)
//
// FIX -- make the in-fused SiLU BANDWIDTH-bound, not latency-bound, so it hides regardless of where
//        gate_up lives (L2 or HBM):
//   (1) VECTORIZE: 128-bit (int4 = 8 bf16) loads/stores, perfect coalescing, 8x fewer transactions.
//   (2) PREFETCH depth 4: issue 4 groups' loads BEFORE consuming -> deep memory-level parallelism that
//       hides HBM latency even at low occupancy (Little's law). PREF=8 overflows the register file -> don't.
//   (3) GEMM config mma_warps=4, nacc=1: spreads tcgen05 issue over 4 warps (TC stays fed, GEMM compute-
//       bound) AND leaves SM issue bandwidth for the SiLU warps -> smallest residual leak.
// RESULT: FUSED within ~1.5% of GEMM-only at EVERY M from 2048..16384 (flat ~1.25x, SIZE-INDEPENDENT),
//   and it HOLDS when the GEMM streams its real 112MB weights from HBM (STREAMW=1: 1.015 at M=8192/16384).
//   ncu confirms genuinely HBM-bound: at M=8192 the fused kernel reads 352MB from DRAM, L2 hit-rate 0.04%,
//   only ~12% of HBM bandwidth. The overlap NO LONGER depends on L2 residency.
//
// ============================ CONFIGURATION ============================
// Target : NVIDIA B200 (sm_100a, GB100, 148 SMs), bf16 in / fp32 accumulate. Maps to Llama-3.1-8B TP2:
//          gate_up [M,4096]x[4096,14336]; SiLU on packed gate_up[M, 2*7168] -> out[M,7168].
// GEMM (Tensor Core, tcgen05): SM100_MMA_F16BF16_SS<bf16,bf16,float,128,128,K> (128x128x16 / inst). Each
//          MMA warp owns `nacc` independent TMEM accumulators (ILP). TMEM budget mma_warps*nacc*128 <= 512.
//          A/B are zeroed SMEM tiles (tensor-throughput PoC); STREAMW=1 adds STREAMWARPS dedicated warps
//          that stream the real 112MB weight from HBM, reproducing production L2/HBM traffic on separate
//          warps (so it does NOT steal tcgen05 issue slots -> GEMM stays compute-bound ~same time).
// SiLU (CUDA cores/SFU): o[t,j]=silu(g)*u=(g/(1+expf(-g)))*u (vLLM-exact). Vectorized 128-bit + prefetch-4
//          (silu8 / silu_strided / silu_vllm) -> bandwidth-bound, hides under the GEMM at any batch size.
//
// build: nvcc -gencode=arch=compute_100a,code=sm_100a -O3 -std=c++17 -I<cutlass_inc> up-silu-overlap.cu -o up_silu -lcuda
// run  : ./up_silu <M=2048> <mma_warps=4> <nacc=1>
//        env: WARPS=warps/CTA[18], GRID=#blocks[148], SMG=GEMM-pool SMs for green-ctx[96],
//             STREAMW=1 -> GEMM streams real W from HBM (realistic L2 contention),
//             LOOP=<n> LOOPMODE=<0 GEMM|1 SiLU|2 FUSED|3 2-stream> -> n back-to-back launches for nsys.
// ======================================================================
#include <cuda_runtime.h>
#include <cuda.h>          // driver API: Green Contexts -> HARD SM isolation (NanoFlow/TokenWeave style)
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
using namespace cute;
#define WARP 32
#define STREAMWARPS 8      // dedicated warps/CTA that stream W from HBM when STREAMW=1 (model GEMM weight reads)
static int WPB=18;         // warps per CTA (host launch); kernel reads blockDim.x/WARP. env WARPS overrides.
                           // 18 = 4 MMA + 14 SiLU (the bandwidth-bound optimum). With STREAMW=1, use a bigger
                           // WARPS (e.g. 28 = 4 MMA + 8 stream + 16 SiLU) so SiLU keeps enough warps.
#define NBLOCKS 148
constexpr int MM=128, NN=128, KK=64;   // per-accumulator UMMA tile; TMEM cols/acc = NN

__device__ __forceinline__ void mbar_init(uint32_t mb){ asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(mb)); }
__device__ __forceinline__ void mma_commit(uint32_t mb){ asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(mb)); }
__device__ __forceinline__ void mbar_wait(uint32_t mb){ asm volatile("{.reg .pred P;L:mbarrier.try_wait.parity.shared::cta.b64 P,[%0],0;@!P bra L;}"::"r"(mb)); }

constexpr int D=7168;   // INTER_SHARD; SiLU input is packed gate_up[M, 2*D], output[M, D]
// Reference scalar form (byte-for-byte vLLM csrc/activation_kernels.cu act_and_mul_kernel<silu_kernel>):
//   silu_kernel(x)=x/(1+expf(-x)) (precise expf, NOT __expf). UNUSED by the hot paths -- they use the
//   vectorized silu8 below -- kept only to document the exact op.
__device__ __forceinline__ void silu_one(const __nv_bfloat16* gu, __nv_bfloat16* o, long t, int j){
    float g=__bfloat162float(__ldg(&gu[t*2L*D + j]));         // gate  (read-only cache, like vLLM)
    float u=__bfloat162float(__ldg(&gu[t*2L*D + D + j]));     // up
    o[t*(long)D + j]=__float2bfloat16( (g/(1.0f+expf(-g))) * u );
}
// one 8-channel (128-bit) SiLU group, registers already loaded
__device__ __forceinline__ void silu8(const int4& gv, const int4& uv, int4& ov){
    const __nv_bfloat16* g16=(const __nv_bfloat16*)&gv; const __nv_bfloat16* u16=(const __nv_bfloat16*)&uv;
    __nv_bfloat16* o16=(__nv_bfloat16*)&ov;
    #pragma unroll
    for(int e=0;e<8;e++){ float g=__bfloat162float(g16[e]), u=__bfloat162float(u16[e]);
        o16[e]=__float2bfloat16( (g/(1.0f+expf(-g)))*u ); }
}
// FAST in-fused SiLU: 128-bit (8xbf16) loads + PREF-deep prefetch -> high memory-level parallelism, so it
// is BANDWIDTH-bound (not latency-bound) even at the fused kernel's low occupancy. This lets it hide under
// the GEMM at ANY batch size -- it no longer needs gate_up to be L2-resident (the large-batch fix).
__device__ __forceinline__ void silu_strided(const __nv_bfloat16* gu, __nv_bfloat16* o,
                                             int M, long tid, long nt){
    const int PREF=4;     // 4-deep prefetch = sweet spot (PREF=8 overflows the register file -> launch fails)
    long groups=(long)M*(D/8);                            // 8 channels per group (16B aligned)
    for(long g0=tid; g0<groups; g0+=(long)PREF*nt){
        int4 gv[PREF], uv[PREF]; long gi[PREF];
        #pragma unroll
        for(int p=0;p<PREF;p++){ gi[p]=g0+(long)p*nt;     // issue ALL loads first -> deep MLP
            if(gi[p]<groups){ long t=gi[p]/(D/8); int j=(int)((gi[p]-t*(D/8))*8);
                const __nv_bfloat16* b=gu+t*2L*D+j; gv[p]=__ldg((const int4*)b); uv[p]=__ldg((const int4*)(b+D)); } }
        #pragma unroll
        for(int p=0;p<PREF;p++){ if(gi[p]<groups){ long t=gi[p]/(D/8); int j=(int)((gi[p]-t*(D/8))*8);
            int4 ov; silu8(gv[p],uv[p],ov); *((int4*)(o+t*(long)D+j))=ov; } }
    }
}
// vLLM-style optimal baseline, vectorized 128-bit (matches a tuned vLLM act_and_mul; one block per token)
__global__ void silu_vllm(const __nv_bfloat16* gu, __nv_bfloat16* o, int M){
    const __nv_bfloat16* gb=gu+(long)blockIdx.x*2L*D; __nv_bfloat16* ob=o+(long)blockIdx.x*D;
    for(int j=threadIdx.x*8; j<D; j+=blockDim.x*8){
        int4 gv=__ldg((const int4*)(gb+j)), uv=__ldg((const int4*)(gb+D+j)), ov; silu8(gv,uv,ov);
        *((int4*)(ob+j))=ov;
    }
}

// mode 0=GEMM-only(mma_warps issue),1=SiLU-only,2=FUSED
__global__ void ws(int mode,int mma_warps,int nacc,int iters,const __nv_bfloat16* gu,__nv_bfloat16* o,int M,
                   const __nv_bfloat16* w,long wn){
    int wib=threadIdx.x/WARP;
    __shared__ bfloat16_t sA[MM*KK];
    __shared__ bfloat16_t sB[NN*KK];
    __shared__ uint32_t tmem_slot[4];
    __shared__ uint64_t mbar_s[1];
    uint32_t mb=cast_smem_ptr_to_uint(mbar_s);
    int tmem_cols = mma_warps*nacc*NN;                   // TMEM columns actually used

    int wpb = blockDim.x/WARP;
    int stream_warps = (wn>0 && mode!=1) ? STREAMWARPS : 0;       // dedicated weight-loader warps
    if(stream_warps > wpb-mma_warps) stream_warps = (wpb>mma_warps)?(wpb-mma_warps):0;
    bool do_mma   = (mode!=1 && wib<mma_warps);
    bool do_stream= (stream_warps && wib>=mma_warps && wib<mma_warps+stream_warps);
    bool do_silu  = (mode==1) || (mode==2 && wib>=mma_warps+stream_warps);

    for(int i=threadIdx.x;i<MM*KK;i+=blockDim.x) sA[i]=bfloat16_t(0);   // parallel zero (cheap with big CTA)
    for(int i=threadIdx.x;i<NN*KK;i+=blockDim.x) sB[i]=bfloat16_t(0);
    if(threadIdx.x==0) mbar_init(mb);
    __syncthreads();
    TMEM::Allocator1Sm alloc;
    if(mode!=1 && threadIdx.x<32) alloc.allocate(tmem_cols, tmem_slot);  // warp-convergent; SiLU-only skips
    __syncthreads();
    uint32_t tmem_base=tmem_slot[0];

    if(do_mma){
        auto layA = tile_to_shape(UMMA::Layout_K_SW128_Atom<bfloat16_t>{}, Shape<Int<MM>,Int<KK>>{});
        auto layB = tile_to_shape(UMMA::Layout_K_SW128_Atom<bfloat16_t>{}, Shape<Int<NN>,Int<KK>>{});
        Tensor tA = make_tensor(make_smem_ptr(sA), layA);
        Tensor tB = make_tensor(make_smem_ptr(sB), layB);
        uint64_t da = UMMA::make_umma_desc<UMMA::Major::K>(tA);
        uint64_t db = UMMA::make_umma_desc<UMMA::Major::K>(tB);
        uint64_t idesc = UMMA::make_runtime_instr_desc<bfloat16_t,bfloat16_t,float,MM,NN,UMMA::Major::K,UMMA::Major::K>();
        for(int k=0;k<iters;k++)                          // K-steps; nacc independent accumulators -> ILP
            for(int a=0;a<nacc;a++)
                SM100_MMA_F16BF16_SS<bfloat16_t,bfloat16_t,float,MM,NN,UMMA::Major::K,UMMA::Major::K>
                    ::fma(da,db, tmem_base+(wib*nacc+a)*NN, (k?1u:0u), idesc);
    } else if(do_stream){
        // REALISTIC HBM TRAFFIC on DEDICATED loader warps: stream the full weight matrix W from HBM,
        // flooding L2 (evicting the SiLU's gate_up) like a production up-proj GEMM -- but on separate
        // warps, so it does NOT steal tcgen05 issue slots from the MMA warps (which would starve the TC).
        long tid = ((long)blockIdx.x*stream_warps + (wib-mma_warps))*WARP + (threadIdx.x%WARP);
        long nt  = (long)gridDim.x*stream_warps*WARP;
        long n4  = wn/8;                                  // # of 16B (float4 = 8 bf16) chunks in W
        float4 acc = make_float4(0,0,0,0);
        for(long i=tid;i<n4;i+=nt){ float4 v=__ldg(((const float4*)w)+i); acc.x+=v.x;acc.y+=v.y;acc.z+=v.z;acc.w+=v.w; }
        if(mode==0x7fffffff) ((float4*)o)[blockIdx.x]=acc;   // DCE guard, never executes
    } else if(do_silu){
        int silu_warps = (mode==1) ? wpb : (wpb - mma_warps - stream_warps);
        int my_warp    = (mode==1) ? wib : (wib - mma_warps - stream_warps);
        int lane       = threadIdx.x % WARP;
        long thread_id  = ((long)blockIdx.x * silu_warps + my_warp) * WARP + lane;
        long thread_cnt = (long)gridDim.x * silu_warps * WARP;
        silu_strided(gu, o, M, thread_id, thread_cnt);
    }
    __syncthreads();
    if(mode!=1 && threadIdx.x==0){ mma_commit(mb); mbar_wait(mb); }   // MMA completion
    __syncthreads();
    if(mode!=1 && threadIdx.x<32) alloc.free(tmem_base,tmem_cols);    // warp-convergent
}

// ---- device buffers + run config (set in main, read by the launchers below) ----
static __nv_bfloat16 *dGU, *dO;     // packed gate_up[M,2D] and out[M,D]
static __nv_bfloat16 *dW=nullptr;   // real gate_up weight [4096,14336] in HBM (streamed by the GEMM)
static long dWn=0;                   // weight element count; >0 => GEMM streams W from HBM (env STREAMW)
static int MMAW, NACC, ITERS;       // MMA warps, accumulators/warp, K-steps/accumulator
static int Mtok;                    // input length M (tokens)
static int GRID = NBLOCKS;          // #blocks launched (default 1 per SM)

void rg() { ws       <<<GRID, WPB*WARP>>>(0, MMAW, NACC, ITERS, dGU, dO, Mtok, dW, dWn); } // GEMM-only
void rs() { silu_vllm<<<Mtok, 1024>>>   (dGU, dO, Mtok); }                                 // vLLM SiLU (baseline)
void rf() { ws       <<<GRID, WPB*WARP>>>(2, MMAW, NACC, ITERS, dGU, dO, Mtok, dW, dWn); } // FUSED within-SM
// Time `it` launches of `f` (after `wu` warm-up launches); returns microseconds per launch.
float tim(void(*f)(), int it=50, int wu=15){
    for(int i=0;i<wu;i++) f();
    if(cudaDeviceSynchronize() != cudaSuccess){
        printf("  [err: %s]\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0;i<it;i++) f();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / it * 1000.0f;
}

// ---- NanoFlow / TokenWeave style: HARD SM isolation via CUDA Green Contexts ----
// This is what scheduler/vllm/tokenweave.py does: split_device_green_ctx_by_sm_count(dev,[48])
// reserves 48 SMs for the comm stream, the rest for compute. Here we split the device into a
// GEMM pool (sm_g SMs) and a SiLU pool (the remainder) and run each kernel on its own pool.
#define CUCK(x) do{ CUresult _r=(x); if(_r!=CUDA_SUCCESS){ const char* _s; cuGetErrorString(_r,&_s); \
    printf("    [cu err %s: %s]\n", #x, _s); return false; } }while(0)

// Time the isolated 2-stream: GEMM on `gpool`, SiLU on `spool` (each a green-context stream).
float tim_2stream_iso(cudaStream_t gpool, cudaStream_t spool, int it=50, int wu=15){
    auto launch_both = [&](){
        ws       <<<GRID, WPB*WARP, 0, gpool>>>(0, MMAW, NACC, ITERS, dGU, dO, Mtok, dW, dWn);  // GEMM on its SMs
        silu_vllm<<<Mtok, 1024,     0, spool>>>(dGU, dO, Mtok);                         // SiLU on its SMs
    };
    for(int i=0;i<wu;i++) launch_both();
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<it;i++) launch_both();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float,std::micro>(t1 - t0).count() / it;
}

// Build a green-context SM split: a GEMM pool (sm_g SMs) + a SiLU pool (the remainder). Hands back
// the two green-ctx streams + their green contexts (caller destroys both) and the ACTUAL SM counts
// (the driver rounds to a multiple of the co-scheduled alignment, 8 on B200). false if unavailable.
bool make_green_split(int sm_g, CUstream* sG, CUstream* sS, CUgreenCtx* gG, CUgreenCtx* gS,
                      int* got_g, int* got_s){
    CUdevice dev;
    CUdevResource whole, group, remainder;
    unsigned int nb = 1;
    CUCK(cuDeviceGet(&dev, 0));                                       // CUDA_VISIBLE_DEVICES pins the phys GPU
    CUCK(cuDeviceGetDevResource(dev, &whole, CU_DEV_RESOURCE_TYPE_SM));
    CUCK(cuDevSmResourceSplitByCount(&group, &nb, &whole, &remainder, 0, (unsigned)sm_g));
    CUdevResourceDesc descG, descS;
    CUCK(cuDevResourceGenerateDesc(&descG, &group,     1));
    CUCK(cuDevResourceGenerateDesc(&descS, &remainder, 1));
    CUCK(cuGreenCtxCreate(gG, descG, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUCK(cuGreenCtxCreate(gS, descS, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUCK(cuGreenCtxStreamCreate(sG, *gG, CU_STREAM_NON_BLOCKING, 0));
    CUCK(cuGreenCtxStreamCreate(sS, *gS, CU_STREAM_NON_BLOCKING, 0));
    *got_g = group.sm.smCount;
    *got_s = remainder.sm.smCount;
    return true;
}

// Time the green-ctx isolated 2-stream (GEMM pool || SiLU pool), then tear down.
bool bench_iso(int sm_g, float* out_us, int* got_g, int* got_s){
    CUstream sG, sS; CUgreenCtx gG, gS;
    if(!make_green_split(sm_g, &sG, &sS, &gG, &gS, got_g, got_s)) return false;
    *out_us = tim_2stream_iso((cudaStream_t)sG, (cudaStream_t)sS);
    cuStreamDestroy(sG);   cuStreamDestroy(sS);
    cuGreenCtxDestroy(gG); cuGreenCtxDestroy(gS);
    return true;
}

int main(int c,char**v){
    int INPUT_LEN = c>1?atoi(v[1]):2048;     // input length M (tokens)
    MMAW          = c>2?atoi(v[2]):4;        // MMA-issuing warps (best: 4 -> spreads tcgen05 issue, leaves
    NACC          = c>3?atoi(v[3]):1;        // SM issue BW for SiLU). nacc=1 -> mma*nacc=4 accumulators.
    if(MMAW*NACC*NN>512){ printf("TMEM overflow: mma_warps*nacc*NN=%d>512\n",MMAW*NACC*NN); return 1; }
    const long HIDDEN=4096, INTER_SHARD=7168, GU=2*INTER_SHARD;
    Mtok = INPUT_LEN;
    if(getenv("GRID")) GRID=atoi(getenv("GRID"));                      // override #blocks (CTAs/SM sweep)
    if(getenv("WARPS")) WPB=atoi(getenv("WARPS"));                     // warps/CTA: bigger CTA -> more SiLU warps
    int sm_g = getenv("SMG") ? atoi(getenv("SMG")) : 96;               // GEMM-pool SMs for the green-ctx 2-stream
    long total_mma = (long)INPUT_LEN * HIDDEN * GU / (128L*NN*16);     // real gate_up MMAs (128xNNx16)
    ITERS = (int)(total_mma / ((long)GRID * MMAW * NACC));             // K-steps per accumulator
    if(ITERS < 1) ITERS = 1;
    cudaMalloc(&dGU,(long)Mtok*GU*2);                                 // packed gate_up[M, 2*7168] bf16
    cudaMalloc(&dO ,(long)Mtok*INTER_SHARD*2);                        // out[M, 7168] bf16
    cudaMemset(dGU,0,(long)Mtok*GU*2);
    // STREAMW=1: allocate the real gate_up weight [4096,14336] in HBM and have the GEMM stream it from HBM
    // (dedicated loader warps flood L2 with ~112MB of weight reads, like production -> realistic L2/HBM
    // contention). The fix above keeps FUSED ~= GEMM-only even under this.
    if(getenv("STREAMW") && atoi(getenv("STREAMW"))){
        dWn = HIDDEN*GU;                                              // 4096*14336 = 58.7M bf16 = 112 MB
        cudaMalloc(&dW, dWn*2); cudaMemset(dW,0,dWn*2);
        printf("[STREAMW] GEMM streams W[%ld,%ld] = %.0f MB from HBM (%d loader warps/CTA)\n",
               HIDDEN, GU, dWn*2/1048576.0, STREAMWARPS);
    }
    // LOOP=<n> runs <n> iterations of one mode back-to-back (for nsys profiling), then exits.
    // LOOPMODE: 0=GEMM-only, 1=vLLM SiLU, 2=FUSED, 3=green-ctx isolated 2-stream (env SMG).
    if(getenv("LOOP")){
        int n = atoi(getenv("LOOP"));
        int m = getenv("LOOPMODE") ? atoi(getenv("LOOPMODE")) : 0;
        CUstream sG=0, sS=0; CUgreenCtx gG=0, gS=0; int gg=0, gs=0;
        if(m==3){                                                               // green-ctx SM isolation
            cuInit(0);
            if(!make_green_split(sm_g, &sG, &sS, &gG, &gS, &gg, &gs)){
                printf("(green contexts unavailable here)\n"); return 1; }
        }
        for(int i=0;i<n;i++){
            if      (m==0) rg();
            else if (m==1) rs();
            else if (m==2) rf();
            else {                                                              // m==3: green-ctx 2-stream
                ws       <<<GRID, WPB*WARP, 0, (cudaStream_t)sG>>>(0, MMAW, NACC, ITERS, dGU, dO, Mtok, dW, dWn);
                silu_vllm<<<Mtok, 1024,     0, (cudaStream_t)sS>>>(dGU, dO, Mtok);
            }
        }
        cudaDeviceSynchronize();
        if(m==3){ cuStreamDestroy(sG); cuStreamDestroy(sS); cuGreenCtxDestroy(gG); cuGreenCtxDestroy(gS); }
        return 0;
    }

    // Measure each regime, then report speedups against the serial baseline.
    float t_gemm    = tim(rg);          // GEMM only
    float t_silu    = tim(rs);          // vLLM SiLU, optimal launch (the fair baseline)
    float t_fused   = tim(rf);          // within-SM warp specialization
    float serial    = t_gemm + t_silu;

    // The 2-stream baseline is NanoFlow/TokenWeave-style: HARD SM isolation via green contexts
    // (GEMM pool of `sm_g` SMs || SiLU pool of the remainder). env SMG overrides the split.
    cuInit(0);
    float t_2stream; int gg = 0, gs = 0;
    bool iso_ok = bench_iso(sm_g, &t_2stream, &gg, &gs);

    printf("M=%d  gate_up[%d,4096]x[4096,14336]  vectorized SiLU(silu(g)*u, packed gate_up)"
           "  (mma_warps=%d nacc=%d iters=%d, WARPS=%d, STREAMW=%s)\n",
           INPUT_LEN, INPUT_LEN, MMAW, NACC, ITERS, WPB, dWn?"on":"off");
    printf("  GEMM(tcgen05)-only        : %7.1f us\n", t_gemm);
    printf("  SiLU-only (vLLM, M blocks): %7.1f us   <- fair baseline (optimal launch)\n", t_silu);
    printf("  serial (GEMM + vLLM SiLU) : %7.1f us\n", serial);
    if(iso_ok)
        printf("  2-stream (green-ctx iso, GEMM=%d||SiLU=%d): %7.1f us  speedup %.2fx   <- NanoFlow SM-isolation\n",
               gg, gs, t_2stream, serial / t_2stream);
    else
        printf("  2-stream (green-ctx iso)  :    (green contexts unavailable here)\n");
    printf("  FUSED warp-spec (%dmma+%dsilu): %7.1f us  speedup %.2fx  (FUSED/GEMM %.3f)   <- within-SM\n",
           MMAW, WPB - MMAW - (dWn?STREAMWARPS:0), t_fused, serial / t_fused, t_fused / t_gemm);
    return 0;
}
