// up-silu-overlap-l40.cu — Ampere/Ada (L40, sm_89) port of up-silu-overlap.cu.
// Same question: can a SwiGLU SiLU (CUDA-core/SFU-bound) overlap with the up-projection GEMM
// (Tensor-Core-bound) on the SAME SMs?  Two mechanisms are compared, exactly as on B200:
//   (A) WITHIN-SM warp specialization (one fused kernel):
//         warps 0..mma_warps-1     : issue mma.sync.m16n8k16 -> per-warp register accumulators (Tensor Core)
//         warps mma_warps..WPB-1   : SiLU(gate)*up                                  (CUDA cores/SFU)
//         ...all in ONE CTA -> same SM.  <- co-residency is GUARANTEED.
//   (B) TWO-STREAM via SM ISOLATION (NanoFlow/TokenWeave style): green contexts hard-partition the
//         SMs into a GEMM pool (env SMG, default 96) + a SiLU pool (the remainder); each kernel runs
//         on its own green-ctx stream confined to its SM slice (driver API).
//
// ========================= WHAT CHANGED vs the B200 (sm_100) version =========================
// L40 is Ada Lovelace (sm_89): NO tcgen05, NO TMEM, NO warpgroup wgmma. Ada/Ampere Tensor Cores use
// the *synchronous, warp-level* `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instruction,
// which accumulates into REGISTERS (no Tensor Memory). So the GEMM core is rewritten:
//   - tcgen05.mma (128x128x16 async, TMEM dest)  ->  mma.sync (16x8x16 sync, register dest).
//   - per-warp TMEM accumulators                 ->  per-warp REGISTER accumulators (nacc of them).
//   - TMEM::Allocator / mbarrier / tcgen05.commit/wait  ->  all GONE (nothing to allocate or await).
//   - CuTe SM100 UMMA atoms + SMEM A/B tiles + descriptors  ->  GONE; operands are zeroed registers
//     (this is a pure Tensor-Core THROUGHPUT PoC, just like the B200 version zeroed A/B).
// The SiLU path and the green-context SM-isolation comparison are byte-for-byte the same.
// ILP: each MMA warp owns NACC (=4) independent register accumulators and issues to them
//   round-robin to hide mma.sync's fixed latency and keep the Tensor Core fed.
//
// Input length    : arg1 = M (tokens), default 2048. Maps to Llama-3.1-8B TP2 per-rank:
//   gate_up [M,4096]x[4096,14336], SiLU on packed gate_up[M, 2*7168] -> out[M,7168].
//   Total Tensor-Core work is FLOP-faithful: M*HIDDEN*(2*INTER_SHARD) MACs, tiled by the 16x8x16
//   mma.sync (=2048 MACs/inst), split across GRID x mma_warps x nacc -> `iters` issues/accumulator.
//
// build (native L40 + PTX fallback so it also JIT-runs on newer GPUs incl. B200):
//   nvcc -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89 \
//        -O3 -std=c++17 up-silu-overlap-l40.cu -o up_silu_l40 -lcuda
//   (mma.sync.m16n8k16 is plain sm_80+; swap arch=compute_80/86 for A100 / A40 -- genuine Ampere.)
// run  : ./up_silu_l40 <input_len_M=2048> <mma_warps=4>
//        (env: WARPS=warps/CTA[16], GRID=#blocks[=device SM count], SMG=GEMM-pool SMs for green-ctx[96])
//        SILU_SCALE=10 is recommended for meaningful overlap: default SiLU (M=2048, ~45 us) is too light
//        for either FUSED or 2-stream to show speedup. At 10x (M=20480, ~1260 us) SiLU is balanced with
//        GEMM (~1340 us) and FUSED warp-spec achieves ~1.35x speedup over serial.
// NOTE: there is no L40 in this dev box (8xB200); the compute_89 PTX above JIT-compiles to sm_100 at
//   load, so the same binary runs here too -- but the printed microseconds then reflect B200, not L40.
// ============================================================================================
#include <cuda_runtime.h>
#include <cuda.h>          // driver API: Green Contexts -> HARD SM isolation (NanoFlow/TokenWeave style)
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#define WARP 32
static int WPB=16;         // warps per CTA (host launch); kernel reads blockDim.x/WARP. env WARPS overrides.
                           // default 16 = 4 MMA + 12 SiLU. HARD CAP WPB<=32 (1024 threads/block on Ada).
constexpr int NACC=4;     // register accumulators per MMA warp (ILP); each = 4 f32 regs

constexpr int D=7168;      // INTER_SHARD; SiLU input is packed gate_up[M, 2*D], output[M, D]

// One Ada/Ampere Tensor-Core MMA: 16x8x16 bf16 inputs, f32 accumulate, in-place (D=A*B+C with D=C).
// A/B operands are zeroed (this is a Tensor-Core THROUGHPUT PoC -- the TC executes regardless of
// operand values, so this measures genuine mma.sync issue throughput). `volatile` => never DCE'd;
// c0..c3 are read-modify-write so the accumulator chain stays live across iterations.
__device__ __forceinline__ void mma_16816(float& c0,float& c1,float& c2,float& c3,
        uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0,uint32_t b1){
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

// Byte-for-byte vLLM csrc/activation_kernels.cu: act_and_mul_kernel<silu_kernel, act_first=true>.
//   silu_kernel(x) = x / (1 + expf(-x))  (precise expf, NOT __expf);  loads via VLLM_LDG = __ldg.
// `rep` (env SILU_REP, default 1) repeats the SiLU/SFU compute per element to STRESS the overlap in a
// balanced GEMM:SiLU regime (rep=1 is the real op; rep=10 ~10x's the CUDA-core/SFU work). The +r*1e-20f
// perturbation makes each pass a distinct expf input so the compiler can't CSE the loop to 1 pass.
__device__ __forceinline__ void silu_one(const __nv_bfloat16* gu, __nv_bfloat16* o, long t, int j, int rep){
    float g=__bfloat162float(__ldg(&gu[t*2L*D + j]));         // gate  (read-only cache, like vLLM)
    float u=__bfloat162float(__ldg(&gu[t*2L*D + D + j]));     // up
    float acc=0.f;
    #pragma unroll 1
    for(int r=0;r<rep;r++){ float gr=g+__int2float_rn(r)*1e-20f; acc += (gr/(1.0f+expf(-gr)))*u; }
    o[t*(long)D + j]=__float2bfloat16( acc * (1.0f/(float)rep) );   // rep=1 => exactly silu(g)*u
}
// one 8-channel (128-bit) SiLU group, registers already loaded (rep folds in the SILU_REP compute-stress)
__device__ __forceinline__ void silu8(const int4& gv, const int4& uv, int4& ov, int rep){
    const __nv_bfloat16* g16=(const __nv_bfloat16*)&gv; const __nv_bfloat16* u16=(const __nv_bfloat16*)&uv;
    __nv_bfloat16* o16=(__nv_bfloat16*)&ov;
    #pragma unroll
    for(int e=0;e<8;e++){ float g=__bfloat162float(g16[e]), u=__bfloat162float(u16[e]);
        float acc=0.f;
        #pragma unroll 1
        for(int r=0;r<rep;r++){ float gr=g+__int2float_rn(r)*1e-20f; acc+=(gr/(1.0f+expf(-gr)))*u; }
        o16[e]=__float2bfloat16( acc*(1.0f/(float)rep) ); }
}
// FAST in-fused SiLU (the Blackwell large-batch fix, ported to no-TMEM Ada/Ampere): 128-bit loads +
// PREF-deep prefetch -> high memory-level parallelism -> BANDWIDTH-bound even at low occupancy. Question
// this answers: can MLP overcome the register/occupancy conflict that exists here (no TMEM -> mma.sync
// accumulators sit in registers and compete with these prefetch buffers)?
__device__ __forceinline__ void silu_strided(const __nv_bfloat16* gu, __nv_bfloat16* o,
                                             int M, long tid, long nt, int rep){
    const int PREF=4;
    long groups=(long)M*(D/8);                            // 8 channels per group (16B aligned)
    for(long g0=tid; g0<groups; g0+=(long)PREF*nt){
        int4 gv[PREF], uv[PREF]; long gi[PREF];
        #pragma unroll
        for(int p=0;p<PREF;p++){ gi[p]=g0+(long)p*nt;
            if(gi[p]<groups){ long t=gi[p]/(D/8); int j=(int)((gi[p]-t*(D/8))*8);
                const __nv_bfloat16* b=gu+t*2L*D+j; gv[p]=__ldg((const int4*)b); uv[p]=__ldg((const int4*)(b+D)); } }
        #pragma unroll
        for(int p=0;p<PREF;p++){ if(gi[p]<groups){ long t=gi[p]/(D/8); int j=(int)((gi[p]-t*(D/8))*8);
            int4 ov; silu8(gv[p],uv[p],ov,rep); *((int4*)(o+t*(long)D+j))=ov; } }
    }
}
// vLLM-style optimal baseline, vectorized 128-bit (one block per token)
__global__ void silu_vllm(const __nv_bfloat16* gu, __nv_bfloat16* o, int M, int rep){
    const __nv_bfloat16* gb=gu+(long)blockIdx.x*2L*D; __nv_bfloat16* ob=o+(long)blockIdx.x*D;
    for(int j=threadIdx.x*8; j<D; j+=blockDim.x*8){
        int4 gv=__ldg((const int4*)(gb+j)), uv=__ldg((const int4*)(gb+D+j)), ov; silu8(gv,uv,ov,rep);
        *((int4*)(ob+j))=ov;
    }
}

// mode 0=GEMM-only(mma_warps issue),1=SiLU-only,2=FUSED
__global__ void ws(int mode,int mma_warps,int iters,const __nv_bfloat16* gu,__nv_bfloat16* o,int M,int rep){
    int wib=threadIdx.x/WARP;
    bool do_mma = (mode!=1 && wib<mma_warps);
    bool do_silu= (mode==1) || (mode==2 && wib>=mma_warps);

    if(do_mma){
        float c[NACC][4];                                    // per-warp register accumulators (ILP)
        #pragma unroll
        for(int a=0;a<NACC;a++){ c[a][0]=c[a][1]=c[a][2]=c[a][3]=0.f; }
        uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;              // zeroed bf16 operands (throughput PoC)
        for(int k=0;k<iters;k++){
            #pragma unroll
            for(int a=0;a<NACC;a++)                           // exactly NACC mmas per iteration, no branches
                mma_16816(c[a][0],c[a][1],c[a][2],c[a][3], a0,a1,a2,a3,b0,b1);
        }
        // DCE guard: mode is a runtime arg, so the compiler can't prove this branch dead and must keep
        // the accumulators (and thus every mma.sync) live. The store never actually executes.
        float s=0.f;
        #pragma unroll
        for(int a=0;a<NACC;a++) s+=c[a][0]+c[a][1]+c[a][2]+c[a][3];
        if(mode==0x7fffffff && threadIdx.x==0) o[blockIdx.x]=__float2bfloat16(s);
    } else if(do_silu){
        int wpb        = blockDim.x / WARP;                  // runtime warps/CTA
        int silu_warps = (mode==1) ? wpb : (wpb - mma_warps);
        int my_warp    = (mode==1) ? wib : (wib - mma_warps);
        int lane       = threadIdx.x % WARP;
        long thread_id  = ((long)blockIdx.x * silu_warps + my_warp) * WARP + lane;
        long thread_cnt = (long)gridDim.x * silu_warps * WARP;
        silu_strided(gu, o, M, thread_id, thread_cnt, rep);
    }
}

// ---- device buffers + run config (set in main, read by the launchers below) ----
static __nv_bfloat16 *dGU, *dO;     // packed gate_up[M,2D] and out[M,D]
static int MMAW, ITERS;             // MMA warps, mma.sync issues/accumulator
static int Msilu;                   // SiLU token count = SILU_SCALE*M (scales SiLU mem+compute, GEMM fixed)
static int GRID;                    // #blocks launched (default = device SM count, 1 per SM)
static int SILU_REP = 1;            // SiLU COMPUTE repeat factor (env SILU_REP): scales SFU work only (affine)
static int SILU_SCALE = 1;          // SiLU TENSOR scale (env SILU_SCALE): SILU_SCALE x tokens -> mem+compute (linear)

void rg() { ws       <<<GRID, WPB*WARP>>>(0, MMAW, ITERS, dGU, dO, Msilu, SILU_REP); } // GEMM-only
void rs() { silu_vllm<<<Msilu, 1024>>>  (dGU, dO, Msilu, SILU_REP); }                        // vLLM SiLU (baseline)
void rf() { ws       <<<GRID, WPB*WARP>>>(2, MMAW, ITERS, dGU, dO, Msilu, SILU_REP); } // FUSED within-SM
// Time `it` launches of `f` (after `wu` warm-up launches); returns microseconds per launch.
float tim(void(*f)(), int it=50, int wu=15){
    for(int i=0;i<wu;i++) f();
    cudaError_t le = cudaGetLastError();           // catch SYNCHRONOUS launch-config errors (e.g.
    if(le != cudaSuccess){                          // too many regs at WARPS=32) -- else sync sees no
        printf("  [launch err: %s]\n", cudaGetErrorString(le));  // pending work and timing reads ~0us
        return -1;                                  // (a failed launch must report, not fabricate a speedup)
    }
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
#define CUCK(x) do{ CUresult _r=(x); if(_r!=CUDA_SUCCESS){ const char* _s; cuGetErrorString(_r,&_s); \
    printf("    [cu err %s: %s]\n", #x, _s); return false; } }while(0)

// Time the isolated 2-stream: GEMM on `gpool`, SiLU on `spool` (each a green-context stream).
float tim_2stream_iso(cudaStream_t gpool, cudaStream_t spool, int it=50, int wu=15){
    auto launch_both = [&](){
        ws       <<<GRID, WPB*WARP, 0, gpool>>>(0, MMAW, ITERS, dGU, dO, Msilu, SILU_REP);  // GEMM on its SMs
        silu_vllm<<<Msilu, 1024,    0, spool>>>(dGU, dO, Msilu, SILU_REP); // SiLU on its SMs
    };
    for(int i=0;i<wu;i++) launch_both();
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<it;i++) launch_both();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float,std::micro>(t1 - t0).count() / it;
}

// Build a green-context SM split: GEMM pool (sm_g SMs) + SiLU pool (the remainder). Hands back the two
// green-ctx streams + contexts (caller destroys both) and the ACTUAL SM counts (driver rounds to the
// co-scheduled alignment). false if green contexts are unavailable here.
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
    MMAW          = c>2?atoi(v[2]):4;        // MMA-issuing warps

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
    GRID = smCount > 0 ? smCount : 142;      // default: one block per SM (L40=142, B200=148, ...)
    if(getenv("GRID"))  GRID = atoi(getenv("GRID"));
    if(getenv("WARPS")) WPB  = atoi(getenv("WARPS"));
    if(MMAW > WPB){ printf("mma_warps=%d exceeds WPB=%d\n", MMAW, WPB); return 1; }
    int sm_g = getenv("SMG") ? atoi(getenv("SMG")) : 96;     // GEMM-pool SMs for the green-ctx 2-stream
    if(getenv("SILU_REP")) SILU_REP = atoi(getenv("SILU_REP"));   // repeat SiLU compute (SFU work, affine)
    if(SILU_REP < 1) SILU_REP = 1;
    if(getenv("SILU_SCALE")) SILU_SCALE = atoi(getenv("SILU_SCALE"));  // scale SiLU tokens (mem+compute, linear)
    if(SILU_SCALE < 1) SILU_SCALE = 1;

    const long HIDDEN=4096, INTER_SHARD=7168, GU=2*INTER_SHARD;
    Msilu = SILU_SCALE * INPUT_LEN;                                   // SiLU token count (GEMM untouched)
    long total_mma = (long)INPUT_LEN * HIDDEN * GU / (16L*8*16);      // real gate_up MACs / 16x8x16 tile
    ITERS = (int)(total_mma / ((long)GRID * MMAW * NACC));            // mma.sync issues per accumulator (NACC is constexpr)
    if(ITERS < 1) ITERS = 1;
    cudaMalloc(&dGU,(long)Msilu*GU*2);                               // packed gate_up[Msilu, 2*7168] bf16
    cudaMalloc(&dO ,(long)Msilu*INTER_SHARD*2);                      // out[Msilu, 7168] bf16
    cudaMemset(dGU,0,(long)Msilu*GU*2);

    // LOOP=<n> runs <n> iterations of one mode back-to-back (for nsys profiling), then exits.
    // LOOPMODE: 0=GEMM-only, 1=vLLM SiLU, 2=FUSED, 3=green-ctx isolated 2-stream (env SMG).
    if(getenv("LOOP")){
        int n = atoi(getenv("LOOP"));
        int m = getenv("LOOPMODE") ? atoi(getenv("LOOPMODE")) : 0;
        CUstream sG=0, sS=0; CUgreenCtx gG=0, gS=0; int gg=0, gs=0;
        if(m==3){
            cuInit(0);
            if(!make_green_split(sm_g, &sG, &sS, &gG, &gS, &gg, &gs)){
                printf("(green contexts unavailable here)\n"); return 1; }
        }
        for(int i=0;i<n;i++){
            if      (m==0) rg();
            else if (m==1) rs();
            else if (m==2) rf();
            else {
                ws       <<<GRID, WPB*WARP, 0, (cudaStream_t)sG>>>(0, MMAW, ITERS, dGU, dO, Msilu, SILU_REP);
                silu_vllm<<<Msilu, 1024,    0, (cudaStream_t)sS>>>(dGU, dO, Msilu, SILU_REP);
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

    cuInit(0);
    float t_2stream; int gg = 0, gs = 0;
    bool iso_ok = bench_iso(sm_g, &t_2stream, &gg, &gs);

    printf("M=%d  gate_up[%d,4096]x[4096,14336]  SiLU on [%d,7168] (vLLM-exact x/(1+e^-x)*u)"
           "  (mma.sync m16n8k16, mma_warps=%d nacc=%d iters=%d, GRID=%d, silu_scale=%d rep=%d)\n",
           INPUT_LEN, INPUT_LEN, Msilu, MMAW, (int)NACC, ITERS, GRID, SILU_SCALE, SILU_REP);
    printf("  GEMM(mma.sync)-only       : %7.1f us\n", t_gemm);
    printf("  SiLU-only (vLLM, Msilu blk): %7.1f us   <- fair baseline (optimal launch)\n", t_silu);
    printf("  serial (GEMM + vLLM SiLU) : %7.1f us\n", serial);
    if(iso_ok)
        printf("  2-stream (green-ctx iso, GEMM=%d||SiLU=%d): %7.1f us  speedup %.2fx   <- NanoFlow SM-isolation\n",
               gg, gs, t_2stream, serial / t_2stream);
    else
    
        printf("  2-stream (green-ctx iso)  :    (green contexts unavailable here)\n");
    printf("  FUSED warp-spec (%dmma+%dsilu): %7.1f us  speedup %.2fx   <- within-SM\n",
           MMAW, WPB - MMAW, t_fused, serial / t_fused);
    return 0;
}
