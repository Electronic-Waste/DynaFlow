// up-silu-overlap-h100.cu — Hopper (H100, sm_90a) port of up-silu-overlap.cu / -l40.cu.
// Same question: can a SwiGLU SiLU (CUDA-core/SFU-bound) overlap with the up-projection GEMM
// (Tensor-Core-bound) on the SAME SMs?  This version exists to test the hypothesis from the L40
// analysis: the big SiLU∥GEMM "conflict" on Ada (sm_89) is caused by `mma.sync` being SYNCHRONOUS,
// warp-level -- the tensor core only runs while the warp keeps issuing, so co-resident SiLU warps
// steal the SMSP issue slots and STARVE the tensor core (tensor pipe collapsed 95%->55%).
//
// ========================= WHAT CHANGED vs Ada (sm_89) =========================
// Hopper has `wgmma.mma_async` — the ASYNCHRONOUS, WARPGROUP-level MMA (a warpgroup = 4 warps = 128
// threads). A warpgroup issues `wgmma.mma_async` and the tensor core runs it in the BACKGROUND while
// the warpgroup does `commit_group`/`wait_group`. So — unlike Ada's `mma.sync` — the MMA does NOT need
// continuous warp issue to stay busy. Prediction: co-resident SiLU warps fill the (genuinely free)
// issue slots WITHOUT starving the tensor core, i.e. the overlap should be clean, like Blackwell.
//   - mma.sync (16x8x16 sync, register dest)  ->  wgmma.mma_async (64xNx16 async, register dest).
//   - per-warp register accumulators          ->  per-WARPGROUP register accumulators (float[32] for N=64).
//   - accumulators STILL live in registers (Hopper has NO TMEM) -> the register-pressure "enabler"
//     remains (see -Xptxas -v), but should be benign because the async MMA tolerates low occupancy.
//   - operands are zeroed/uninitialized SMEM tiles + canonical GMMA descriptors (throughput PoC, exactly
//     like the sm_100 version zeroed its A/B).  The SiLU path + green-ctx comparison are byte-for-byte the same.
//
// build (H100): nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -I<cutlass_inc> \
//                    up-silu-overlap-h100.cu -o up_silu_h100 -lcuda
// run  : ./up_silu_h100 <M=2048> <mma_warps=4 (multiple of 4)>
//        env: WARPS=warps/CTA[16], GRID=#blocks[=SM count], SMG, SILU_SCALE, SILU_REP,
//             LOOP=<n> LOOPMODE=<0 GEMM|1 SiLU|2 FUSED|3 2-stream>.
// NOTE: `wgmma` is Hopper-only (sm_90a). Blackwell (sm_100) dropped it for tcgen05, so this binary
//   will NOT JIT-run on the B200 dev box — it needs a real H100. (Register usage from -Xptxas -v IS
//   valid though, as it is set at compile time.)
// ==============================================================================
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
using namespace cute;
#define WARP 32
static int WPB=16;         // warps per CTA (host launch); kernel reads blockDim.x/WARP. env WARPS overrides.
                           // default 16 = 4 MMA (1 warpgroup) + 12 SiLU. HARD CAP WPB<=32 on Hopper.
constexpr int GN=64;       // wgmma N tile: m64n64k16 -> CRegisters = float[32] per thread (64*64/128).
constexpr int NACC=2;      // independent accumulator groups per warpgroup (ILP; 2 -> 64 f32/thread).
constexpr int MACS_PER_WGMMA = 64*GN*16;

constexpr int D=7168;      // INTER_SHARD; SiLU input is packed gate_up[M, 2*D], output[M, D]

// ---- SiLU path: byte-for-byte identical to up-silu-overlap-l40.cu ----
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
__device__ __forceinline__ void silu_strided(const __nv_bfloat16* gu, __nv_bfloat16* o,
                                             int M, long tid, long nt, int rep){
    const int PREF=4;
    long groups=(long)M*(D/8);
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
__global__ void silu_vllm(const __nv_bfloat16* gu, __nv_bfloat16* o, int M, int rep){
    const __nv_bfloat16* gb=gu+(long)blockIdx.x*2L*D; __nv_bfloat16* ob=o+(long)blockIdx.x*D;
    for(int j=threadIdx.x*8; j<D; j+=blockDim.x*8){
        int4 gv=__ldg((const int4*)(gb+j)), uv=__ldg((const int4*)(gb+D+j)), ov; silu8(gv,uv,ov,rep);
        *((int4*)(ob+j))=ov;
    }
}

// one m64n64k16 wgmma into a float[32] accumulator group (accumulate: C += A*B, scale_D=One)
#define WGMMA_N64(DA,DB,C) SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>::fma( \
    (DA),(DB), \
    (C)[0],(C)[1],(C)[2],(C)[3],(C)[4],(C)[5],(C)[6],(C)[7],(C)[8],(C)[9],(C)[10],(C)[11],(C)[12],(C)[13],(C)[14],(C)[15], \
    (C)[16],(C)[17],(C)[18],(C)[19],(C)[20],(C)[21],(C)[22],(C)[23],(C)[24],(C)[25],(C)[26],(C)[27],(C)[28],(C)[29],(C)[30],(C)[31], \
    GMMA::ScaleOut::One)

// mode 0=GEMM-only(mma warpgroups issue wgmma),1=SiLU-only,2=FUSED
__global__ void ws(int mode,int mma_warps,int iters,const __nv_bfloat16* gu,__nv_bfloat16* o,int M,int rep){
    int wib=threadIdx.x/WARP;
    bool do_mma = (mode!=1 && wib<mma_warps);            // first mma_warps warps = mma_warps/4 warpgroups
    bool do_silu= (mode==1) || (mode==2 && wib>=mma_warps);

    // zeroed SMEM operand tiles (throughput PoC): A[64,16], B[GN,16], K-major. Shared by all warpgroups.
    __shared__ __align__(128) bfloat16_t sA[64*16];
    __shared__ __align__(128) bfloat16_t sB[GN*16];

    if(do_mma){
        // canonical no-swizzle (INTERLEAVE) K-major GMMA layouts -> descriptors accepted by wgmma.
        auto layA = tile_to_shape(SM90::GMMA::Layout_K_INTER_Atom<bfloat16_t>{}, Shape<Int<64>,Int<16>>{});
        auto layB = tile_to_shape(SM90::GMMA::Layout_K_INTER_Atom<bfloat16_t>{}, Shape<Int<GN>,Int<16>>{});
        Tensor tA = make_tensor(make_smem_ptr(sA), layA);
        Tensor tB = make_tensor(make_smem_ptr(sB), layB);
        uint64_t da = SM90::GMMA::make_gmma_desc<GMMA::Major::K>(tA);   // GmmaDescriptor -> uint64_t (operator)
        uint64_t db = SM90::GMMA::make_gmma_desc<GMMA::Major::K>(tB);

        float c[NACC][32];
        #pragma unroll
        for(int a=0;a<NACC;a++)
            #pragma unroll
            for(int i=0;i<32;i++) c[a][i]=0.f;

        float* cf=&c[0][0];
        #pragma unroll
        for(int i=0;i<NACC*32;i++) warpgroup_fence_operand(reinterpret_cast<uint32_t&>(cf[i]));

        for(int k=0;k<iters;k++){
            warpgroup_arrive();
            #pragma unroll
            for(int a=0;a<NACC;a++) WGMMA_N64(da,db,c[a]);   // NACC async wgmmas -> tensor core, back-to-back
            warpgroup_commit_batch();                        // fire; warp is now free (async!)
            warpgroup_wait<0>();                             // drain this batch before reusing accumulators
        }

        #pragma unroll
        for(int i=0;i<NACC*32;i++) warpgroup_fence_operand(reinterpret_cast<uint32_t&>(cf[i]));
        // DCE guard: keep every wgmma live. Store never actually executes (mode is a runtime arg).
        float s=0.f;
        #pragma unroll
        for(int a=0;a<NACC;a++)
            #pragma unroll
            for(int i=0;i<32;i++) s+=c[a][i];
        if(mode==0x7fffffff && threadIdx.x==0) o[blockIdx.x]=__float2bfloat16(s);
    } else if(do_silu){
        int wpb        = blockDim.x / WARP;
        int silu_warps = (mode==1) ? wpb : (wpb - mma_warps);
        int my_warp    = (mode==1) ? wib : (wib - mma_warps);
        int lane       = threadIdx.x % WARP;
        long thread_id  = ((long)blockIdx.x * silu_warps + my_warp) * WARP + lane;
        long thread_cnt = (long)gridDim.x * silu_warps * WARP;
        silu_strided(gu, o, M, thread_id, thread_cnt, rep);
    }
}

// ---- device buffers + run config (set in main, read by the launchers below) ----
static __nv_bfloat16 *dGU, *dO;
static int MMAW, ITERS;             // MMA warps (multiple of 4), wgmma batches/accumulator
static int Msilu;
static int GRID;
static int SILU_REP = 1;
static int SILU_SCALE = 1;

void rg() { ws       <<<GRID, WPB*WARP>>>(0, MMAW, ITERS, dGU, dO, Msilu, SILU_REP); } // GEMM-only
void rs() { silu_vllm<<<Msilu, 1024>>>  (dGU, dO, Msilu, SILU_REP); }                  // vLLM SiLU (baseline)
void rf() { ws       <<<GRID, WPB*WARP>>>(2, MMAW, ITERS, dGU, dO, Msilu, SILU_REP); } // FUSED within-SM
float tim(void(*f)(), int it=50, int wu=15){
    for(int i=0;i<wu;i++) f();
    cudaError_t le = cudaGetLastError();
    if(le != cudaSuccess){ printf("  [launch err: %s]\n", cudaGetErrorString(le)); return -1; }
    if(cudaDeviceSynchronize() != cudaSuccess){
        printf("  [err: %s]\n", cudaGetErrorString(cudaGetLastError())); return -1; }
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0;i<it;i++) f();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop); return ms / it * 1000.0f;
}

// ---- NanoFlow / TokenWeave style: HARD SM isolation via CUDA Green Contexts ----
#define CUCK(x) do{ CUresult _r=(x); if(_r!=CUDA_SUCCESS){ const char* _s; cuGetErrorString(_r,&_s); \
    printf("    [cu err %s: %s]\n", #x, _s); return false; } }while(0)
float tim_2stream_iso(cudaStream_t gpool, cudaStream_t spool, int it=50, int wu=15){
    auto launch_both = [&](){
        ws       <<<GRID, WPB*WARP, 0, gpool>>>(0, MMAW, ITERS, dGU, dO, Msilu, SILU_REP);
        silu_vllm<<<Msilu, 1024,    0, spool>>>(dGU, dO, Msilu, SILU_REP);
    };
    for(int i=0;i<wu;i++) launch_both();
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<it;i++) launch_both();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float,std::micro>(t1 - t0).count() / it;
}
bool make_green_split(int sm_g, CUstream* sG, CUstream* sS, CUgreenCtx* gG, CUgreenCtx* gS,
                      int* got_g, int* got_s){
    CUdevice dev; CUdevResource whole, group, remainder; unsigned int nb = 1;
    CUCK(cuDeviceGet(&dev, 0));
    CUCK(cuDeviceGetDevResource(dev, &whole, CU_DEV_RESOURCE_TYPE_SM));
    CUCK(cuDevSmResourceSplitByCount(&group, &nb, &whole, &remainder, 0, (unsigned)sm_g));
    CUdevResourceDesc descG, descS;
    CUCK(cuDevResourceGenerateDesc(&descG, &group,     1));
    CUCK(cuDevResourceGenerateDesc(&descS, &remainder, 1));
    CUCK(cuGreenCtxCreate(gG, descG, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUCK(cuGreenCtxCreate(gS, descS, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUCK(cuGreenCtxStreamCreate(sG, *gG, CU_STREAM_NON_BLOCKING, 0));
    CUCK(cuGreenCtxStreamCreate(sS, *gS, CU_STREAM_NON_BLOCKING, 0));
    *got_g = group.sm.smCount; *got_s = remainder.sm.smCount; return true;
}
bool bench_iso(int sm_g, float* out_us, int* got_g, int* got_s){
    CUstream sG, sS; CUgreenCtx gG, gS;
    if(!make_green_split(sm_g, &sG, &sS, &gG, &gS, got_g, got_s)) return false;
    *out_us = tim_2stream_iso((cudaStream_t)sG, (cudaStream_t)sS);
    cuStreamDestroy(sG); cuStreamDestroy(sS); cuGreenCtxDestroy(gG); cuGreenCtxDestroy(gS); return true;
}

int main(int c,char**v){
    int INPUT_LEN = c>1?atoi(v[1]):2048;
    MMAW          = c>2?atoi(v[2]):4;
    if(MMAW % 4 != 0){ printf("mma_warps=%d must be a multiple of 4 (wgmma is warpgroup-level)\n", MMAW); return 1; }

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
    GRID = smCount > 0 ? smCount : 132;      // default: one block per SM (H100=132, ...)
    if(getenv("GRID"))  GRID = atoi(getenv("GRID"));
    if(getenv("WARPS")) WPB  = atoi(getenv("WARPS"));
    if(MMAW > WPB){ printf("mma_warps=%d exceeds WPB=%d\n", MMAW, WPB); return 1; }
    int sm_g = getenv("SMG") ? atoi(getenv("SMG")) : 96;
    if(getenv("SILU_REP"))   SILU_REP   = atoi(getenv("SILU_REP"));   if(SILU_REP < 1) SILU_REP = 1;
    if(getenv("SILU_SCALE")) SILU_SCALE = atoi(getenv("SILU_SCALE")); if(SILU_SCALE < 1) SILU_SCALE = 1;

    const long HIDDEN=4096, INTER_SHARD=7168, GU=2*INTER_SHARD;
    Msilu = SILU_SCALE * INPUT_LEN;
    int mma_wg = MMAW / 4;                                            // warpgroups (each does one wgmma)
    long total_mma = (long)INPUT_LEN * HIDDEN * GU / MACS_PER_WGMMA;  // # of m64n64k16 wgmmas needed
    ITERS = (int)(total_mma / ((long)GRID * mma_wg * NACC));          // wgmma batches per accumulator group
    if(ITERS < 1) ITERS = 1;
    cudaMalloc(&dGU,(long)Msilu*GU*2);
    cudaMalloc(&dO ,(long)Msilu*INTER_SHARD*2);
    cudaMemset(dGU,0,(long)Msilu*GU*2);

    if(getenv("LOOP")){
        int n = atoi(getenv("LOOP"));
        int m = getenv("LOOPMODE") ? atoi(getenv("LOOPMODE")) : 0;
        CUstream sG=0, sS=0; CUgreenCtx gG=0, gS=0; int gg=0, gs=0;
        if(m==3){ cuInit(0);
            if(!make_green_split(sm_g, &sG, &sS, &gG, &gS, &gg, &gs)){ printf("(green contexts unavailable)\n"); return 1; } }
        for(int i=0;i<n;i++){
            if      (m==0) rg();
            else if (m==1) rs();
            else if (m==2) rf();
            else { ws       <<<GRID, WPB*WARP, 0, (cudaStream_t)sG>>>(0, MMAW, ITERS, dGU, dO, Msilu, SILU_REP);
                   silu_vllm<<<Msilu, 1024,    0, (cudaStream_t)sS>>>(dGU, dO, Msilu, SILU_REP); }
        }
        cudaDeviceSynchronize();
        if(m==3){ cuStreamDestroy(sG); cuStreamDestroy(sS); cuGreenCtxDestroy(gG); cuGreenCtxDestroy(gS); }
        return 0;
    }

    float t_gemm    = tim(rg);
    float t_silu    = tim(rs);
    float t_fused   = tim(rf);
    float serial    = t_gemm + t_silu;

    cuInit(0);
    float t_2stream; int gg = 0, gs = 0;
    bool iso_ok = bench_iso(sm_g, &t_2stream, &gg, &gs);

    printf("M=%d  gate_up[%d,4096]x[4096,14336]  SiLU on [%d,7168] (vLLM-exact x/(1+e^-x)*u)"
           "  (wgmma m64n%dk16, mma_warps=%d (=%d wg) nacc=%d iters=%d, GRID=%d, silu_scale=%d rep=%d)\n",
           INPUT_LEN, INPUT_LEN, Msilu, GN, MMAW, mma_wg, (int)NACC, ITERS, GRID, SILU_SCALE, SILU_REP);
    printf("  GEMM(wgmma)-only          : %7.1f us\n", t_gemm);
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
