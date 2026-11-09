// Skinny quantized GEMM on the matrix unit: R <= 16 activation rows against a
// K x N quantized weight, `y[r, :] = W @ x[r, :]`.
//
// Why this kernel exists: the f32-from-LDS batch GEMV multiplies every column
// separately, so its cost grows with R (1.90x at 4 columns, 2.95x at 8 while
// it is still streaming the same 17 GB). Padding R to one 16-wide WMMA tile
// makes the columns nearly free: the weight stream is read once and the matrix
// unit does the column fan-out, so the step cost is the tile, not R.
//
// The layout fact this rests on: in `wmma_f32_16x16x16_f16` (wave32, gfx12) a
// lane's B fragment element e is weight row `lane % 16`, k = `8*(lane/16) + e`
// — eight consecutive k of one row, exactly one Q4_K sub-block row run, which
// is what `pf_dequant8` produces. So a lane dequantises its own 8 values into
// its B fragment, and the same 8 values serve all 16 token rows.
//
// Per block: PWS_RT output rows, staged as fp16 in shared one PWS_KW-wide k
// tile at a time; warp w owns output rows [16w, 16w+16) and one 16x16
// accumulator for the 16 (padded) token rows. Per k16 step a warp issues two
// LDS.128 (one A fragment, one B fragment) and one WMMA. Weight bytes read are
// independent of R; the only R-dependent work is the token-row staging and the
// epilogue.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "prefill_common.inc"

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "gemm_skinny: the gfx12 WMMA intrinsics are wave32 only"
#endif

// Shipped geometry. Every value is a template argument so the sweep entries
// below can be one build; the defaults are the measured best.
#ifndef PWS_RT
#define PWS_RT 64      // output rows per block (4 warp tiles)
#endif
#ifndef PWS_KW
#define PWS_KW 64      // k values per shared tile
#endif
#ifndef PWS_THREADS
#define PWS_THREADS 128
#endif

typedef _Float16 __attribute__((ext_vector_type(8))) pws_h8;
typedef float __attribute__((ext_vector_type(8))) pws_f8;

template <int KIND, int RT, int KW, int THR>
static __device__ __forceinline__ void pws_body(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int k,
    int r)
{
    constexpr int XS = KW + 8;              // fp16 row stride, 16 B aligned
    constexpr int WS = KW + 8;
    constexpr int NW = RT / 16;             // warp n-tiles (= compute warps)
    constexpr int XROW = KW / 4;            // threads per token row (float4)
    constexpr int XRP = THR / XROW;         // token rows staged per pass
    constexpr int DROW = THR / 4;           // weight rows staged per pass
    static_assert(RT % 16 == 0, "RT must be a multiple of the 16 tile");
    static_assert(KW % 32 == 0, "KW must be 32-value aligned (pf_dequant8)");
    static_assert(16 % XRP == 0, "token staging must cover 16 rows");

    __shared__ __align__(16) _Float16 shx[16 * XS];
    __shared__ __align__(16) _Float16 shw[RT * WS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row_base = blockIdx.x * RT;
    const size_t wrow = pf_row_bytes(k, KIND);

    // x staging: XROW threads per token row, four floats each.
    const int xt = tid / XROW;
    const int xc = (tid % XROW) * 4;
    // weight staging: four threads per row, eight values each.
    const int d_row = tid >> 2;
    const int d_q = tid & 3;

    auto stage = [&](int base) {
#pragma unroll
        for (int p = 0; p < 16 / XRP; ++p) {
            const int rl = p * XRP + xt;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (rl < r)
                v = *reinterpret_cast<const float4*>(x + (size_t)rl * k + base + xc);
            __half2* dst = reinterpret_cast<__half2*>(shx + rl * XS + xc);
            dst[0] = __floats2half2_rn(v.x, v.y);
            dst[1] = __floats2half2_rn(v.z, v.w);
        }
#pragma unroll
        for (int q = 0; q < KW / 32; ++q)
#pragma unroll
        for (int p = 0; p < RT; p += DROW) {
            const int rl = p + d_row;
            if (rl >= RT) continue;
            const int gr = row_base + rl;
            const int kc = q * 32 + d_q * 8;
            pws_h8 hv;
            if (gr < n) {
                float v[8];
                pf_dequant8<KIND>(w + (size_t)gr * wrow, base + kc, v);
#pragma unroll
                for (int i = 0; i < 8; ++i) hv[i] = (_Float16)v[i];
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) hv[i] = (_Float16)0.0f;
            }
            *reinterpret_cast<pws_h8*>(shw + rl * WS + kc) = hv;
        }
    };

    pws_f8 acc;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;

    // Fragment offsets: A row is the token row, B row is the output row.
    const int frow = lane & 15;
    const int fcol = (lane >> 4) * 8;
    const int xoff = frow * XS + fcol;
    const int boff = (warp * 16 + frow) * WS + fcol;

    for (int base = 0; base < k; base += KW) {
        __syncthreads(); // the previous multiply has finished with the tiles
        stage(base);
        __syncthreads();
        if (warp < NW) {
#pragma unroll
            for (int kk = 0; kk < KW; kk += 16) {
                const pws_h8 a =
                    *reinterpret_cast<const pws_h8*>(shx + xoff + kk);
                const pws_h8 b =
                    *reinterpret_cast<const pws_h8*>(shw + boff + kk);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
            }
        }
    }

    // Accumulator -> f32: this lane holds column (output row) `lane & 15` of
    // token rows 8*(lane>>4) + e.
    if (warp < NW) {
        const int gr = row_base + warp * 16 + (lane & 15);
        if (gr < n) {
            const int t0 = 8 * (lane >> 4);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int t = t0 + e;
                if (t < r) y[(size_t)t * n + gr] = acc[e];
            }
        }
    }
}

// The shipped entries: PWS_RT x PWS_KW, one per resident block format.
extern "C" __global__ void gemm_skinny_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int k,
    int r)
{
    pws_body<0, PWS_RT, PWS_KW, PWS_THREADS>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int k,
    int r)
{
    pws_body<1, PWS_RT, PWS_KW, PWS_THREADS>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int k,
    int r)
{
    pws_body<2, PWS_RT, PWS_KW, PWS_THREADS>(w, x, y, n, k, r);
}

// Geometry sweep on the Q4_K entry: same body, one build. Each is a manifest
// row (`RowTile { rows = RT, block = THR }`) and runs the same parity case, so
// a sweep cannot pick a geometry that no test covers. The winning geometry is
// baked into PWS_* and the rest are removed.
extern "C" __global__ void gemm_skinny_q4_kg1( // RT 16, KW 64
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 16, 64, 128>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_kg2( // RT 32, KW 64
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 32, 64, 128>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_kg3( // RT 32, KW 128
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 32, 128, 128>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_kg4( // RT 32, KW 256
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 32, 256, 128>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_kg5( // RT 64, KW 128
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 64, 128, 256>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_kg6( // RT 128, KW 64
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 128, 64, 256>(w, x, y, n, k, r);
}
