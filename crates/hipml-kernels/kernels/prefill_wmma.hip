// fp16 prefill GEMM on the matrix unit (gfx12 WMMA), over Q4_K / Q6_K / Q8_0.
//
// Same contract as `prefill_f16` — `y[t, :] = W @ x[t, :]`, weights dequantised
// once per tile into shared, f32 out — but the inner product is
// `wmma_f32_16x16x16_f16` instead of `v_dot2_f32_f16`. The VALU dot2 path
// measures 41.4 TFLOPS on this card (register-only probe); the matrix unit
// measures ~185 TFLOPS for this instruction under the same probe, so the dot2
// ceiling is what this kernel exists to get past.
//
// Tile: PWW_TT prompt tokens x PWW_RT output rows per block, 8 warps. Warp w
// owns PWW_WT tokens and PWW_WR rows, i.e. PWW_WR/16 accumulators of 16x16.
// Per k16 step a warp loads PWW_WT/16 + PWW_WR/16 fragments (one LDS.128 per
// lane each) from a row-major half tile and issues its accumulator grid.
//
// Fragment layout, measured on gfx1201 (wave32) with a one-warp probe that
// reads the accumulator back for A*I, for A = row indices and for B = column
// indices:
//   operands A/B: lane l, element e -> index l % 16, k = 8*(l / 16) + e
//   accumulator:  lane l, element e -> col l % 16, row 8*(l / 16) + e
// so both operands load as one LDS.128 per lane from a row-major tile, and the
// epilogue writes f32 straight to `y` with no shared transpose: for one
// (accumulator, element) pair the 32 lanes cover 2 tokens x 16 consecutive
// rows, four fully-covered 32 B sectors.
//
// The dequant is the same 8-value reader the dot2 kernel uses
// (`prefill_common.inc`), four threads per weight row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "prefill_common.inc"

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "prefill_wmma: the gfx12 WMMA intrinsics are wave32 only"
#endif

// Geometry. The defaults are the measured best; every one is overridable so a
// shape sweep is a rebuild with -D and no source edit.
#ifndef PWW_TT
#define PWW_TT 256         // prompt tokens per block
#endif
#ifndef PWW_RT
#define PWW_RT 128         // output rows per block
#endif
#ifndef PWW_WT
#define PWW_WT 32          // tokens per warp
#endif
#ifndef PWW_WR
#define PWW_WR 64          // rows per warp
#endif
#define PWW_KW 32          // k values per shared k-tile
#define PWW_XS 40          // x row stride in halfs: 32 values + 8 pad (16 B aligned)
#define PWW_WS 40          // weight row stride, same
#ifndef PWW_THREADS
#define PWW_THREADS 512
#endif
#define PWW_WARPS (PWW_THREADS / 32)
#define PWW_TG (PWW_TT / PWW_WT)      // token groups across the block
#define PWW_RG (PWW_RT / PWW_WR)      // row groups across the block
#define PWW_NTG (PWW_WT / 16)         // 16-token tiles per warp
#define PWW_NRG (PWW_WR / 16)         // 16-row tiles per warp
#define PWW_ACC (PWW_NTG * PWW_NRG)   // accumulators per warp
#define PWW_DPROW (PWW_THREADS / 4)   // weight rows dequantised per pass
#define PWW_XROW (PWW_THREADS / 8)    // token rows staged per pass

static_assert(PWW_TG * PWW_RG == PWW_WARPS, "warp grid must cover the block");
static_assert(PWW_TT % PWW_WT == 0 && PWW_RT % PWW_WR == 0, "ragged warp tiles");
static_assert(PWW_WT % 16 == 0 && PWW_WR % 16 == 0, "warp tile must be a 16 tile");
static_assert(PWW_TT % PWW_XROW == 0 && PWW_RT % PWW_DPROW == 0, "ragged staging");

typedef _Float16 __attribute__((ext_vector_type(8))) pww_h8;
typedef float __attribute__((ext_vector_type(8))) pww_f8;

template <int KIND>
static __device__ __forceinline__ void pww_body(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) _Float16 sx[PWW_TT * PWW_XS];
    __shared__ __align__(16) _Float16 sw[PWW_RT * PWW_WS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wtg = warp / PWW_RG;   // which token group
    const int wrg = warp % PWW_RG;   // which row group

    const int tok_base = blockIdx.y * PWW_TT;
    const int row_base = blockIdx.x * PWW_RT;
    const size_t wrow = pf_row_bytes(k, KIND);

    // x tile: 8 threads per token row, 4 floats each.
    const int xt = tid >> 3;
    const int xc = (tid & 7) * 4;
    __half2* xd = reinterpret_cast<__half2*>(sx + xt * PWW_XS + xc);

    // Weight tile: 4 threads per row, 8 values each.
    const int d_row = tid >> 2;
    const int d_q = tid & 3;

    // Fragment offsets, fixed for the whole k loop.
    const int frow = lane & 15;          // row of the tile this lane feeds
    const int fcol = (lane >> 4) * 8;    // k offset of this lane's 8 values
    const int xoff = (wtg * PWW_WT + frow) * PWW_XS + fcol;
    const int boff = (wrg * PWW_WR + frow) * PWW_WS + fcol;

    pww_f8 acc[PWW_ACC];
#pragma unroll
    for (int j = 0; j < PWW_ACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;

    for (int base = 0; base < k; base += PWW_KW) {
        __syncthreads(); // the previous multiply has finished with the tiles

#pragma unroll
        for (int p = 0; p < PWW_TT / PWW_XROW; ++p) {
            const int rl = p * PWW_XROW + xt;
            const int gt = tok_base + rl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens)
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + xc);
            __half2* dst = reinterpret_cast<__half2*>(sx + rl * PWW_XS + xc);
            dst[0] = __floats2half2_rn(v.x, v.y);
            dst[1] = __floats2half2_rn(v.z, v.w);
        }

#pragma unroll
        for (int p = 0; p < PWW_RT / PWW_DPROW; ++p) {
            const int rl = p * PWW_DPROW + d_row;
            const int gr = row_base + rl;
            pww_h8 hv;
            if (gr < m) {
                if (KIND == 3) {
                    const _Float16* hp = reinterpret_cast<const _Float16*>(w) +
                                         (size_t)gr * k + base + d_q * 8;
                    hv = *reinterpret_cast<const pww_h8*>(hp);
                } else {
                    const unsigned char* rp = w + (size_t)gr * wrow;
                    float v[8];
                    pf_dequant8<KIND>(rp, base + d_q * 8, v);
#pragma unroll
                    for (int i = 0; i < 8; ++i) hv[i] = (_Float16)v[i];
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) hv[i] = (_Float16)0.0f;
            }
            *reinterpret_cast<pww_h8*>(sw + rl * PWW_WS + d_q * 8) = hv;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < PWW_KW / 16; ++kk) {
            pww_h8 a[PWW_NTG];
            pww_h8 b[PWW_NRG];
#pragma unroll
            for (int i = 0; i < PWW_NTG; ++i)
                a[i] = *reinterpret_cast<const pww_h8*>(sx + xoff + i * 16 * PWW_XS + kk * 16);
#pragma unroll
            for (int j = 0; j < PWW_NRG; ++j)
                b[j] = *reinterpret_cast<const pww_h8*>(sw + boff + j * 16 * PWW_WS + kk * 16);
#pragma unroll
            for (int i = 0; i < PWW_NTG; ++i)
#pragma unroll
                for (int j = 0; j < PWW_NRG; ++j) {
                    const int idx = i * PWW_ACC + j;
                    acc[idx] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a[i], b[j], acc[idx]);
                }
        }
    }

    // Accumulator -> row-major f32 output, straight to global: this lane holds
    // column (lane & 15) of rows 8*(lane >> 4) + e.
    const int mrow = 8 * (lane >> 4);
    const int gr = row_base + wrg * PWW_WR + (lane & 15);
#pragma unroll
    for (int i = 0; i < PWW_NTG; ++i)
#pragma unroll
        for (int j = 0; j < PWW_NRG; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int gt = tok_base + wtg * PWW_WT + i * 16 + mrow + e;
                const int r = gr + j * 16;
                if (gt < tokens && r < m)
                    y[(size_t)gt * m + r] = acc[i * PWW_NRG + j][e];
            }
        }
}

extern "C" __global__ void prefill_q4_k_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pww_body<0>(w, x, y, m, k, tokens);
}

extern "C" __global__ void prefill_q6_k_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pww_body<1>(w, x, y, m, k, tokens);
}

extern "C" __global__ void prefill_q8_0_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pww_body<2>(w, x, y, m, k, tokens);
}

// Weights already dequantised to half (Q5_K, done once at load).
extern "C" __global__ void prefill_halfw_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pww_body<3>(w, x, y, m, k, tokens);
}
