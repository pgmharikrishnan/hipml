// fp16 prefill GEMM on the matrix unit (gfx12 WMMA), over Q4_K / Q6_K / Q8_0.
//
// Same contract as `prefill_f16` — `y[t, :] = W @ x[t, :]`, weights dequantised
// once per tile into shared, f32 out — but the inner product is
// `wmma_f32_16x16x16_f16` instead of `v_dot2_f32_f16`. The VALU dot2 path
// measures 41.4 TFLOPS on this card (register-only probe); the quoted 95.7
// TFLOPS dense fp16 figure is the matrix unit's, and this kernel is what can
// reach it.
//
// Geometry: 32 prompt tokens x 256 output rows per block, 256 threads. Warp w
// owns 16 tokens (w >> 2) x 64 rows (w & 3), i.e. four 16x16 tiles of f32
// accumulators. Per k16 step a warp loads one A fragment (its 16 tokens x 16 k)
// and four B fragments (its 64 rows x 16 k) from a row-major half tile and
// issues four WMMA: 32 KiB of MMA work per 80 B of LDS per lane, 12.8 FLOP per
// shared byte against a 128 B/cycle LDS.
//
// Fragment layout (measured on gfx1201, wave32 — `wmma_probe`, the accumulator
// readout of A*I):
//   operand A/B: lane l, element e -> index (l >> 1), k = (l & 1) * 8 + e
//   accumulator: lane l, element e -> row 8*(l&1) + 4*((l>>4)&1) + (e>>1),
//                                      col ((l>>1)&7) + 8*(e&1)
// so both operands load as one LDS.128 per lane from a row-major tile, and the
// epilogue writes f32 straight to `y` with no shared transpose: for one
// (accumulator, element) pair the 32 lanes cover 4 tokens x 8 consecutive rows,
// which is four fully-covered 32 B sectors.
//
// The dequant is the same 8-value reader the dot2 kernel uses
// (`prefill_common.inc`), four threads per weight row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "prefill_common.inc"

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "prefill_wmma: the gfx12 WMMA intrinsics are wave32 only"
#endif

#define PWW_TT 32          // prompt tokens per block
#define PWW_RT 256         // output rows per block
#define PWW_KW 32          // k values per shared k-tile
#define PWW_XS 40          // x row stride in halfs: 32 values + 8 pad (16 B aligned)
#define PWW_WS 40          // weight row stride, same
#define PWW_THREADS 256
#define PWW_WT 16          // tokens per warp
#define PWW_WR 64          // rows per warp
#define PWW_ACC (PWW_WR / 16)
#define PWW_DPROW (PWW_THREADS / 4)   // weight rows dequantised per pass

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
    const int wtg = warp >> 2;   // which 16-token group
    const int wrg = warp & 3;    // which 64-row group

    const int tok_base = blockIdx.y * PWW_TT;
    const int row_base = blockIdx.x * PWW_RT;
    const size_t wrow = pf_row_bytes(k, KIND);

    // x tile: 8 threads per token row, 4 floats each, one pass for the block.
    const int xt = tid >> 3;
    const int xc = (tid & 7) * 4;
    const int gxt = tok_base + xt;
    __half2* xd = reinterpret_cast<__half2*>(sx + xt * PWW_XS + xc);

    // Weight tile: 4 threads per row, 8 values each, PWW_RT/64 passes.
    const int d_row = tid >> 2;
    const int d_q = tid & 3;

    // Fragment offsets, fixed for the whole k loop.
    const int arow = lane >> 1;
    const int acol = (lane & 1) * 8;
    const int xoff = (wtg * PWW_WT + arow) * PWW_XS + acol;
    const int boff = (wrg * PWW_WR + (lane >> 1)) * PWW_WS + acol;

    pww_f8 acc[PWW_ACC];
#pragma unroll
    for (int j = 0; j < PWW_ACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;

    for (int base = 0; base < k; base += PWW_KW) {
        __syncthreads(); // the previous multiply has finished with the tiles

        {
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gxt < tokens)
                v = *reinterpret_cast<const float4*>(x + (size_t)gxt * k + base + xc);
            xd[0] = __floats2half2_rn(v.x, v.y);
            xd[1] = __floats2half2_rn(v.z, v.w);
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
            const pww_h8 a = *reinterpret_cast<const pww_h8*>(sx + xoff + kk * 16);
#pragma unroll
            for (int j = 0; j < PWW_ACC; ++j) {
                const pww_h8 b = *reinterpret_cast<const pww_h8*>(
                    sw + boff + (j * 16) * PWW_WS + kk * 16);
                acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[j]);
            }
        }
    }

    // Accumulator -> row-major f32 output, straight to global.
    const int mrow = 8 * (lane & 1) + 4 * ((lane >> 4) & 1);
    const int ncol = (lane >> 1) & 7;
#pragma unroll
    for (int j = 0; j < PWW_ACC; ++j) {
        const int gr = row_base + wrg * PWW_WR + j * 16 + ncol;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int gt = tok_base + wtg * PWW_WT + mrow + (e >> 1);
            const int r = gr + 8 * (e & 1);
            if (gt < tokens && r < m)
                y[(size_t)gt * m + r] = acc[j][e];
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
