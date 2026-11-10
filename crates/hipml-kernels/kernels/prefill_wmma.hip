// SPDX-License-Identifier: Apache-2.0
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
#ifndef PWW_KW
#define PWW_KW 32          // k values per shared k-tile
#endif
#define PWW_XS (PWW_KW + 8)  // x row stride in halfs: KW values + 8 pad (16 B aligned)
#define PWW_WS (PWW_KW + 8)  // weight row stride, same
#define PWW_KQ (PWW_KW / 32) // 32-value staging passes per row
#ifndef PWW_THREADS
#define PWW_THREADS 512
#endif
// Probe-only switches, used by `scripts/probe_wmma_gemm.cpp` to split the
// loop's cost into staging (`NO_STAGE`), fragment LDS reads (`NO_FRAG`) and
// matrix work (`NO_COMPUTE`). All default 0: the shipped kernel is unaffected.
#ifndef PWW_PROBE_NO_STAGE
#define PWW_PROBE_NO_STAGE 0
#endif
#ifndef PWW_PROBE_NO_FRAG
#define PWW_PROBE_NO_FRAG 0
#endif
#ifndef PWW_PROBE_NO_COMPUTE
#define PWW_PROBE_NO_COMPUTE 0
#endif
// Double-buffered staging: one barrier per k-tile, tile k+1 loaded while tile k
// multiplies. `PWW_BUF` is the shared-buffer count.
#ifndef PWW_DB
#define PWW_DB 0
#endif
#define PWW_BUF (PWW_DB ? 2 : 1)
// Software prefetch across the k-tile boundary: 0 off, 1 touch the next tile's
// lines (no registers held), 2 hold the next x chunk in registers.
#ifndef PWW_PF
#define PWW_PF 0
#endif
#define PWW_XN (PWW_KQ * (PWW_TT / PWW_XROW))  // x float4 chunks per thread
#define PWW_RP ((PWW_RT + PWW_DPROW - 1) / PWW_DPROW)  // weight staging passes
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
static_assert(PWW_TT % PWW_XROW == 0 && (PWW_RT % PWW_DPROW == 0 || PWW_RT < PWW_DPROW),
              "ragged staging");

typedef _Float16 __attribute__((ext_vector_type(8))) pww_h8;
typedef float __attribute__((ext_vector_type(8))) pww_f8;

// XHALF: the activation tile is already fp16 in global memory, so staging is a
// 16 B copy instead of an f32 load + convert, and the x stream is half the
// bytes. The engine converts once per projection (pww_xhalf below).
template <int KIND, bool XHALF = false>
static __device__ __forceinline__ void pww_body(
    const unsigned char* __restrict__ w,
    const void* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) _Float16 shx[PWW_BUF * PWW_TT * PWW_XS];
    __shared__ __align__(16) _Float16 shw[PWW_BUF * PWW_RT * PWW_WS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wtg = warp / PWW_RG;   // which token group
    const int wrg = warp % PWW_RG;   // which row group

    const int tok_base = blockIdx.y * PWW_TT;
    const int row_base = blockIdx.x * PWW_RT;
    const size_t wrow = pf_row_bytes(k, KIND);

    // x tile: fp32 -> 8 threads per token row, 4 floats each; fp16 -> 4 threads
    // per row, one 16 B fragment each.
    const int xt = XHALF ? (tid >> 2) : (tid >> 3);
    const int xc = XHALF ? (tid & 3) * 8 : (tid & 7) * 4;
#define PWW_XRP ((PWW_THREADS / (XHALF ? 4 : 8)))  // token rows staged per pass

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

#if PWW_PROBE_NO_FRAG
    // Probe only: the fragments are read once, so the k loop measures the
    // matrix work without LDS read traffic.
    pww_h8 a0[PWW_NTG];
    pww_h8 b0[PWW_NRG];
#pragma unroll
    for (int i = 0; i < PWW_NTG; ++i)
        a0[i] = *reinterpret_cast<const pww_h8*>(shx + xoff + i * 16 * PWW_XS);
#pragma unroll
    for (int j = 0; j < PWW_NRG; ++j)
        b0[j] = *reinterpret_cast<const pww_h8*>(shw + boff + j * 16 * PWW_WS);
#endif

    // Stage one k-tile (x halfs + dequantised weights) into the given buffers.
    auto stage_x = [&](_Float16* xs, int base) {
        if (XHALF) {
        const _Float16* xh = reinterpret_cast<const _Float16*>(x);
#pragma unroll
        for (int q = 0; q < PWW_KQ; ++q)
#pragma unroll
        for (int p = 0; p < PWW_TT / PWW_XRP; ++p) {
            const int rl = p * PWW_XRP + xt;
            const int gt = tok_base + rl;
            const int kc = q * 32 + xc;
            pww_h8 v;
            if (gt < tokens) {
                v = *reinterpret_cast<const pww_h8*>(xh + (size_t)gt * k + base + kc);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) v[i] = (_Float16)0.0f;
            }
            *reinterpret_cast<pww_h8*>(xs + rl * PWW_XS + kc) = v;
        }
        } else {
#pragma unroll
        for (int q = 0; q < PWW_KQ; ++q)
#pragma unroll
        for (int p = 0; p < PWW_TT / PWW_XROW; ++p) {
            const int rl = p * PWW_XROW + xt;
            const int gt = tok_base + rl;
            const int kc = base + q * 32 + xc;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens)
                v = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(x) +
                                                     (size_t)gt * k + kc);
            __half2* dst = reinterpret_cast<__half2*>(xs + rl * PWW_XS + kc - base);
            dst[0] = __floats2half2_rn(v.x, v.y);
            dst[1] = __floats2half2_rn(v.z, v.w);
        }
        }
    };

    auto stage_w = [&](_Float16* ws, int base) {
#pragma unroll
        for (int q = 0; q < PWW_KQ; ++q)
#pragma unroll
        for (int p = 0; p < PWW_RP; ++p) {
            const int rl = p * PWW_DPROW + d_row;
            if (rl >= PWW_RT) continue;
            const int gr = row_base + rl;
            const int kc = q * 32 + d_q * 8;
            pww_h8 hv;
            if (gr < m) {
                if (KIND == 3) {
                    const _Float16* hp = reinterpret_cast<const _Float16*>(w) +
                                         (size_t)gr * k + base + kc;
                    hv = *reinterpret_cast<const pww_h8*>(hp);
                } else {
                    const unsigned char* rp = w + (size_t)gr * wrow;
                    float v[8];
                    pf_dequant8<KIND>(rp, base + kc, v);
                    for (int i = 0; i < 8; ++i) hv[i] = (_Float16)v[i];
                }
            } else {
                for (int i = 0; i < 8; ++i) hv[i] = (_Float16)0.0f;
            }
            *reinterpret_cast<pww_h8*>(ws + rl * PWW_WS + kc) = hv;
        }
    };

#if PWW_PF
    // Touch the next tile's weight lines so the staging loads hit L2, and (PF=2)
    // hold the next x chunk in registers across the multiply.
    auto touch = [&](int base) {
#pragma unroll
        for (int q = 0; q < PWW_KQ; ++q)
#pragma unroll
        for (int p = 0; p < PWW_RP; ++p) {
            const int rl = p * PWW_DPROW + d_row;
            if (rl >= PWW_RT) continue;
            const int gr = row_base + rl;
            if (gr >= m) continue;
            unsigned v;
            if (KIND == 3) {
                const _Float16* hp = reinterpret_cast<const _Float16*>(w) +
                                     (size_t)gr * k + base + q * 32 + d_q * 8;
                v = *reinterpret_cast<const unsigned*>(hp);
            } else {
                const int blk_bytes = KIND == 0 ? PF_Q4K_BLOCK_BYTES
                                    : KIND == 1 ? PF_Q6K_BLOCK_BYTES
                                                : PF_Q8_0_BLOCK_BYTES;
                const int nblk = KIND == 2 ? ((base + q * 32) >> 5) : ((base + q * 32) >> 8);
                const int off = KIND == 2 ? 0 : (d_q * 64 < blk_bytes - 4 ? d_q * 64 : blk_bytes - 4);
                const unsigned char* rp = w + (size_t)gr * wrow + (size_t)nblk * blk_bytes + off;
                v = *reinterpret_cast<const unsigned*>(rp);
            }
            asm volatile("" : : "v"(v) : "memory");
        }
    };
#if PWW_PF == 2
    float4 px[PWW_XN];
    auto pf_x = [&](int base) {
#pragma unroll
        for (int q = 0; q < PWW_KQ; ++q)
#pragma unroll
        for (int p = 0; p < PWW_TT / PWW_XROW; ++p) {
            const int gt = tok_base + p * PWW_XROW + xt;
            const int kc = base + q * 32 + xc;
            px[q * (PWW_TT / PWW_XROW) + p] =
                gt < tokens ? *reinterpret_cast<const float4*>(x + (size_t)gt * k + kc)
                            : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    auto commit_x = [&](_Float16* xs) {
#pragma unroll
        for (int q = 0; q < PWW_KQ; ++q)
#pragma unroll
        for (int p = 0; p < PWW_TT / PWW_XROW; ++p) {
            const int rl = p * PWW_XROW + xt;
            const float4 v = px[q * (PWW_TT / PWW_XROW) + p];
            __half2* dst = reinterpret_cast<__half2*>(xs + rl * PWW_XS + q * 32 + xc);
            dst[0] = __floats2half2_rn(v.x, v.y);
            dst[1] = __floats2half2_rn(v.z, v.w);
        }
    };
#endif
#endif

    auto stage = [&](_Float16* xs, _Float16* ws, int base) {
#if PWW_PROBE_NO_STAGE
        (void)xs; (void)ws; (void)base;
#else
        stage_x(xs, base);
        stage_w(ws, base);
#endif
    };

    // One warp's share of one 16-wide k step: the 16x16x16 grid over its
    // accumulators. Splitting it lets a second buffer be staged between the two
    // halves, so the loads of the next tile overlap the matrix work.
    auto compute_kk = [&](const _Float16* xs, const _Float16* ws, int kk) {
#if PWW_PROBE_NO_COMPUTE
        (void)xs; (void)ws; (void)kk;
#elif PWW_PROBE_NO_FRAG
        (void)xs; (void)ws; (void)kk;
#pragma unroll
            for (int i = 0; i < PWW_NTG; ++i)
#pragma unroll
                for (int j = 0; j < PWW_NRG; ++j) {
                    const int idx = i * PWW_NRG + j;
                    acc[idx] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a0[i], b0[j], acc[idx]);
                }
#else
            pww_h8 a[PWW_NTG];
            pww_h8 b[PWW_NRG];
#pragma unroll
            for (int i = 0; i < PWW_NTG; ++i)
                a[i] = *reinterpret_cast<const pww_h8*>(xs + xoff + i * 16 * PWW_XS + kk * 16);
#pragma unroll
            for (int j = 0; j < PWW_NRG; ++j)
                b[j] = *reinterpret_cast<const pww_h8*>(ws + boff + j * 16 * PWW_WS + kk * 16);
#pragma unroll
            for (int i = 0; i < PWW_NTG; ++i)
#pragma unroll
                for (int j = 0; j < PWW_NRG; ++j) {
                    const int idx = i * PWW_NRG + j;
                    acc[idx] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a[i], b[j], acc[idx]);
                }
#endif
    };

    // One warp's share of the k-tile: the 16x16x16 grid over its accumulators.
    auto compute = [&](const _Float16* xs, const _Float16* ws) {
#if PWW_PROBE_NO_COMPUTE
        (void)xs; (void)ws;
#elif PWW_PROBE_NO_FRAG
        (void)xs; (void)ws;
#pragma unroll
        for (int kk = 0; kk < PWW_KW / 16; ++kk)
#pragma unroll
            for (int i = 0; i < PWW_NTG; ++i)
#pragma unroll
                for (int j = 0; j < PWW_NRG; ++j) {
                    const int idx = i * PWW_NRG + j;
                    acc[idx] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a0[i], b0[j], acc[idx]);
                }
#else
#pragma unroll
        for (int kk = 0; kk < PWW_KW / 16; ++kk) {
            pww_h8 a[PWW_NTG];
            pww_h8 b[PWW_NRG];
#pragma unroll
            for (int i = 0; i < PWW_NTG; ++i)
                a[i] = *reinterpret_cast<const pww_h8*>(xs + xoff + i * 16 * PWW_XS + kk * 16);
#pragma unroll
            for (int j = 0; j < PWW_NRG; ++j)
                b[j] = *reinterpret_cast<const pww_h8*>(ws + boff + j * 16 * PWW_WS + kk * 16);
#pragma unroll
            for (int i = 0; i < PWW_NTG; ++i)
#pragma unroll
                for (int j = 0; j < PWW_NRG; ++j) {
                    const int idx = i * PWW_NRG + j;
                    acc[idx] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a[i], b[j], acc[idx]);
                }
        }
#endif
    };

#if PWW_PF == 2
    pf_x(0);
    touch(0);
    __syncthreads();
    commit_x(shx);
    stage_w(shw, 0);
    __syncthreads();
    for (int base = 0; base < k; base += PWW_KW) {
        const bool more = base + PWW_KW < k;
        if (more) { pf_x(base + PWW_KW); touch(base + PWW_KW); }
        compute(shx, shw);
        __syncthreads();
        if (more) { commit_x(shx); stage_w(shw, base + PWW_KW); }
        __syncthreads();
    }
#elif PWW_PF == 1
    touch(0);
    __syncthreads();
    stage(shx, shw, 0);
    __syncthreads();
    for (int base = 0; base < k; base += PWW_KW) {
        const bool more = base + PWW_KW < k;
        if (more) touch(base + PWW_KW);
        compute(shx, shw);
        __syncthreads();
        if (more) stage(shx, shw, base + PWW_KW);
        __syncthreads();
    }
#elif PWW_DB
    // Double buffered: stage tile k+1 into the other buffer while tile k is
    // still being multiplied, so one barrier per tile instead of two and the
    // global loads / dequant of k+1 overlap the matrix work of k.
    stage(shx, shw, 0);
    __syncthreads();
    for (int base = 0, b = 0; base < k; base += PWW_KW, b ^= 1) {
        if (base + PWW_KW < k)
            stage(shx + (b ^ 1) * PWW_TT * PWW_XS, shw + (b ^ 1) * PWW_RT * PWW_WS,
                  base + PWW_KW);
        compute(shx + b * PWW_TT * PWW_XS, shw + b * PWW_RT * PWW_WS);
        __syncthreads();
    }
#else
    for (int base = 0; base < k; base += PWW_KW) {
        __syncthreads(); // the previous multiply has finished with the tiles
        stage(shx, shw, base);
        __syncthreads();
        compute(shx, shw);
    }
#endif


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

// Same kernels with the activation tile already fp16 in global memory.
extern "C" __global__ void prefill_q4_k_wh(
    const unsigned char* __restrict__ w,
    const void* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pww_body<0, true>(w, x, y, m, k, tokens);
}

extern "C" __global__ void prefill_q6_k_wh(
    const unsigned char* __restrict__ w,
    const void* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pww_body<1, true>(w, x, y, m, k, tokens);
}

extern "C" __global__ void prefill_q8_0_wh(
    const unsigned char* __restrict__ w,
    const void* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pww_body<2, true>(w, x, y, m, k, tokens);
}

extern "C" __global__ void prefill_halfw_wh(
    const unsigned char* __restrict__ w,
    const void* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pww_body<3, true>(w, x, y, m, k, tokens);
}

// f32 activations -> fp16, once per projection, so the GEMM streams half the
// x bytes (the activation re-read is the widest stream in the ffn shapes).
extern "C" __global__ void prefill_xhalf(
    const float* __restrict__ x,
    _Float16* __restrict__ xh,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xh[i] = (_Float16)x[i];
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
