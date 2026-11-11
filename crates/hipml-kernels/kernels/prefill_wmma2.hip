// SPDX-License-Identifier: Apache-2.0
// Occupancy-driven prefill GEMM on the gfx12 matrix unit.
//
// Same contract and same numerics as `prefill_wmma.hip` (`y[t,:] = W @ x[t,:]`,
// weights dequantised into shared per k-tile, f32 accumulators, f32 out), but
// the block shape and the k loop are aimed at the measured bottleneck: the
// shipped 512-thread / 8-accumulator shape uses 115 VGPRs, so only one block
// fits on a CU and every warp of that block waits out the staging phase.
//
// This file has three independent switches, all measured with
// `scripts/probe_wmma2.cpp`:
//
//   PWW2_TT/RT/WT/WR  block shape. 256 threads with 4 accumulators per warp is
//                     84 VGPRs, so three blocks are resident per CU.
//   PWW2_PF           0  stage -> barrier -> multiply -> barrier (reference)
//                     1  double buffer, one barrier per k-tile
//                     2  double buffer plus an explicit register pipeline: the
//                        next tile's raw quant bytes and activations are loaded
//                        into registers before the multiply and only converted
//                        into shared after it, so the global latency overlaps
//                        the matrix work instead of preceding it.
//   PWW2_TOKFAST      swap the grid so token tiles run fastest, which keeps one
//                      weight panel hot in L2 across all token tiles.
//
// The k order is unchanged (16-value steps, in order), so every shape here is
// bit-identical to the shipped kernel; only the tile boundaries move.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "prefill_common.inc"

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "prefill_wmma2: the gfx12 WMMA intrinsics are wave32 only"
#endif

#ifndef PWW2_TT
#define PWW2_TT 256        // prompt tokens per block
#endif
#ifndef PWW2_RT
#define PWW2_RT 128        // output rows per block
#endif
#ifndef PWW2_WT
#define PWW2_WT 64         // tokens per warp
#endif
#ifndef PWW2_WR
#define PWW2_WR 64         // rows per warp
#endif
#ifndef PWW2_KW
#define PWW2_KW 32         // k values per shared k-tile
#endif
#ifndef PWW2_THREADS
#define PWW2_THREADS 256
#endif
#ifndef PWW2_PF
#define PWW2_PF 0
#endif
// One resident block is all the 16-accumulator shape can have, so ask the
// compiler for the whole register file: a pipelined build that is capped at the
// two-block limit spills instead of hoisting its prefetch loads.
#ifndef PWW2_MIN_BLOCKS
#define PWW2_MIN_BLOCKS 1
#endif
#ifndef PWW2_TOKFAST
#define PWW2_TOKFAST 0
#endif
// Probe-only switches: split the loop's cost into staging, fragment reads and
// matrix work. They default to 0 and the shipped kernel is unaffected.
#ifndef PWW2_PROBE_NO_STAGE
#define PWW2_PROBE_NO_STAGE 0
#endif
#ifndef PWW2_PROBE_NO_FRAG
#define PWW2_PROBE_NO_FRAG 0
#endif
#ifndef PWW2_PROBE_NO_COMPUTE
#define PWW2_PROBE_NO_COMPUTE 0
#endif

#define PWW2_XS (PWW2_KW + 8)   // x row stride in halfs (16 B aligned)
#define PWW2_WS (PWW2_KW + 8)   // weight row stride in halfs
#define PWW2_KQ (PWW2_KW / 32)  // 32-value staging passes per row
#define PWW2_BUF (PWW2_PF ? 2 : 1)
#define PWW2_XROW (PWW2_THREADS / 8)          // token rows staged per pass, f32 x
#define PWW2_XROW_H (PWW2_THREADS / 4)        // token rows staged per pass, fp16 x
#define PWW2_DPROW (PWW2_THREADS / 4)         // weight rows dequantised per pass
#define PWW2_RP ((PWW2_RT + PWW2_DPROW - 1) / PWW2_DPROW)
#define PWW2_WARPS (PWW2_THREADS / 32)
#define PWW2_TG (PWW2_TT / PWW2_WT)
#define PWW2_RG (PWW2_RT / PWW2_WR)
#define PWW2_NTG (PWW2_WT / 16)
#define PWW2_NRG (PWW2_WR / 16)
#define PWW2_ACC (PWW2_NTG * PWW2_NRG)
#define PWW2_WN (PWW2_KQ * PWW2_RP)
#define PWW2_XN (PWW2_KQ * ((PWW2_TT + PWW2_XROW - 1) / PWW2_XROW))
#define PWW2_XNH (PWW2_KQ * ((PWW2_TT + PWW2_XROW_H - 1) / PWW2_XROW_H))

static_assert(PWW2_TG * PWW2_RG == PWW2_WARPS, "warp grid must cover the block");
static_assert(PWW2_TT % PWW2_WT == 0 && PWW2_RT % PWW2_WR == 0, "ragged warp tiles");
static_assert(PWW2_WT % 16 == 0 && PWW2_WR % 16 == 0, "warp tile must be a 16 tile");
static_assert(PWW2_TT % PWW2_XROW == 0 && PWW2_TT % PWW2_XROW_H == 0, "ragged x staging");
static_assert(PWW2_KW % 32 == 0 && PWW2_KW <= 128, "k tile must be 32..128");

typedef _Float16 __attribute__((ext_vector_type(8))) pww2_h8;
typedef float __attribute__((ext_vector_type(8))) pww2_f8;
typedef unsigned int __attribute__((ext_vector_type(4))) pww2_u4;

static __device__ __forceinline__ void pww2_ld8b(const unsigned char* p, pww2_u4& r, int lo)
{
    unsigned a, b;
    __builtin_memcpy(&a, p, 4);
    __builtin_memcpy(&b, p + 4, 4);
    if (lo) { r.x = a; r.y = b; } else { r.z = a; r.w = b; }
}

// One half out of the four 32-bit words of a prefetched 16-byte run.
static __device__ __forceinline__ __half pww2_h(const pww2_u4& r, int i)
{
    const unsigned w = i < 2 ? r.x : (i < 4 ? r.y : (i < 6 ? r.z : r.w));
    return __ushort_as_half((unsigned short)((w >> (16 * (i & 1))) & 0xFFFFu));
}

// One byte out of the two 32-bit words of a prefetched 8-byte run.
static __device__ __forceinline__ int pww2_b(unsigned lo, unsigned hi, int i)
{
    return (int)(((i < 4 ? lo >> (8 * i) : hi >> (8 * (i - 4)))) & 0xFFu);
}

// Raw quant bytes an 8-value run needs, loaded early and dequantised later.
template <int KIND>
static __device__ __forceinline__ pww2_u4 pww2_loadraw(const unsigned char* __restrict__ rp, int v)
{
    pww2_u4 r;
    r.x = r.y = r.z = r.w = 0u;
    if (KIND == 0) {
        const int blk = v >> 8, within = v & 255;
        const unsigned char* qp = rp + (size_t)blk * PF_Q4K_BLOCK_BYTES + 16 +
                                  ((within >> 5) >> 1) * 32 + ((within & 31) >> 3) * 8;
        pww2_ld8b(qp, r, 1);
    } else if (KIND == 1) {
        const int blk = v >> 8, within = v & 255, j = within >> 4, l0 = within & 15;
        const int half = j >> 3, s = j & 7, group = s >> 1, is = s & 1;
        const unsigned char* b = rp + (size_t)blk * PF_Q6K_BLOCK_BYTES;
        const int ql = half * 64 + (group & 1) * 32 + is * 16 + l0;
        const int qh = 128 + half * 32 + is * 16 + l0;
        pww2_ld8b(b + ql, r, 1);
        pww2_ld8b(b + qh, r, 0);
    } else if (KIND == 2) {
        const int blk = v >> 5, part = (v & 31) >> 3;
        pww2_ld8b(rp + (size_t)blk * PF_Q8_0_BLOCK_BYTES + 2 + part * 8, r, 1);
    } else if (KIND == 3) {
        __builtin_memcpy(&r, rp + (size_t)v * 2, 16);
    } else if (KIND == 4) {
        const int blk = v >> 8, within = v & 255, sub = within >> 5;
        const int part = (within & 31) >> 3;
        pww2_ld8b(rp + (size_t)blk * PF_IQ4_XS_BLOCK_BYTES + 8 + sub * 16 +
                      (part & 1) * 8, r, 1);
    } else {
        const int blk = v >> 5, part = (v & 31) >> 3;
        pww2_ld8b(rp + (size_t)blk * PF_IQ4_NL_BLOCK_BYTES + 2 + (part & 1) * 8, r, 1);
    }
    return r;
}

// Dequantise one 8-value run from the prefetched bytes. Scales, mins and the
// Q6_K/Q8_0 block scales are read here: they are 2-16 bytes per 128-256 values
// and land in the same 128 B line the raw load already pulled in.
template <int KIND>
static __device__ __forceinline__ void pww2_dequant8(const pww2_u4& r,
                                                     const unsigned char* __restrict__ rp,
                                                     int v, float* out)
{
    if (KIND == 0) {
        const int blk = v >> 8, within = v & 255, sub = within >> 5;
        const unsigned char* b = rp + (size_t)blk * PF_Q4K_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b)));
        const float dmin = __half2float(pf_ldh(reinterpret_cast<const __half*>(b + 2)));
        int sc, mn;
        pf_scale_min(sub, b + 4, &sc, &mn);
        const float scale = d * (float)sc;
        const float minv = dmin * (float)mn;
        const bool high = (sub & 1) != 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = pww2_b(r.x, r.y, i);
            out[i] = scale * (float)(high ? (q >> 4) : (q & 0xF)) - minv;
        }
    } else if (KIND == 1) {
        const int blk = v >> 8, within = v & 255, j = within >> 4;
        const unsigned char* b = rp + (size_t)blk * PF_Q6K_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b + 208)));
        const float scale = d * (float)(signed char)pf_ld8(b + 192 + j);
        const int s = j & 7, group = s >> 1, is = s & 1;
        const int shift = 2 * group;
        const bool high = group >= 2;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int qb = pww2_b(r.x, r.y, i);
            const int hb = (pww2_b(r.z, r.w, i) >> shift) & 3;
            const int nib = high ? (qb >> 4) : (qb & 0xF);
            out[i] = scale * (float)((nib | (hb << 4)) - 32);
        }
    } else if (KIND == 2) {
        const int blk = v >> 5, part = (v & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_Q8_0_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b)));
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = d * (float)(signed char)pww2_b(r.x, r.y, i);
    } else if (KIND == 3) {
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = __half2float(pww2_h(r, i));
    } else if (KIND == 4) {
        const int blk = v >> 8, within = v & 255, sub = within >> 5;
        const int part = (within & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_IQ4_XS_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b)));
        const unsigned short scales_h =
            *reinterpret_cast<const unsigned short*>(b + 2);
        const int ls = ((b[4 + (sub >> 1)] >> (4 * (sub & 1))) & 0xF) |
                       (((scales_h >> (2 * sub)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const bool high = part >= 2;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = pww2_b(r.x, r.y, i);
            out[i] = dl * (float)PF_IQ4NL_KV[high ? (q >> 4) : (q & 0xF)];
        }
    } else {
        const int blk = v >> 5, part = (v & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_IQ4_NL_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b)));
        const bool high = part >= 2;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = pww2_b(r.x, r.y, i);
            out[i] = d * (float)PF_IQ4NL_KV[high ? (q >> 4) : (q & 0xF)];
        }
    }
}

// SPLIT: two slices, the first writing `y` and the second `scratch` (WP36).
// NS: `n` equal slices stacked along blockIdx.z; slice 0 writes `y` and slice
// `z > 0` writes `scratch` at `(z - 1) * tokens * m`, so the caller sums them
// in order. NS exists for the 64-row SSM projections, where one row tile is so
// few blocks that the matrix units idle.
template <int KIND, bool XHALF, bool SPLIT = false, bool NS = false>
static __device__ __forceinline__ void pww2_body_impl(
    const unsigned char* __restrict__ w,
    const void* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ scratch,
    int m,
    int k,
    int tokens,
    int ks)
{
    const int z = SPLIT ? (int)blockIdx.z : 0;
    const int kb = NS ? z * ks : (SPLIT ? (z ? ks : 0) : 0);
    const int ke = NS ? ((kb + ks < k) ? kb + ks : k) : (SPLIT ? (z ? k : ks) : k);
    float* __restrict__ out =
        NS ? (z ? scratch + (size_t)(z - 1) * tokens * m : y)
           : (SPLIT ? (z ? scratch : y) : y);
    __shared__ __align__(16) _Float16 shx[PWW2_BUF * PWW2_TT * PWW2_XS];
    __shared__ __align__(16) _Float16 shw[PWW2_BUF * PWW2_RT * PWW2_WS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wtg = warp / PWW2_RG;
    const int wrg = warp % PWW2_RG;

    const int tok_base = (PWW2_TOKFAST ? blockIdx.x : blockIdx.y) * PWW2_TT;
    const int row_base = (PWW2_TOKFAST ? blockIdx.y : blockIdx.x) * PWW2_RT;
    const size_t wrow = pf_row_bytes(k, KIND);

    const int xt = XHALF ? (tid >> 2) : (tid >> 3);
    const int xc = XHALF ? (tid & 3) * 8 : (tid & 7) * 4;
    const int d_row = tid >> 2;
    const int d_q = tid & 3;

    const int frow = lane & 15;
    const int fcol = (lane >> 4) * 8;
    const int xoff = (wtg * PWW2_WT + frow) * PWW2_XS + fcol;
    const int boff = (wrg * PWW2_WR + frow) * PWW2_WS + fcol;

    pww2_f8 acc[PWW2_ACC];
#pragma unroll
    for (int j = 0; j < PWW2_ACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;

#if PWW2_PROBE_NO_FRAG
    pww2_h8 a0[PWW2_NTG];
    pww2_h8 b0[PWW2_NRG];
#pragma unroll
    for (int i = 0; i < PWW2_NTG; ++i)
        a0[i] = *reinterpret_cast<const pww2_h8*>(shx + xoff + i * 16 * PWW2_XS);
#pragma unroll
    for (int j = 0; j < PWW2_NRG; ++j)
        b0[j] = *reinterpret_cast<const pww2_h8*>(shw + boff + j * 16 * PWW2_WS);
#endif

    // Progressive prefix of the raw loads so the arrays are declared once.
    pww2_u4 wraw[PWW2_WN];
    float4 xf[PWW2_XN];
    pww2_u4 xh[PWW2_XNH];

    auto load_x = [&](int base) {
#pragma unroll
        for (int q = 0; q < PWW2_KQ; ++q)
#pragma unroll
            for (int p = 0; p < (XHALF ? PWW2_TT / PWW2_XROW_H : PWW2_TT / PWW2_XROW); ++p) {
                const int idx = q * (XHALF ? PWW2_TT / PWW2_XROW_H : PWW2_TT / PWW2_XROW) + p;
                const int rl = p * (XHALF ? PWW2_XROW_H : PWW2_XROW) + xt;
                const int gt = tok_base + rl;
                const int kc = base + q * 32 + xc;
                if (gt < tokens) {
                    if (XHALF) {
                        const unsigned char* sp =
                            reinterpret_cast<const unsigned char*>(x) + (size_t)gt * k * 2 + kc * 2;
                        __builtin_memcpy(&xh[idx], sp, 16);
                    } else {
                        xf[idx] = *reinterpret_cast<const float4*>(
                            reinterpret_cast<const float*>(x) + (size_t)gt * k + kc);
                    }
                } else {
                    if (XHALF) xh[idx].x = xh[idx].y = xh[idx].z = xh[idx].w = 0u;
                    else xf[idx] = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
    };

    auto load_w = [&](int base) {
#pragma unroll
        for (int q = 0; q < PWW2_KQ; ++q)
#pragma unroll
            for (int p = 0; p < PWW2_RP; ++p) {
                const int idx = q * PWW2_RP + p;
                const int rl = p * PWW2_DPROW + d_row;
                const int gr = row_base + rl;
                const int kc = base + q * 32 + d_q * 8;
                if (rl < PWW2_RT && gr < m)
                    wraw[idx] = pww2_loadraw<KIND>(w + (size_t)gr * wrow, kc);
                else
                    wraw[idx].x = wraw[idx].y = wraw[idx].z = wraw[idx].w = 0u;
            }
    };

    auto commit_x = [&](_Float16* xs, int base) {
        (void)base;
#pragma unroll
        for (int q = 0; q < PWW2_KQ; ++q)
#pragma unroll
            for (int p = 0; p < (XHALF ? PWW2_TT / PWW2_XROW_H : PWW2_TT / PWW2_XROW); ++p) {
                const int idx = q * (XHALF ? PWW2_TT / PWW2_XROW_H : PWW2_TT / PWW2_XROW) + p;
                const int rl = p * (XHALF ? PWW2_XROW_H : PWW2_XROW) + xt;
                if (XHALF) {
                    *reinterpret_cast<pww2_u4*>(xs + rl * PWW2_XS + q * 32 + xc) = xh[idx];
                } else {
                    const float4 v = xf[idx];
                    __half2* dst = reinterpret_cast<__half2*>(xs + rl * PWW2_XS + q * 32 + xc);
                    dst[0] = __floats2half2_rn(v.x, v.y);
                    dst[1] = __floats2half2_rn(v.z, v.w);
                }
            }
    };

    auto commit_w = [&](_Float16* ws, int base) {
#pragma unroll
        for (int q = 0; q < PWW2_KQ; ++q)
#pragma unroll
            for (int p = 0; p < PWW2_RP; ++p) {
                const int idx = q * PWW2_RP + p;
                const int rl = p * PWW2_DPROW + d_row;
                if (rl >= PWW2_RT) continue;
                const int gr = row_base + rl;
                const int kc = base + q * 32 + d_q * 8;
                float v[8];
                if (gr < m) {
                    pww2_dequant8<KIND>(wraw[idx], w + (size_t)gr * wrow, kc, v);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) v[i] = 0.f;
                }
                pww2_h8 hv;
#pragma unroll
                for (int i = 0; i < 8; ++i) hv[i] = (_Float16)v[i];
                *reinterpret_cast<pww2_h8*>(ws + rl * PWW2_WS + q * 32 + d_q * 8) = hv;
            }
    };

    auto stage = [&](_Float16* xs, _Float16* ws, int base) {
#if PWW2_PROBE_NO_STAGE
        (void)xs; (void)ws; (void)base;
#else
        load_x(base);
        commit_x(xs, base);
        load_w(base);
        commit_w(ws, base);
#endif
    };

    auto compute_kk = [&](const _Float16* xs, const _Float16* ws, int kk) {
#if PWW2_PROBE_NO_COMPUTE
        (void)xs; (void)ws; (void)kk;
#elif PWW2_PROBE_NO_FRAG
        (void)xs; (void)ws; (void)kk;
#pragma unroll
        for (int i = 0; i < PWW2_NTG; ++i)
#pragma unroll
            for (int j = 0; j < PWW2_NRG; ++j) {
                const int idx = i * PWW2_NRG + j;
                acc[idx] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0[i], b0[j], acc[idx]);
            }
#else
        pww2_h8 a[PWW2_NTG];
        pww2_h8 b[PWW2_NRG];
#pragma unroll
        for (int i = 0; i < PWW2_NTG; ++i)
            a[i] = *reinterpret_cast<const pww2_h8*>(xs + xoff + i * 16 * PWW2_XS + kk * 16);
#pragma unroll
        for (int j = 0; j < PWW2_NRG; ++j)
            b[j] = *reinterpret_cast<const pww2_h8*>(ws + boff + j * 16 * PWW2_WS + kk * 16);
#pragma unroll
        for (int i = 0; i < PWW2_NTG; ++i)
#pragma unroll
            for (int j = 0; j < PWW2_NRG; ++j) {
                const int idx = i * PWW2_NRG + j;
                acc[idx] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[idx]);
            }
#endif
    };

    auto compute = [&](const _Float16* xs, const _Float16* ws) {
#pragma unroll
        for (int kk = 0; kk < PWW2_KW / 16; ++kk) compute_kk(xs, ws, kk);
    };

#if PWW2_PF >= 1
    stage(shx, shw, kb);
    __syncthreads();
    for (int base = kb, b = 0; base < ke; base += PWW2_KW, b ^= 1) {
        const bool more = base + PWW2_KW < ke;
#if PWW2_PF >= 3
        // x-only pipeline: the fp16 activation run is the widest global stream
        // in the tile, and it needs no conversion, so hold just that array in
        // registers across the multiply. The weight bytes are fetched after it,
        // which keeps the live set inside the register file (PF=2, which holds
        // both, spills at this shape).
#if !PWW2_PROBE_NO_STAGE
        if (more) load_x(base + PWW2_KW);
#endif
        compute(shx + b * PWW2_TT * PWW2_XS, shw + b * PWW2_RT * PWW2_WS);
#if !PWW2_PROBE_NO_STAGE
        if (more) {
            load_w(base + PWW2_KW);
            commit_x(shx + (b ^ 1) * PWW2_TT * PWW2_XS, base + PWW2_KW);
            commit_w(shw + (b ^ 1) * PWW2_RT * PWW2_WS, base + PWW2_KW);
        }
#endif
#elif PWW2_PF >= 2
#if !PWW2_PROBE_NO_STAGE
        if (more) {
            load_x(base + PWW2_KW);
            load_w(base + PWW2_KW);
        }
#endif
        compute(shx + b * PWW2_TT * PWW2_XS, shw + b * PWW2_RT * PWW2_WS);
#if !PWW2_PROBE_NO_STAGE
        if (more) {
            commit_x(shx + (b ^ 1) * PWW2_TT * PWW2_XS, base + PWW2_KW);
            commit_w(shw + (b ^ 1) * PWW2_RT * PWW2_WS, base + PWW2_KW);
        }
#endif
#else
        if (more) {
            load_x(base + PWW2_KW);
            commit_x(shx + (b ^ 1) * PWW2_TT * PWW2_XS, base + PWW2_KW);
            load_w(base + PWW2_KW);
            commit_w(shw + (b ^ 1) * PWW2_RT * PWW2_WS, base + PWW2_KW);
        }
        compute(shx + b * PWW2_TT * PWW2_XS, shw + b * PWW2_RT * PWW2_WS);
#endif
        __syncthreads();
    }
#else
    for (int base = kb; base < ke; base += PWW2_KW) {
        __syncthreads();
        stage(shx, shw, base);
        __syncthreads();
        compute(shx, shw);
    }
#endif

    const int mrow = 8 * (lane >> 4);
    const int gr = row_base + wrg * PWW2_WR + (lane & 15);
#pragma unroll
    for (int i = 0; i < PWW2_NTG; ++i)
#pragma unroll
        for (int j = 0; j < PWW2_NRG; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int gt = tok_base + wtg * PWW2_WT + i * 16 + mrow + e;
                const int r = gr + j * 16;
                if (gt < tokens && r < m) {
                    const size_t o = (size_t)gt * m + r;
                    out[o] = acc[i * PWW2_NRG + j][e];
                }
            }
        }
}

template <int KIND, bool XHALF>
static __device__ __forceinline__ void pww2_body(
    const unsigned char* __restrict__ w,
    const void* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pww2_body_impl<KIND, XHALF, false>(w, x, y, nullptr, m, k, tokens, 0);
}

template <int KIND, bool XHALF>
static __device__ __forceinline__ void pww2_body_split(
    const unsigned char* __restrict__ w,
    const void* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ scratch,
    int m,
    int k,
    int tokens,
    int ks)
{
    pww2_body_impl<KIND, XHALF, true>(w, x, y, scratch, m, k, tokens, ks);
}

template <int KIND, bool XHALF>
static __device__ __forceinline__ void pww2_body_slice(
    const unsigned char* __restrict__ w,
    const void* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ scratch,
    int m,
    int k,
    int tokens,
    int ks)
{
    pww2_body_impl<KIND, XHALF, true, true>(w, x, y, scratch, m, k, tokens, ks);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_q4_k_w2(const unsigned char* __restrict__ w,
                                           const float* __restrict__ x,
                                           float* __restrict__ y, int m, int k, int tokens)
{
    pww2_body<0, false>(w, x, y, m, k, tokens);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_q6_k_w2(const unsigned char* __restrict__ w,
                                           const float* __restrict__ x,
                                           float* __restrict__ y, int m, int k, int tokens)
{
    pww2_body<1, false>(w, x, y, m, k, tokens);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_q8_0_w2(const unsigned char* __restrict__ w,
                                           const float* __restrict__ x,
                                           float* __restrict__ y, int m, int k, int tokens)
{
    pww2_body<2, false>(w, x, y, m, k, tokens);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_halfw_w2(const unsigned char* __restrict__ w,
                                            const float* __restrict__ x,
                                            float* __restrict__ y, int m, int k, int tokens)
{
    pww2_body<3, false>(w, x, y, m, k, tokens);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_q4_k_w2h(const unsigned char* __restrict__ w,
                                            const void* __restrict__ x,
                                            float* __restrict__ y, int m, int k, int tokens)
{
    pww2_body<0, true>(w, x, y, m, k, tokens);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_q6_k_w2h(const unsigned char* __restrict__ w,
                                            const void* __restrict__ x,
                                            float* __restrict__ y, int m, int k, int tokens)
{
    pww2_body<1, true>(w, x, y, m, k, tokens);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_q8_0_w2h(const unsigned char* __restrict__ w,
                                            const void* __restrict__ x,
                                            float* __restrict__ y, int m, int k, int tokens)
{
    pww2_body<2, true>(w, x, y, m, k, tokens);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_halfw_w2h(const unsigned char* __restrict__ w,
                                             const void* __restrict__ x,
                                             float* __restrict__ y, int m, int k, int tokens)
{
    pww2_body<3, true>(w, x, y, m, k, tokens);
}

// IQ4 family: same tile, the IQ4_XS/IQ4_NL decode lives in prefill_common.inc.
extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_iq4_xs_w2h(const unsigned char* __restrict__ w,
                                             const void* __restrict__ x,
                                             float* __restrict__ y, int m, int k, int tokens)
{
    pww2_body<4, true>(w, x, y, m, k, tokens);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_iq4_nl_w2h(const unsigned char* __restrict__ w,
                                             const void* __restrict__ x,
                                             float* __restrict__ y, int m, int k, int tokens)
{
    pww2_body<5, true>(w, x, y, m, k, tokens);
}

// Split-K variants: same body over a k slice. The launcher runs [0,ks) with
// accumulate 0 and [ks,k) with accumulate 1, so the second launch adds its
// partial to the first. Launches on one stream are ordered, so no atomics and
// the two-partial sum is deterministic.
extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_q4_k_w2h_sk(const unsigned char* __restrict__ w,
                                               const void* __restrict__ x,
                                               float* __restrict__ y, float* __restrict__ scratch, int m, int k, int tokens,
                                               int ks)
{
    pww2_body_split<0, true>(w, x, y, scratch, m, k, tokens, ks);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_q6_k_w2h_sk(const unsigned char* __restrict__ w,
                                               const void* __restrict__ x,
                                               float* __restrict__ y, float* __restrict__ scratch, int m, int k, int tokens,
                                               int ks)
{
    pww2_body_split<1, true>(w, x, y, scratch, m, k, tokens, ks);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_q8_0_w2h_sk(const unsigned char* __restrict__ w,
                                               const void* __restrict__ x,
                                               float* __restrict__ y, float* __restrict__ scratch, int m, int k, int tokens,
                                               int ks)
{
    pww2_body_split<2, true>(w, x, y, scratch, m, k, tokens, ks);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_halfw_w2h_sk(const unsigned char* __restrict__ w,
                                                const void* __restrict__ x,
                                                float* __restrict__ y, float* __restrict__ scratch, int m, int k, int tokens,
                                                int ks)
{
    pww2_body_split<3, true>(w, x, y, scratch, m, k, tokens, ks);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_iq4_xs_w2h_sk(const unsigned char* __restrict__ w,
                                                const void* __restrict__ x,
                                                float* __restrict__ y, float* __restrict__ scratch, int m, int k, int tokens,
                                                int ks)
{
    pww2_body_split<4, true>(w, x, y, scratch, m, k, tokens, ks);
}

extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_iq4_nl_w2h_sk(const unsigned char* __restrict__ w,
                                                const void* __restrict__ x,
                                                float* __restrict__ y, float* __restrict__ scratch, int m, int k, int tokens,
                                                int ks)
{
    pww2_body_split<5, true>(w, x, y, scratch, m, k, tokens, ks);
}

// n equal k slices in one launch: `blockIdx.z` selects the slice, `ks` is its
// width. Slice 0 writes `y`, the rest write `scratch` in z order; the host sums
// them with `reduce_slices_f32`.
extern "C" __global__ void __launch_bounds__(PWW2_THREADS, PWW2_MIN_BLOCKS) prefill_halfw_w2h_sn(const unsigned char* __restrict__ w,
                                                const void* __restrict__ x,
                                                float* __restrict__ y, float* __restrict__ scratch, int m, int k, int tokens,
                                                int ks)
{
    pww2_body_slice<3, true>(w, x, y, scratch, m, k, tokens, ks);
}
