// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's K-quant / q8_0 layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// Tiled prefill GEMM over Q8_0 weights. Structure identical to prefill_q4_k:
// one block owns 256 tokens x 64 output rows, the weight tile is dequantised
// once into shared and reused for every token, f32 accumulators in registers.
//
// Q8_0 block (34 bytes, 32 values): d(2) qs(32 signed bytes). A 32-value k-tile
// is exactly one Q8_0 block; a thread's 8 contiguous values are a quarter of it,
// so one scale covers them.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define PF8_BLOCK_BYTES 34
#define PF8_TPB 256
#define PF8_RPB 64
#define PF8_KT 32
#define PF8_TT 8
#define PF8_RR 8
#define PF8_TG (PF8_TPB / PF8_TT)
#define PF8_RG (PF8_RPB / PF8_RR)
#define PF8_THREADS (PF8_TG * PF8_RG)
#define PF8_XSTRIDE (PF8_KT + 1)
#define PF8_WSTRIDE (PF8_KT + 1)
#define PF8_HALF_TOKENS (PF8_TPB / 2)

// Dequantise 8 contiguous values of one row, starting at value index `v`
// (a multiple of 8) into `out`.
static __device__ __forceinline__ void pf8_dequant8(const unsigned char* __restrict__ rp,
                                                    int v, float* out)
{
    const int blk = v >> 5;
    const int part = (v & 31) >> 3;
    const unsigned char* b = rp + (size_t)blk * PF8_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const signed char* qs = reinterpret_cast<const signed char*>(b + 2 + part * 8);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = d * (float)qs[i];
}

extern "C" __global__ void prefill_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) float pf8_x[PF8_TPB * PF8_XSTRIDE];
    __shared__ __align__(16) float pf8_w[PF8_RPB * PF8_WSTRIDE];

    const int tid = threadIdx.x;
    const int tg = tid % PF8_TG;
    const int rg = tid / PF8_TG;
    const int tok_base = blockIdx.y * PF8_TPB;
    const int row_base = blockIdx.x * PF8_RPB;
    const int nblocks = k / 32;

    const int d_row = tid >> 2;
    const int d_q = tid & 3;

    float acc[PF8_TT][PF8_RR];
#pragma unroll
    for (int tt = 0; tt < PF8_TT; ++tt)
#pragma unroll
        for (int rr = 0; rr < PF8_RR; ++rr) acc[tt][rr] = 0.0f;

    const int xq = PF8_TPB * (PF8_KT / 4) / PF8_THREADS;

    for (int base = 0; base < k; base += PF8_KT) {
        __syncthreads();

        {
            const int r = row_base + d_row;
            if (r < m) {
                const unsigned char* rp =
                    w + (size_t)r * (size_t)nblocks * PF8_BLOCK_BYTES;
                float v[8];
                pf8_dequant8(rp, base + d_q * 8, v);
                float* dst = pf8_w + d_row * PF8_WSTRIDE + d_q * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = v[i];
            }
        }

        for (int s = 0; s < xq; ++s) {
            const int i = tid + s * PF8_THREADS;
            const int tl = i >> 3;
            const int c4 = (i & 7) * 4;
            const int gt = tok_base + tl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens) {
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + c4);
            }
            float* dst = pf8_x + tl * PF8_XSTRIDE + c4;
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        }

        __syncthreads();

        const int x0 = tg * PF8_XSTRIDE;
        const int w0 = rg * PF8_RR * PF8_WSTRIDE;
#pragma unroll 2
        for (int kk = 0; kk < PF8_KT; ++kk) {
            float xv[PF8_TT];
            float wv[PF8_RR];
#pragma unroll
            for (int tt = 0; tt < PF8_TT; ++tt)
                xv[tt] = pf8_x[x0 + tt * PF8_TG * PF8_XSTRIDE + kk];
#pragma unroll
            for (int rr = 0; rr < PF8_RR; ++rr)
                wv[rr] = pf8_w[w0 + rr * PF8_WSTRIDE + kk];
#pragma unroll
            for (int tt = 0; tt < PF8_TT; ++tt)
#pragma unroll
                for (int rr = 0; rr < PF8_RR; ++rr)
                    acc[tt][rr] = fmaf(xv[tt], wv[rr], acc[tt][rr]);
        }
    }

    float* sbuf = pf8_x;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < PF8_TT / 2; ++u)
#pragma unroll
            for (int rr = 0; rr < PF8_RR; ++rr)
                sbuf[(tg + PF8_TG * u) * PF8_RPB + (rg * PF8_RR + rr)] =
                    acc[half * (PF8_TT / 2) + u][rr];
        __syncthreads();
        for (int i = tid; i < PF8_HALF_TOKENS * PF8_RPB; i += PF8_THREADS) {
            const int tl = i / PF8_RPB;
            const int rl = i - tl * PF8_RPB;
            const int gt = tok_base + half * PF8_HALF_TOKENS + tl;
            const int gr = row_base + rl;
            if (gt < tokens && gr < m) y[(size_t)gt * m + gr] = sbuf[i];
        }
    }
}
