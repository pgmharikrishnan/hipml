// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's IQ4_NL layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// Tiled prefill GEMM over IQ4_NL weights.
//
// Same geometry as `prefill_q4_k`: 256 prompt tokens and 64 output rows per
// block, 256 threads, 8x8 accumulators, and the weight tile dequantised once
// into shared for every token in the block.
//
// IQ4_NL is 18 bytes per 32 values, so each thread dequantises 8 contiguous
// values: part 0/1 take the low nibbles of bytes 0..7 / 8..15 and part 2/3 the
// high nibbles of the same bytes.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define PFN4_BLOCK_BYTES 18
#define PFN4_TPB 256
#define PFN4_RPB 64
#define PFN4_KT 32
#define PFN4_TT 8
#define PFN4_RR 8
#define PFN4_TG (PFN4_TPB / PFN4_TT)
#define PFN4_RG (PFN4_RPB / PFN4_RR)
#define PFN4_THREADS (PFN4_TG * PFN4_RG)
#define PFN4_XSTRIDE (PFN4_KT + 1)
#define PFN4_WSTRIDE (PFN4_KT + 1)
#define PFN4_HALF_TOKENS (PFN4_TPB / 2)

static __device__ __constant__ const signed char PFN4_KV[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// 18-byte blocks are only byte-aligned past the scale; copy, don't cast.
static __device__ __forceinline__ void pfn4_read16(const unsigned char* src,
                                                   unsigned char* dst)
{
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = src[i];
}

static __device__ __forceinline__ void pfn4_dequant8(const unsigned char* __restrict__ rp,
                                                     int v, float* out)
{
    const int blk = v >> 5;
    const int part = (v & 31) >> 3;

    const unsigned char* b = rp + (size_t)blk * PFN4_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    unsigned char q[16];
    pfn4_read16(b + 2, q);
    const bool high = part >= 2;
    const int off = (part & 1) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned char byte = q[off + i];
        const int e = high ? (byte >> 4) : (byte & 0xF);
        out[i] = d * (float)PFN4_KV[e];
    }
}

extern "C" __global__ void prefill_iq4_nl(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) float pfn4_x[PFN4_TPB * PFN4_XSTRIDE];
    __shared__ __align__(16) float pfn4_w[PFN4_RPB * PFN4_WSTRIDE];

    const int tid = threadIdx.x;
    const int tg = tid % PFN4_TG;
    const int rg = tid / PFN4_TG;
    const int tok_base = blockIdx.y * PFN4_TPB;
    const int row_base = blockIdx.x * PFN4_RPB;
    const int nblocks = k / 32;

    const int d_row = tid >> 2;
    const int d_q = tid & 3;

    float acc[PFN4_TT][PFN4_RR];
#pragma unroll
    for (int tt = 0; tt < PFN4_TT; ++tt)
#pragma unroll
        for (int rr = 0; rr < PFN4_RR; ++rr) acc[tt][rr] = 0.0f;

    const int xq = PFN4_TPB * (PFN4_KT / 4) / PFN4_THREADS;

    for (int base = 0; base < k; base += PFN4_KT) {
        __syncthreads();

        {
            const int r = row_base + d_row;
            if (r < m) {
                const unsigned char* rp =
                    w + (size_t)r * (size_t)nblocks * PFN4_BLOCK_BYTES;
                float v[8];
                pfn4_dequant8(rp, base + d_q * 8, v);
                float* dst = pfn4_w + d_row * PFN4_WSTRIDE + d_q * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = v[i];
            }
        }

        for (int s = 0; s < xq; ++s) {
            const int i = tid + s * PFN4_THREADS;
            const int tl = i >> 3;
            const int c4 = (i & 7) * 4;
            const int gt = tok_base + tl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens) {
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + c4);
            }
            float* dst = pfn4_x + tl * PFN4_XSTRIDE + c4;
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        }

        __syncthreads();

        const int x0 = tg * PFN4_XSTRIDE;
        const int w0 = rg * PFN4_RR * PFN4_WSTRIDE;
#pragma unroll 2
        for (int kk = 0; kk < PFN4_KT; ++kk) {
            float xv[PFN4_TT];
            float wv[PFN4_RR];
#pragma unroll
            for (int tt = 0; tt < PFN4_TT; ++tt)
                xv[tt] = pfn4_x[x0 + tt * PFN4_TG * PFN4_XSTRIDE + kk];
#pragma unroll
            for (int rr = 0; rr < PFN4_RR; ++rr)
                wv[rr] = pfn4_w[w0 + rr * PFN4_WSTRIDE + kk];
#pragma unroll
            for (int tt = 0; tt < PFN4_TT; ++tt)
#pragma unroll
                for (int rr = 0; rr < PFN4_RR; ++rr)
                    acc[tt][rr] = fmaf(xv[tt], wv[rr], acc[tt][rr]);
        }
    }

    float* sbuf = pfn4_x;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < PFN4_TT / 2; ++u)
#pragma unroll
            for (int rr = 0; rr < PFN4_RR; ++rr)
                sbuf[(tg + PFN4_TG * u) * PFN4_RPB + (rg * PFN4_RR + rr)] =
                    acc[half * (PFN4_TT / 2) + u][rr];
        __syncthreads();
        for (int i = tid; i < PFN4_HALF_TOKENS * PFN4_RPB; i += PFN4_THREADS) {
            const int tl = i / PFN4_RPB;
            const int rl = i - tl * PFN4_RPB;
            const int gt = tok_base + half * PFN4_HALF_TOKENS + tl;
            const int gr = row_base + rl;
            if (gt < tokens && gr < m) y[(size_t)gt * m + gr] = sbuf[i];
        }
    }
}
