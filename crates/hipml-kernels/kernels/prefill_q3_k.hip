// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's Q3_K layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// Tiled prefill GEMM over Q3_K weights.
//
// y[t, row] = W[row, :] . x[t, :] for `tokens` rows of x at once. Same geometry
// and shared staging as `prefill_q4_k`: a weight tile is dequantised ONCE into
// shared and reused for every prompt token in the block.
//
//   PF3_TPB 256 tokens per block, PF3_RPB 64 output rows, PF3_KT 32 k/tile.
//   256 threads: 32 token groups x 8 row groups, 8x8 accumulators per thread.
//
// Each thread dequantises 8 contiguous values, a quarter of a 32-value
// sub-block: one `n`/`j`/`half` triple, so one six-bit scale, one 2-bit shift
// and one hmask bit cover the run. A Q3_K super-block is 110 bytes, so the
// block base is two-byte aligned and the quant bytes are read as bytes.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define PF3_BLOCK_BYTES 110
#define PF3_TPB 256
#define PF3_RPB 64
#define PF3_KT 32
#define PF3_TT 8
#define PF3_RR 8
#define PF3_TG (PF3_TPB / PF3_TT)
#define PF3_RG (PF3_RPB / PF3_RR)
#define PF3_THREADS (PF3_TG * PF3_RG)
#define PF3_XSTRIDE (PF3_KT + 1)
#define PF3_WSTRIDE (PF3_KT + 1)
#define PF3_HALF_TOKENS (PF3_TPB / 2)

static __device__ __forceinline__ unsigned int pf3_ld32(const unsigned char* p)
{
    unsigned int v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

// `scales[is] - 32` from the 12 packed bytes at b+96 (the Q3_K shuffle).
static __device__ __forceinline__ int pf3_scale(const unsigned char* __restrict__ b, int is)
{
    const unsigned char* s = b + 96;
    const unsigned int KM1 = 0x03030303u;
    const unsigned int KM2 = 0x0f0f0f0fu;
    const int ai = is >> 2;
    unsigned int a;
    if (ai == 0) {
        a = (pf3_ld32(s) & KM2) | ((pf3_ld32(s + 8) & KM1) << 4);
    } else if (ai == 1) {
        a = (pf3_ld32(s + 4) & KM2) | (((pf3_ld32(s + 8) >> 2) & KM1) << 4);
    } else if (ai == 2) {
        a = ((pf3_ld32(s) >> 4) & KM2) | (((pf3_ld32(s + 8) >> 4) & KM1) << 4);
    } else {
        a = ((pf3_ld32(s + 4) >> 4) & KM2) | (((pf3_ld32(s + 8) >> 6) & KM1) << 4);
    }
    return (int)(signed char)((a >> (8 * (is & 3))) & 0xFFu) - 32;
}

// Dequantise 8 contiguous values of one row, starting at value index `v`
// (a multiple of 8) into `out`.
static __device__ __forceinline__ void pf3_dequant8(const unsigned char* __restrict__ rp,
                                                    int v, float* out)
{
    const int blk = v >> 8;
    const int within = v & 255;
    const int n = within >> 7;
    const int j = (within >> 5) & 3;
    const int half = (within >> 4) & 1;
    const int l0 = within & 15;

    const unsigned char* b = rp + (size_t)blk * PF3_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b + 108));
    const float scale = d * (float)pf3_scale(b, n * 8 + j * 2 + half);
    const unsigned char* qp = b + 32 + n * 32 + half * 16 + l0;
    const unsigned char* hp = b + half * 16 + l0;
    const int shift = 2 * j;
    const int bit = 1 << (n * 4 + j);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = (qp[i] >> shift) & 3;
        const int sub = (hp[i] & bit) ? 0 : 4;
        out[i] = scale * (float)(q - sub);
    }
}

extern "C" __global__ void prefill_q3_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) float pf3_x[PF3_TPB * PF3_XSTRIDE];
    __shared__ __align__(16) float pf3_w[PF3_RPB * PF3_WSTRIDE];

    const int tid = threadIdx.x;
    const int tg = tid % PF3_TG;   // token group: tokens tg + 32*tt
    const int rg = tid / PF3_TG;   // row group: rows rg*8 .. rg*8+7
    const int tok_base = blockIdx.y * PF3_TPB;
    const int row_base = blockIdx.x * PF3_RPB;
    const int nblocks = k / 256;

    const int d_row = tid >> 2;    // 64 rows
    const int d_q = tid & 3;       // 4 chunks of 8 values per 32-value k-tile

    float acc[PF3_TT][PF3_RR];
#pragma unroll
    for (int tt = 0; tt < PF3_TT; ++tt)
#pragma unroll
        for (int rr = 0; rr < PF3_RR; ++rr) acc[tt][rr] = 0.0f;

    const int xq = PF3_TPB * (PF3_KT / 4) / PF3_THREADS; // 8 float4 per thread

    for (int base = 0; base < k; base += PF3_KT) {
        __syncthreads(); // previous multiply finished before shared is reused

        // Weight tile: 64 rows x 32 values, dequantised once.
        {
            const int r = row_base + d_row;
            if (r < m) {
                const unsigned char* rp =
                    w + (size_t)r * (size_t)nblocks * PF3_BLOCK_BYTES;
                float v[8];
                pf3_dequant8(rp, base + d_q * 8, v);
                float* dst = pf3_w + d_row * PF3_WSTRIDE + d_q * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = v[i];
            }
        }

        // x tile: 256 tokens x 32 values, token-major.
        for (int s = 0; s < xq; ++s) {
            const int i = tid + s * PF3_THREADS;
            const int tl = i >> 3;         // token within the block
            const int c4 = (i & 7) * 4;    // k offset, multiple of 4
            const int gt = tok_base + tl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens) {
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + c4);
            }
            float* dst = pf3_x + tl * PF3_XSTRIDE + c4;
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        }

        __syncthreads();

        const int x0 = tg * PF3_XSTRIDE;
        const int w0 = rg * PF3_RR * PF3_WSTRIDE;
#pragma unroll 2
        for (int kk = 0; kk < PF3_KT; ++kk) {
            float xv[PF3_TT];
            float wv[PF3_RR];
#pragma unroll
            for (int tt = 0; tt < PF3_TT; ++tt)
                xv[tt] = pf3_x[x0 + tt * PF3_TG * PF3_XSTRIDE + kk];
#pragma unroll
            for (int rr = 0; rr < PF3_RR; ++rr)
                wv[rr] = pf3_w[w0 + rr * PF3_WSTRIDE + kk];
#pragma unroll
            for (int tt = 0; tt < PF3_TT; ++tt)
#pragma unroll
                for (int rr = 0; rr < PF3_RR; ++rr)
                    acc[tt][rr] = fmaf(xv[tt], wv[rr], acc[tt][rr]);
        }
    }

    // Epilogue in two 128-token halves so the x buffer (33.8 KB) can hold the
    // transpose of the 8 KB output half. Global stores are then coalesced along
    // `row`: y is row-major [tokens, m], so consecutive rows are consecutive.
    float* sbuf = pf3_x;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < PF3_TT / 2; ++u)
#pragma unroll
            for (int rr = 0; rr < PF3_RR; ++rr)
                sbuf[(tg + PF3_TG * u) * PF3_RPB + (rg * PF3_RR + rr)] =
                    acc[half * (PF3_TT / 2) + u][rr];
        __syncthreads();
        for (int i = tid; i < PF3_HALF_TOKENS * PF3_RPB; i += PF3_THREADS) {
            const int tl = i / PF3_RPB;
            const int rl = i - tl * PF3_RPB;
            const int gt = tok_base + half * PF3_HALF_TOKENS + tl;
            const int gr = row_base + rl;
            if (gt < tokens && gr < m) y[(size_t)gt * m + gr] = sbuf[i];
        }
    }
}
