// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's IQ4_XS layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// Tiled prefill GEMM over IQ4_XS weights.
//
// y[t, row] = W[row, :] . x[t, :] for `tokens` rows of x at once. Same geometry
// and shared staging as `prefill_q4_k`: a weight tile is dequantised ONCE into
// shared and reused for every prompt token in the block.
//
//   PFI4_TPB 256 tokens per block, PFI4_RPB 64 output rows, PFI4_KT 32 k/tile.
//   256 threads: 32 token groups x 8 row groups, 8x8 accumulators per thread.
//
// Each thread dequantises 8 contiguous values, a quarter of a 32-value
// sub-block: the low nibbles are values 0..15 and the high nibbles 16..31, so
// part 0/1 read bytes 0..7 / 8..15 low and part 2/3 the same bytes high.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define PFI4_BLOCK_BYTES 136
#define PFI4_TPB 256
#define PFI4_RPB 64
#define PFI4_KT 32
#define PFI4_TT 8
#define PFI4_RR 8
#define PFI4_TG (PFI4_TPB / PFI4_TT)
#define PFI4_RG (PFI4_RPB / PFI4_RR)
#define PFI4_THREADS (PFI4_TG * PFI4_RG)
#define PFI4_XSTRIDE (PFI4_KT + 1)
#define PFI4_WSTRIDE (PFI4_KT + 1)
#define PFI4_HALF_TOKENS (PFI4_TPB / 2)

static __device__ __constant__ const signed char PFI4_KV[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// Dequantise 8 contiguous values of one row, starting at value index `v`
// (a multiple of 8) into `out`.
static __device__ __forceinline__ void pfi4_dequant8(const unsigned char* __restrict__ rp,
                                                     int v, float* out)
{
    const int blk = v >> 8;
    const int within = v & 255;
    const int sub = within >> 5;
    const int part = (within & 31) >> 3;

    const unsigned char* b = rp + (size_t)blk * PFI4_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const unsigned short scales_h = *reinterpret_cast<const unsigned short*>(b + 2);
    const int ls = ((b[4 + (sub >> 1)] >> (4 * (sub & 1))) & 0xF) |
                   (((scales_h >> (2 * sub)) & 3) << 4);
    const float dl = d * (float)(ls - 32);
    const bool high = part >= 2;
    const unsigned char* qp = b + 8 + sub * 16 + (part & 1) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = high ? (qp[i] >> 4) : (qp[i] & 0xF);
        out[i] = dl * (float)PFI4_KV[q];
    }
}

extern "C" __global__ void prefill_iq4_xs(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) float pfi4_x[PFI4_TPB * PFI4_XSTRIDE];
    __shared__ __align__(16) float pfi4_w[PFI4_RPB * PFI4_WSTRIDE];

    const int tid = threadIdx.x;
    const int tg = tid % PFI4_TG;   // token group: tokens tg + 32*tt
    const int rg = tid / PFI4_TG;   // row group: rows rg*8 .. rg*8+7
    const int tok_base = blockIdx.y * PFI4_TPB;
    const int row_base = blockIdx.x * PFI4_RPB;
    const int nblocks = k / 256;

    const int d_row = tid >> 2;    // 64 rows
    const int d_q = tid & 3;       // 4 chunks of 8 values per 32-value k-tile

    float acc[PFI4_TT][PFI4_RR];
#pragma unroll
    for (int tt = 0; tt < PFI4_TT; ++tt)
#pragma unroll
        for (int rr = 0; rr < PFI4_RR; ++rr) acc[tt][rr] = 0.0f;

    const int xq = PFI4_TPB * (PFI4_KT / 4) / PFI4_THREADS; // 8 float4 per thread

    for (int base = 0; base < k; base += PFI4_KT) {
        __syncthreads(); // previous multiply finished before shared is reused

        // Weight tile: 64 rows x 32 values, dequantised once.
        {
            const int r = row_base + d_row;
            if (r < m) {
                const unsigned char* rp =
                    w + (size_t)r * (size_t)nblocks * PFI4_BLOCK_BYTES;
                float v[8];
                pfi4_dequant8(rp, base + d_q * 8, v);
                float* dst = pfi4_w + d_row * PFI4_WSTRIDE + d_q * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = v[i];
            }
        }

        // x tile: 256 tokens x 32 values, token-major.
        for (int s = 0; s < xq; ++s) {
            const int i = tid + s * PFI4_THREADS;
            const int tl = i >> 3;         // token within the block
            const int c4 = (i & 7) * 4;    // k offset, multiple of 4
            const int gt = tok_base + tl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens) {
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + c4);
            }
            float* dst = pfi4_x + tl * PFI4_XSTRIDE + c4;
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        }

        __syncthreads();

        const int x0 = tg * PFI4_XSTRIDE;
        const int w0 = rg * PFI4_RR * PFI4_WSTRIDE;
#pragma unroll 2
        for (int kk = 0; kk < PFI4_KT; ++kk) {
            float xv[PFI4_TT];
            float wv[PFI4_RR];
#pragma unroll
            for (int tt = 0; tt < PFI4_TT; ++tt)
                xv[tt] = pfi4_x[x0 + tt * PFI4_TG * PFI4_XSTRIDE + kk];
#pragma unroll
            for (int rr = 0; rr < PFI4_RR; ++rr)
                wv[rr] = pfi4_w[w0 + rr * PFI4_WSTRIDE + kk];
#pragma unroll
            for (int tt = 0; tt < PFI4_TT; ++tt)
#pragma unroll
                for (int rr = 0; rr < PFI4_RR; ++rr)
                    acc[tt][rr] = fmaf(xv[tt], wv[rr], acc[tt][rr]);
        }
    }

    // Epilogue in two 128-token halves so the x buffer can hold the transpose
    // of the 8 KB output half; global stores are then coalesced along `row`.
    float* sbuf = pfi4_x;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < PFI4_TT / 2; ++u)
#pragma unroll
            for (int rr = 0; rr < PFI4_RR; ++rr)
                sbuf[(tg + PFI4_TG * u) * PFI4_RPB + (rg * PFI4_RR + rr)] =
                    acc[half * (PFI4_TT / 2) + u][rr];
        __syncthreads();
        for (int i = tid; i < PFI4_HALF_TOKENS * PFI4_RPB; i += PFI4_THREADS) {
            const int tl = i / PFI4_RPB;
            const int rl = i - tl * PFI4_RPB;
            const int gt = tok_base + half * PFI4_HALF_TOKENS + tl;
            const int gr = row_base + rl;
            if (gt < tokens && gr < m) y[(size_t)gt * m + gr] = sbuf[i];
        }
    }
}
