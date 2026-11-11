// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's K-quant / q8_0 layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// Tiled prefill GEMM over Q5_K weights. Structure identical to prefill_q4_k:
// one block owns 256 tokens x 64 output rows, the weight tile is dequantised
// once into shared and reused for every token, and the f32 accumulators stay in
// registers (8 tokens x 8 rows per thread, 256 threads, 32-value k-tiles).
//
// Q5_K super-block (176 bytes, 256 values): d(2) dmin(2) scales(12) qh(32)
// qs(128). The value is the affine form `y = d*scale*q - dmin*min` where `q` is
// a 5-bit value: the 4-bit `qs` nibble plus the `qh` fifth bit. Both the scale
// bytes (`get_scale_min_k4`) and the `qh` bit order are Q4_K's; Q5_K only adds
// one bit test per value. `dequantize_row_q5_K` (docs/reference/ggml-quants.c,
// lines 1741-1766) is the specification: `qh[l]` covers values l and l+32 of
// each 64-value group, with bit `1 << 2g` for the first and `2 << 2g` for the
// second.
//
// A thread dequantises 8 contiguous values, a quarter of a 32-value sub-block,
// so one d/dmin/scale/min set and one qh bit position cover all 8.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define PF5_BLOCK_BYTES 176
#define PF5_TPB 256
#define PF5_RPB 64
#define PF5_KT 32
#define PF5_TT 8
#define PF5_RR 8
#define PF5_TG (PF5_TPB / PF5_TT)
#define PF5_RG (PF5_RPB / PF5_RR)
#define PF5_THREADS (PF5_TG * PF5_RG)
#define PF5_XSTRIDE (PF5_KT + 1)
#define PF5_WSTRIDE (PF5_KT + 1)
#define PF5_HALF_TOKENS (PF5_TPB / 2)

static __device__ __forceinline__ void pf5_scale_min(int j, const unsigned char* q,
                                                     int* sc, int* mn)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantise 8 contiguous values of one row, starting at value index `v`
// (a multiple of 8) into `out`.
static __device__ __forceinline__ void pf5_dequant8(const unsigned char* __restrict__ rp,
                                                    int v, float* out)
{
    const int blk = v >> 8;
    const int within = v & 255;
    const int sub = within >> 5;   // 32-value sub-block: scale/min index
    const int g = within >> 6;     // 64-value group: qh bit pair
    const int part = (within & 31) >> 3;

    const unsigned char* b = rp + (size_t)blk * PF5_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
    int sc, mn;
    pf5_scale_min(sub, b + 4, &sc, &mn);
    const float scale = d * (float)sc;
    const float minv = dmin * (float)mn;
    const bool high = (sub & 1) != 0;
    const unsigned int bit = high ? (2u << (2 * g)) : (1u << (2 * g));

    // qs holds the 4-bit quants (two per byte, low nibble first); qh holds the
    // fifth bit for values 0..31 of the 64-value group.
    const unsigned char* qp = b + 48 + g * 32 + part * 8;
    const unsigned char* hp = b + 16 + part * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int nib = high ? (qp[i] >> 4) : (qp[i] & 0xF);
        const int q = nib + ((hp[i] & bit) ? 16 : 0);
        out[i] = scale * (float)q - minv;
    }
}

extern "C" __global__ void prefill_q5_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) float pf5_x[PF5_TPB * PF5_XSTRIDE];
    __shared__ __align__(16) float pf5_w[PF5_RPB * PF5_WSTRIDE];

    const int tid = threadIdx.x;
    const int tg = tid % PF5_TG;   // token group: tokens tg + 32*tt
    const int rg = tid / PF5_TG;   // row group: rows rg*8 .. rg*8+7
    const int tok_base = blockIdx.y * PF5_TPB;
    const int row_base = blockIdx.x * PF5_RPB;
    const int nblocks = k / 256;

    const int d_row = tid >> 2;    // 64 rows
    const int d_q = tid & 3;       // 4 chunks of 8 values per 32-value k-tile

    float acc[PF5_TT][PF5_RR];
#pragma unroll
    for (int tt = 0; tt < PF5_TT; ++tt)
#pragma unroll
        for (int rr = 0; rr < PF5_RR; ++rr) acc[tt][rr] = 0.0f;

    const int xq = PF5_TPB * (PF5_KT / 4) / PF5_THREADS; // 8 float4 per thread

    for (int base = 0; base < k; base += PF5_KT) {
        __syncthreads(); // previous multiply finished before shared is reused

        // Weight tile: 64 rows x 32 values, dequantised once.
        {
            const int r = row_base + d_row;
            if (r < m) {
                const unsigned char* rp =
                    w + (size_t)r * (size_t)nblocks * PF5_BLOCK_BYTES;
                float v[8];
                pf5_dequant8(rp, base + d_q * 8, v);
                float* dst = pf5_w + d_row * PF5_WSTRIDE + d_q * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = v[i];
            }
        }

        // x tile: 256 tokens x 32 values, token-major.
        for (int s = 0; s < xq; ++s) {
            const int i = tid + s * PF5_THREADS;
            const int tl = i >> 3;         // token within the block
            const int c4 = (i & 7) * 4;    // k offset, multiple of 4
            const int gt = tok_base + tl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens) {
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + c4);
            }
            float* dst = pf5_x + tl * PF5_XSTRIDE + c4;
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        }

        __syncthreads();

        const int x0 = tg * PF5_XSTRIDE;
        const int w0 = rg * PF5_RR * PF5_WSTRIDE;
#pragma unroll 2
        for (int kk = 0; kk < PF5_KT; ++kk) {
            float xv[PF5_TT];
            float wv[PF5_RR];
#pragma unroll
            for (int tt = 0; tt < PF5_TT; ++tt)
                xv[tt] = pf5_x[x0 + tt * PF5_TG * PF5_XSTRIDE + kk];
#pragma unroll
            for (int rr = 0; rr < PF5_RR; ++rr)
                wv[rr] = pf5_w[w0 + rr * PF5_WSTRIDE + kk];
#pragma unroll
            for (int tt = 0; tt < PF5_TT; ++tt)
#pragma unroll
                for (int rr = 0; rr < PF5_RR; ++rr)
                    acc[tt][rr] = fmaf(xv[tt], wv[rr], acc[tt][rr]);
        }
    }

    // Epilogue in two 128-token halves so the x buffer (33.8 KB) can hold the
    // transpose of the 8 KB output half, which keeps the global stores
    // coalesced along `row` (y is row-major [tokens, m]).
    float* sbuf = pf5_x;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < PF5_TT / 2; ++u)
#pragma unroll
            for (int rr = 0; rr < PF5_RR; ++rr)
                sbuf[(tg + PF5_TG * u) * PF5_RPB + (rg * PF5_RR + rr)] =
                    acc[half * (PF5_TT / 2) + u][rr];
        __syncthreads();
        for (int i = tid; i < PF5_HALF_TOKENS * PF5_RPB; i += PF5_THREADS) {
            const int tl = i / PF5_RPB;
            const int rl = i - tl * PF5_RPB;
            const int gt = tok_base + half * PF5_HALF_TOKENS + tl;
            const int gr = row_base + rl;
            if (gt < tokens && gr < m) y[(size_t)gt * m + gr] = sbuf[i];
        }
    }
}
