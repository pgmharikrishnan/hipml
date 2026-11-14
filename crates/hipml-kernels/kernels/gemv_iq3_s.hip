// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's IQ3_S layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// y[row] = dequant_iq3_s(W[row,:]) . x
//
// Same row-tile GEMV as `gemv_q4_k`: one block owns IQ3S_ROW_TILE output rows
// and stages the x k-tile in shared once, so a value read from shared feeds
// IQ3S_ROWS_PER_WARP weight rows before it is discarded.
//
// Each lane owns IQ3S_VPL (16) consecutive columns, i.e. half of one 32-value
// sub-block: two 8-value grid groups, one shared scale nibble. The 9-bit grid
// index is the low 8 bits from `qs` plus one `qh` bit, and `signs` carries one
// sign bit per value.
//
// IQ3_S super-block (110 bytes, 256 values):
//   d(2) qs(64) qh(8) signs(32) scales(4)
// Eight 32-value sub-blocks in four (low nibble, high nibble) scale pairs;
// db = d * (1 + 2*nibble), so every scale is positive.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define IQ3S_VALUES 256
#define IQ3S_BLOCK_BYTES 110
#define IQ3S_BLOCK 256
#define IQ3S_WARPS (IQ3S_BLOCK / 32)
#define IQ3S_ROWS_PER_WARP 2
#define IQ3S_ROW_TILE (IQ3S_WARPS * IQ3S_ROWS_PER_WARP)
#define IQ3S_VPL 16
#define IQ3S_GROUP (IQ3S_VPL * 32)
#define IQ3S_K_TILE 2048

// Mirrors the manifest's KernelKind::RowTile { rows = IQ3S_ROW_TILE, block =
// IQ3S_BLOCK }; change both together.

// ggml-common.h `iq3s_grid`, four packed codebook values per index.
#include "iq3s_grid.inc"

// One scale nibble pair as `1 + 2*nibble` (never negative).
static __device__ __forceinline__ float iq3s_scale(const unsigned char* __restrict__ b,
                                                   int pair, int half)
{
    const unsigned int sc = b[106 + pair];
    const int nib = half ? (int)(sc >> 4) : (int)(sc & 0xF);
    return (float)(1 + 2 * nib);
}

static __device__ __forceinline__ void iq3s_stage_group(const float* __restrict__ x,
                                                        float* __restrict__ sx,
                                                        int base, int j)
{
    const int g = j >> 5; // 32 chunks of 16 values per 512-value group
    const int lane = j & 31;
    float* s = sx + g * IQ3S_GROUP + lane;
    const float4* src = reinterpret_cast<const float4*>(x + base + IQ3S_VPL * j);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 v = src[q];
        s[(4 * q + 0) * 32] = v.x;
        s[(4 * q + 1) * 32] = v.y;
        s[(4 * q + 2) * 32] = v.z;
        s[(4 * q + 3) * 32] = v.w;
    }
}

// Dequantise this lane's 16 values of one row into `wv`.
static __device__ __forceinline__ void iq3s_load(const unsigned char* __restrict__ rp,
                                                 int p, float* wv)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int pair = within >> 6;
    const int half = (within >> 5) & 1;
    const int l0 = (within & 31) >> 3;

    const unsigned char* b = rp + (size_t)blk * IQ3S_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float db = d * iq3s_scale(b, pair, half);
    const unsigned int qh = b[66 + pair * 2 + half];
    const unsigned char* qp = b + 2 + pair * 16 + half * 8;
    const unsigned char* sp = b + 74 + pair * 8 + half * 4;

#pragma unroll
    for (int ll = 0; ll < 2; ++ll) {
        const int l = l0 + ll;
        const int i1 = (int)qp[2 * l] | (int)((qh << (8 - 2 * l)) & 256u);
        const int i2 = (int)qp[2 * l + 1] | (int)((qh << (7 - 2 * l)) & 256u);
        const unsigned int g1 = IQ3S_GRID[i1];
        const unsigned int g2 = IQ3S_GRID[i2];
        const unsigned int sign = sp[l];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float s1 = (sign & (1u << t)) ? -1.0f : 1.0f;
            const float s2 = (sign & (1u << (t + 4))) ? -1.0f : 1.0f;
            wv[ll * 8 + t] = db * (float)((g1 >> (8 * t)) & 0xFFu) * s1;
            wv[ll * 8 + 4 + t] = db * (float)((g2 >> (8 * t)) & 0xFFu) * s2;
        }
    }
}

// Sixteen-value dot in the order the single-row kernel accumulates: the two
// grid groups of one `l` interleave (g1[0], g2[0], g1[1], ...), not 0..15.
// Both the single-row and the batched kernel go through this, so row b of the
// batched result is the same 32 bits as a one-column decode.
static __device__ __forceinline__ float iq3s_dot16(const float* wv, const float* xv, float a)
{
#pragma unroll
    for (int ll = 0; ll < 2; ++ll)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            a = fmaf(wv[ll * 8 + t], xv[ll * 8 + t], a);
            a = fmaf(wv[ll * 8 + 4 + t], xv[ll * 8 + 4 + t], a);
        }
    return a;
}

// Sixteen-column contribution of one IQ3_S row to `acc`.
static __device__ __forceinline__ void iq3s_acc(const unsigned char* __restrict__ rp,
                                                int p, const float* xv, float* acc)
{
    float wv[IQ3S_VPL];
    iq3s_load(rp, p, wv);
    *acc = iq3s_dot16(wv, xv, *acc);
}

// One row/warp when RPW is 1, two when it is IQ3S_ROWS_PER_WARP.
template <int RPW>
static __device__ __forceinline__ void iq3s_tile(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / IQ3S_VALUES;
    const int row_tile = IQ3S_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * IQ3S_BLOCK_BYTES;
        acc[rr] = 0.0f;
    }

    for (int base = 0; base < k; base += IQ3S_K_TILE) {
        const int n = min(IQ3S_K_TILE, k - base);
        const int nchunks = n / IQ3S_VPL;
        for (int j = tid; j < nchunks; j += IQ3S_BLOCK) iq3s_stage_group(x, sx, base, j);
        __syncthreads();

        for (int c = lane * IQ3S_VPL; c < n; c += IQ3S_GROUP) {
            const int g = c / IQ3S_GROUP;
            const int s = g * IQ3S_GROUP + lane;
            float xv[IQ3S_VPL];
#pragma unroll
            for (int comp = 0; comp < IQ3S_VPL; ++comp) xv[comp] = sx[s + comp * 32];
            const int p = base + c;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) iq3s_acc(rp[rr], p, xv, &acc[rr]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        float a = acc[rr];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
        if (lane == 0 && row[rr] < m) y[row[rr]] = a;
    }
}

extern "C" __global__ void gemv_iq3_s(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[IQ3S_K_TILE];
    iq3s_tile<IQ3S_ROWS_PER_WARP>(w, x, y, m, k, sx, blockIdx.x);
}

// ---------------------------------------------------------------------------
// Batched GEMV: B token rows share one pass over the weight. Same reuse and
// same per-lane k order as `gemv_iq3_s`, so row b is bit-identical to a
// single-sequence launch on x[b].
// ---------------------------------------------------------------------------

#define IQ3SB_K_TILE_BIG 1024
#define IQ3SB_K_TILE_SMALL 512
#define IQ3SB_ROWS_PER_WARP 2

template <int B, int RPW, int TILE>
static __device__ __forceinline__ void iq3s_tile_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / IQ3S_VALUES;
    const int row_tile = IQ3S_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW][B];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * IQ3S_BLOCK_BYTES;
#pragma unroll
        for (int bi = 0; bi < B; ++bi) acc[rr][bi] = 0.0f;
    }

    for (int base = 0; base < k; base += TILE) {
        const int n = min(TILE, k - base);
        const int nchunks = n / IQ3S_VPL;
        for (int j = tid; j < nchunks; j += IQ3S_BLOCK) {
#pragma unroll
            for (int bi = 0; bi < B; ++bi)
                iq3s_stage_group(x + (size_t)bi * k, sx + bi * TILE, base, j);
        }
        __syncthreads();

        for (int c = lane * IQ3S_VPL; c < n; c += IQ3S_GROUP) {
            const int g = c / IQ3S_GROUP;
            const int s = g * IQ3S_GROUP + lane;
            const int p = base + c;
            float wv[RPW][IQ3S_VPL];
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) iq3s_load(rp[rr], p, wv[rr]);
#pragma unroll
            for (int bi = 0; bi < B; ++bi) {
                const float* src = sx + bi * TILE + s;
                float xv[IQ3S_VPL];
#pragma unroll
                for (int i = 0; i < IQ3S_VPL; ++i) xv[i] = src[i * 32];
#pragma unroll
                for (int rr = 0; rr < RPW; ++rr)
                    acc[rr][bi] = iq3s_dot16(wv[rr], xv, acc[rr][bi]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
#pragma unroll
        for (int bi = 0; bi < B; ++bi) {
            float a = acc[rr][bi];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
            if (lane == 0 && row[rr] < m) y[(size_t)bi * m + row[rr]] = a;
        }
    }
}

template <int B, int TILE>
static __device__ __forceinline__ void iq3s_entry_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    extern __shared__ __align__(16) float hipml_sx_b[];
    iq3s_tile_b<B, IQ3SB_ROWS_PER_WARP, TILE>(w, x, y, m, k, hipml_sx_b, blockIdx.x);
}

extern "C" __global__ void gemv_iq3_s_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int batch)
{
    if (batch <= 4) {
        switch (batch) {
        case 1: iq3s_entry_b<1, IQ3SB_K_TILE_BIG>(w, x, y, m, k); break;
        case 2: iq3s_entry_b<2, IQ3SB_K_TILE_BIG>(w, x, y, m, k); break;
        case 3: iq3s_entry_b<3, IQ3SB_K_TILE_BIG>(w, x, y, m, k); break;
        case 4: iq3s_entry_b<4, IQ3SB_K_TILE_BIG>(w, x, y, m, k); break;
        }
    } else {
        switch (batch) {
        case 5: iq3s_entry_b<5, IQ3SB_K_TILE_SMALL>(w, x, y, m, k); break;
        case 6: iq3s_entry_b<6, IQ3SB_K_TILE_SMALL>(w, x, y, m, k); break;
        case 7: iq3s_entry_b<7, IQ3SB_K_TILE_SMALL>(w, x, y, m, k); break;
        default: iq3s_entry_b<8, IQ3SB_K_TILE_SMALL>(w, x, y, m, k); break;
        }
    }
}
