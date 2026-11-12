// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's IQ4_XS layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// y[row] = dequant_iq4_xs(W[row,:]) . x
//
// Same row-tile GEMV as `gemv_q4_k`: one block owns IQ4X_ROW_TILE output rows
// and stages the x k-tile in shared once, so a value read from shared feeds
// IQ4X_ROWS_PER_WARP rows before it is discarded. With 4.25-bit weights x is
// ~7.5x the weight bytes, so that reuse is what keeps the loop off the shared
// bandwidth ceiling.
//
// Each lane owns IQ4X_VPL (16) consecutive columns: half of one 32-value
// sub-block, so one d/ls set covers all 16 values. IQ4_XS stores 16 bytes per
// sub-block; the low nibble is the first 16 values and the high nibble the
// second 16, so both halves read the same 16 bytes and the half selects the
// nibble.
//
// IQ4_XS super-block (136 bytes, 256 values):
//   d(2) scales_h(2) scales_l(4) qs(128)
// Per sub-block: ls = low-nibble(scales_l) | 2-bit(scales_h) << 4, and
// dl = d * (ls - 32), which is negative for ls < 32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define IQ4X_VALUES 256
#define IQ4X_BLOCK_BYTES 136
#define IQ4X_BLOCK 256
#define IQ4X_WARPS (IQ4X_BLOCK / 32)
#define IQ4X_ROWS_PER_WARP 2
#define IQ4X_ROW_TILE (IQ4X_WARPS * IQ4X_ROWS_PER_WARP)
#define IQ4X_VPL 16
#define IQ4X_GROUP (IQ4X_VPL * 32)
#define IQ4X_K_TILE 2048

// ggml-common.h `kvalues_iq4nl`; the only codebook IQ4_XS/IQ4_NL use.
static __device__ __constant__ const signed char IQ4NL_VALUES[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// Six-bit sub-block scale as the signed multiplier `ls - 32`.
static __device__ __forceinline__ int iq4x_scale(const unsigned char* b, int sub)
{
    const unsigned short scales_h = *reinterpret_cast<const unsigned short*>(b + 2);
    const int ls = ((b[4 + (sub >> 1)] >> (4 * (sub & 1))) & 0xF) |
                   (((scales_h >> (2 * sub)) & 3) << 4);
    return ls - 32;
}

static __device__ __forceinline__ void iq4x_stage_group(const float* __restrict__ x,
                                                   float* __restrict__ sx,
                                                   int base, int j)
{
    const int g = j >> 5; // 32 chunks of 16 values per 512-value group
    const int lane = j & 31;
    float* s = sx + g * IQ4X_GROUP + lane;
    const float4* src = reinterpret_cast<const float4*>(x + base + IQ4X_VPL * j);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 v = src[q];
        s[(4 * q + 0) * 32] = v.x;
        s[(4 * q + 1) * 32] = v.y;
        s[(4 * q + 2) * 32] = v.z;
        s[(4 * q + 3) * 32] = v.w;
    }
}

// Sixteen-column contribution of one IQ4_XS row to `acc`.
//
// The 16-entry codebook is read from shared memory as floats instead of indexed
// out of the `__constant__` table per value: on this target `__constant__` lives
// in global memory, so a divergent index costs an address add plus a load per
// value. Same table, same values.
static __device__ __forceinline__ void iq4x_acc(const unsigned char* __restrict__ rp,
                                                int p, const float* xv,
                                                const float* __restrict__ scb, float* acc)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int sub = within >> 5;
    const int part = (within & 31) >> 4;

    const unsigned char* b = rp + (size_t)blk * IQ4X_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float dl = d * (float)iq4x_scale(b, sub);

    const unsigned int* qp = reinterpret_cast<const unsigned int*>(b + 8 + sub * 16);
    union IQ4Bytes {
        unsigned int w[4];
        unsigned char b[16];
    } q;
    q.w[0] = qp[0];
    q.w[1] = qp[1];
    q.w[2] = qp[2];
    q.w[3] = qp[3];
    const bool high = part != 0;

    float a = *acc;
#pragma unroll
    for (int i = 0; i < IQ4X_VPL; ++i) {
        const int v = high ? (int)(q.b[i] >> 4) : (int)(q.b[i] & 0xF);
        a = fmaf(dl * scb[v], xv[i], a);
    }
    *acc = a;
}

template <int RPW>
static __device__ __forceinline__ void iq4x_tile(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    float* __restrict__ scb,
    int tile)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / IQ4X_VALUES;
    const int row_tile = IQ4X_WARPS * RPW;

    // The first `__syncthreads()` after the x staging covers this store.
    if (tid < 16) scb[tid] = (float)IQ4NL_VALUES[tid];

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * IQ4X_BLOCK_BYTES;
        acc[rr] = 0.0f;
    }

    for (int base = 0; base < k; base += IQ4X_K_TILE) {
        const int n = min(IQ4X_K_TILE, k - base);
        const int nchunks = n / IQ4X_VPL;
        for (int j = tid; j < nchunks; j += IQ4X_BLOCK) iq4x_stage_group(x, sx, base, j);
        __syncthreads();

        for (int c = lane * IQ4X_VPL; c < n; c += IQ4X_GROUP) {
            const int g = c / IQ4X_GROUP;
            const int s = g * IQ4X_GROUP + lane;
            float xv[IQ4X_VPL];
#pragma unroll
            for (int comp = 0; comp < IQ4X_VPL; ++comp) xv[comp] = sx[s + comp * 32];
            const int p = base + c;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) iq4x_acc(rp[rr], p, xv, scb, &acc[rr]);
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

extern "C" __global__ void gemv_iq4_xs(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[IQ4X_K_TILE];
    __shared__ __align__(16) float scb[16];
    iq4x_tile<IQ4X_ROWS_PER_WARP>(w, x, y, m, k, sx, scb, blockIdx.x);
}

// ---------------------------------------------------------------------------
// Batched GEMV: B token rows share one pass over the weight.
//
//   y[b][row] = dequant_iq4_xs(W[row,:]) . x[b][:]
//
// Same reuse as `gemv_q4_k_b`: a decoded weight value feeds all B x rows, and
// one shared x read feeds IQ4XB_ROWS_PER_WARP weight rows. Each row walks k in
// exactly the order `gemv_iq4_xs` does, so row b is bit-identical to a
// single-sequence launch on x[b].
// ---------------------------------------------------------------------------

#define IQ4XB_K_TILE_BIG 1024
#define IQ4XB_K_TILE_SMALL 512
#define IQ4XB_ROWS_PER_WARP 2

// Dequantise this lane's 16 values of one row into `wv`.
static __device__ __forceinline__ void iq4x_load(const unsigned char* __restrict__ rp,
                                                 int p, float* wv)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int sub = within >> 5;
    const int part = (within & 31) >> 4;

    const unsigned char* b = rp + (size_t)blk * IQ4X_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float dl = d * (float)iq4x_scale(b, sub);

    const unsigned int* qp = reinterpret_cast<const unsigned int*>(b + 8 + sub * 16);
    union IQ4BytesB {
        unsigned int w[4];
        unsigned char b[16];
    } q;
    q.w[0] = qp[0];
    q.w[1] = qp[1];
    q.w[2] = qp[2];
    q.w[3] = qp[3];
    const bool high = part != 0;
#pragma unroll
    for (int i = 0; i < IQ4X_VPL; ++i) {
        const int v = high ? (int)(q.b[i] >> 4) : (int)(q.b[i] & 0xF);
        wv[i] = dl * (float)IQ4NL_VALUES[v];
    }
}

template <int B, int RPW, int TILE>
static __device__ __forceinline__ void iq4x_tile_b(
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
    const int nblocks = k / IQ4X_VALUES;
    const int row_tile = IQ4X_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW][B];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * IQ4X_BLOCK_BYTES;
#pragma unroll
        for (int bi = 0; bi < B; ++bi) acc[rr][bi] = 0.0f;
    }

    for (int base = 0; base < k; base += TILE) {
        const int n = min(TILE, k - base);
        const int nchunks = n / IQ4X_VPL;
        for (int j = tid; j < nchunks; j += IQ4X_BLOCK) {
#pragma unroll
            for (int bi = 0; bi < B; ++bi)
                iq4x_stage_group(x + (size_t)bi * k, sx + bi * TILE, base, j);
        }
        __syncthreads();

        for (int c = lane * IQ4X_VPL; c < n; c += IQ4X_GROUP) {
            const int g = c / IQ4X_GROUP;
            const int s = g * IQ4X_GROUP + lane;
            const int p = base + c;
            float wv[RPW][IQ4X_VPL];
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) iq4x_load(rp[rr], p, wv[rr]);
#pragma unroll
            for (int bi = 0; bi < B; ++bi) {
                const float* src = sx + bi * TILE + s;
                float xv[IQ4X_VPL];
#pragma unroll
                for (int i = 0; i < IQ4X_VPL; ++i) xv[i] = src[i * 32];
#pragma unroll
                for (int rr = 0; rr < RPW; ++rr) {
                    float a = acc[rr][bi];
#pragma unroll
                    for (int i = 0; i < IQ4X_VPL; ++i) a = fmaf(wv[rr][i], xv[i], a);
                    acc[rr][bi] = a;
                }
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
static __device__ __forceinline__ void iq4x_entry_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    extern __shared__ __align__(16) float hipml_sx_b[];
    iq4x_tile_b<B, IQ4XB_ROWS_PER_WARP, TILE>(w, x, y, m, k, hipml_sx_b, blockIdx.x);
}

extern "C" __global__ void gemv_iq4_xs_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int batch)
{
    if (batch <= 4) {
        switch (batch) {
        case 1: iq4x_entry_b<1, IQ4XB_K_TILE_BIG>(w, x, y, m, k); break;
        case 2: iq4x_entry_b<2, IQ4XB_K_TILE_BIG>(w, x, y, m, k); break;
        case 3: iq4x_entry_b<3, IQ4XB_K_TILE_BIG>(w, x, y, m, k); break;
        case 4: iq4x_entry_b<4, IQ4XB_K_TILE_BIG>(w, x, y, m, k); break;
        }
    } else {
        switch (batch) {
        case 5: iq4x_entry_b<5, IQ4XB_K_TILE_SMALL>(w, x, y, m, k); break;
        case 6: iq4x_entry_b<6, IQ4XB_K_TILE_SMALL>(w, x, y, m, k); break;
        case 7: iq4x_entry_b<7, IQ4XB_K_TILE_SMALL>(w, x, y, m, k); break;
        default: iq4x_entry_b<8, IQ4XB_K_TILE_SMALL>(w, x, y, m, k); break;
        }
    }
}
