// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's IQ4_NL layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// y[row] = dequant_iq4_nl(W[row,:]) . x
//
// Same row-tile GEMV as `gemv_q4_k`, tiled at 32 values per block like Q8_0.
// Each lane owns IQ4N_VPL (16) consecutive columns, i.e. half of one block:
// the low nibbles are the first 16 values and the high nibbles the second 16,
// so both halves read the same 16 bytes and the half selects the nibble.
//
// IQ4_NL block (18 bytes, 32 values): d(2) qs[16]. Codebook is the same
// 16-entry `kvalues_iq4nl` as IQ4_XS.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define IQ4N_VALUES 32
#define IQ4N_BLOCK_BYTES 18
#define IQ4N_BLOCK 256
#define IQ4N_WARPS (IQ4N_BLOCK / 32)
#define IQ4N_ROWS_PER_WARP 2
#define IQ4N_ROW_TILE (IQ4N_WARPS * IQ4N_ROWS_PER_WARP)
#define IQ4N_VPL 16
#define IQ4N_GROUP (IQ4N_VPL * 32)
#define IQ4N_K_TILE 2048

// ggml-common.h `kvalues_iq4nl`.
static __device__ __constant__ const signed char IQ4N_KV[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static __device__ __forceinline__ void iq4n_stage_group(const float* __restrict__ x,
                                                   float* __restrict__ sx,
                                                   int base, int j)
{
    const int g = j >> 5;
    const int lane = j & 31;
    float* s = sx + g * IQ4N_GROUP + lane;
    const float4* src = reinterpret_cast<const float4*>(x + base + IQ4N_VPL * j);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 v = src[q];
        s[(4 * q + 0) * 32] = v.x;
        s[(4 * q + 1) * 32] = v.y;
        s[(4 * q + 2) * 32] = v.z;
        s[(4 * q + 3) * 32] = v.w;
    }
}

// IQ4_NL blocks are 18 bytes, so the payload at b+2 is only 2-byte aligned.
// Four dword loads instead of sixteen byte loads: same bytes, and a global load
// only needs 4-byte alignment on this target.
static __device__ __forceinline__ void iq4n_read16(const unsigned char* src,
                                                   unsigned int* dst)
{
#pragma unroll
    for (int i = 0; i < 4; ++i) __builtin_memcpy(dst + i, src + 4 * i, 4);
}

// The 16-entry codebook is read from shared memory as floats instead of indexed
// out of the `__constant__` table per value: on this target `__constant__` lives
// in global memory, so a divergent index costs an address add plus a load per
// value. Same table, same values.
static __device__ __forceinline__ void iq4n_acc(const unsigned char* __restrict__ rp,
                                                int p, const float* xv,
                                                const float* __restrict__ scb, float* acc)
{
    const int blk = p >> 5;
    const int part = (p & 31) >> 4;

    const unsigned char* b = rp + (size_t)blk * IQ4N_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    union IQ4NBytes {
        unsigned int w[4];
        unsigned char b[16];
    } q;
    iq4n_read16(b + 2, q.w);
    const bool high = part != 0;

    float a = *acc;
#pragma unroll
    for (int i = 0; i < IQ4N_VPL; ++i) {
        const int v = high ? (int)(q.b[i] >> 4) : (int)(q.b[i] & 0xF);
        a = fmaf(d * scb[v], xv[i], a);
    }
    *acc = a;
}

template <int RPW>
static __device__ __forceinline__ void iq4n_tile(
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
    const int nblocks = k / IQ4N_VALUES;
    const int row_tile = IQ4N_WARPS * RPW;

    // The first `__syncthreads()` after the x staging covers this store.
    if (tid < 16) scb[tid] = (float)IQ4N_KV[tid];

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * IQ4N_BLOCK_BYTES;
        acc[rr] = 0.0f;
    }

    for (int base = 0; base < k; base += IQ4N_K_TILE) {
        const int n = min(IQ4N_K_TILE, k - base);
        const int nchunks = n / IQ4N_VPL;
        for (int j = tid; j < nchunks; j += IQ4N_BLOCK) iq4n_stage_group(x, sx, base, j);
        __syncthreads();

        for (int c = lane * IQ4N_VPL; c < n; c += IQ4N_GROUP) {
            const int g = c / IQ4N_GROUP;
            const int s = g * IQ4N_GROUP + lane;
            float xv[IQ4N_VPL];
#pragma unroll
            for (int comp = 0; comp < IQ4N_VPL; ++comp) xv[comp] = sx[s + comp * 32];
            const int p = base + c;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) iq4n_acc(rp[rr], p, xv, scb, &acc[rr]);
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

extern "C" __global__ void gemv_iq4_nl(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[IQ4N_K_TILE];
    __shared__ __align__(16) float scb[16];
    iq4n_tile<IQ4N_ROWS_PER_WARP>(w, x, y, m, k, sx, scb, blockIdx.x);
}

// ---------------------------------------------------------------------------
// Batched GEMV: B token rows share one pass over the weight.
// ---------------------------------------------------------------------------

#define IQ4NB_K_TILE_BIG 1024
#define IQ4NB_K_TILE_SMALL 512
#define IQ4NB_ROWS_PER_WARP 2

static __device__ __forceinline__ void iq4n_load(const unsigned char* __restrict__ rp,
                                                 int p, float* wv)
{
    const int blk = p >> 5;
    const int part = (p & 31) >> 4;
    const unsigned char* b = rp + (size_t)blk * IQ4N_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    union IQ4NBytesB {
        unsigned int w[4];
        unsigned char b[16];
    } q;
    iq4n_read16(b + 2, q.w);
    const bool high = part != 0;
#pragma unroll
    for (int i = 0; i < IQ4N_VPL; ++i) {
        const int v = high ? (int)(q.b[i] >> 4) : (int)(q.b[i] & 0xF);
        wv[i] = d * (float)IQ4N_KV[v];
    }
}

template <int B, int RPW, int TILE>
static __device__ __forceinline__ void iq4n_tile_b(
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
    const int nblocks = k / IQ4N_VALUES;
    const int row_tile = IQ4N_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW][B];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * IQ4N_BLOCK_BYTES;
#pragma unroll
        for (int bi = 0; bi < B; ++bi) acc[rr][bi] = 0.0f;
    }

    for (int base = 0; base < k; base += TILE) {
        const int n = min(TILE, k - base);
        const int nchunks = n / IQ4N_VPL;
        for (int j = tid; j < nchunks; j += IQ4N_BLOCK) {
#pragma unroll
            for (int bi = 0; bi < B; ++bi)
                iq4n_stage_group(x + (size_t)bi * k, sx + bi * TILE, base, j);
        }
        __syncthreads();

        for (int c = lane * IQ4N_VPL; c < n; c += IQ4N_GROUP) {
            const int g = c / IQ4N_GROUP;
            const int s = g * IQ4N_GROUP + lane;
            const int p = base + c;
            float wv[RPW][IQ4N_VPL];
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) iq4n_load(rp[rr], p, wv[rr]);
#pragma unroll
            for (int bi = 0; bi < B; ++bi) {
                const float* src = sx + bi * TILE + s;
                float xv[IQ4N_VPL];
#pragma unroll
                for (int i = 0; i < IQ4N_VPL; ++i) xv[i] = src[i * 32];
#pragma unroll
                for (int rr = 0; rr < RPW; ++rr) {
                    float a = acc[rr][bi];
#pragma unroll
                    for (int i = 0; i < IQ4N_VPL; ++i) a = fmaf(wv[rr][i], xv[i], a);
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
static __device__ __forceinline__ void iq4n_entry_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    extern __shared__ __align__(16) float hipml_sx_b[];
    iq4n_tile_b<B, IQ4NB_ROWS_PER_WARP, TILE>(w, x, y, m, k, hipml_sx_b, blockIdx.x);
}

extern "C" __global__ void gemv_iq4_nl_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int batch)
{
    if (batch <= 4) {
        switch (batch) {
        case 1: iq4n_entry_b<1, IQ4NB_K_TILE_BIG>(w, x, y, m, k); break;
        case 2: iq4n_entry_b<2, IQ4NB_K_TILE_BIG>(w, x, y, m, k); break;
        case 3: iq4n_entry_b<3, IQ4NB_K_TILE_BIG>(w, x, y, m, k); break;
        case 4: iq4n_entry_b<4, IQ4NB_K_TILE_BIG>(w, x, y, m, k); break;
        }
    } else {
        switch (batch) {
        case 5: iq4n_entry_b<5, IQ4NB_K_TILE_SMALL>(w, x, y, m, k); break;
        case 6: iq4n_entry_b<6, IQ4NB_K_TILE_SMALL>(w, x, y, m, k); break;
        case 7: iq4n_entry_b<7, IQ4NB_K_TILE_SMALL>(w, x, y, m, k); break;
        default: iq4n_entry_b<8, IQ4NB_K_TILE_SMALL>(w, x, y, m, k); break;
        }
    }
}
