// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's Q3_K layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// y[row] = dequant_q3_k(W[row,:]) . x
//
// Same row-tile GEMV as `gemv_q4_k`: one block owns Q3K_ROW_TILE output rows
// and stages the x k-tile in shared once, so a value read from shared feeds
// Q3K_ROWS_PER_WARP weight rows before it is discarded.
//
// Each lane owns Q3K_VPL (16) consecutive columns, i.e. half of one 32-value
// sub-block. That half shares one `n`/`j`/`half` triple, so one six-bit scale,
// one 2-bit shift and one hmask bit cover all 16 values. The 16 quant bytes and
// the 16 hmask bytes are four dword memcpys each (a Q3_K super-block is 110
// bytes, so the block base is only two-byte aligned; `__builtin_memcpy` is the
// unaligned-safe load, as in `prefill_wmma2.hip`).
//
// Q3_K super-block (110 bytes, 256 values):
//   hmask(32) qs(64) scales(12) d(2)
// Value = d*(scale[j]-32) * (q2 - 4*(1 - hmask_bit)), q2 in 0..3.
//
// Two barriers per k-tile; the only reduction is a warp shuffle tree per row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q3K_VALUES 256
#define Q3K_BLOCK_BYTES 110
#define Q3K_BLOCK 256
#define Q3K_WARPS (Q3K_BLOCK / 32)
#define Q3K_ROWS_PER_WARP 2
#define Q3K_ROW_TILE (Q3K_WARPS * Q3K_ROWS_PER_WARP)
#define Q3K_VPL 16
#define Q3K_GROUP (Q3K_VPL * 32)
#define Q3K_K_TILE 2048

// Mirrors the manifest's KernelKind::RowTile { rows = Q3K_ROW_TILE, block = Q3K_BLOCK };
// change both together.

// The unsigned dword at `p` (any two-byte alignment): the Q3_K block stride is
// 110, so a block's fields are only two-byte aligned.
static __device__ __forceinline__ unsigned int q3k_ld32(const unsigned char* p)
{
    unsigned int v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

// `scales[is] - 32` from the 12 packed bytes at b+96, exactly the shuffle
// `dequantize_row_q3_K` does (four u32 words, KMASK1/KMASK2).
static __device__ __forceinline__ int q3k_scale(const unsigned char* __restrict__ b, int is)
{
    const unsigned char* s = b + 96;
    const unsigned int KM1 = 0x03030303u;
    const unsigned int KM2 = 0x0f0f0f0fu;
    const int ai = is >> 2;
    unsigned int a;
    if (ai == 0) {
        a = (q3k_ld32(s) & KM2) | ((q3k_ld32(s + 8) & KM1) << 4);
    } else if (ai == 1) {
        a = (q3k_ld32(s + 4) & KM2) | (((q3k_ld32(s + 8) >> 2) & KM1) << 4);
    } else if (ai == 2) {
        a = ((q3k_ld32(s) >> 4) & KM2) | (((q3k_ld32(s + 8) >> 4) & KM1) << 4);
    } else {
        a = ((q3k_ld32(s + 4) >> 4) & KM2) | (((q3k_ld32(s + 8) >> 6) & KM1) << 4);
    }
    return (int)(signed char)((a >> (8 * (is & 3))) & 0xFFu) - 32;
}

static __device__ __forceinline__ void q3k_stage_group(const float* __restrict__ x,
                                                       float* __restrict__ sx,
                                                       int base, int j)
{
    const int g = j >> 5; // 32 chunks of 16 values per 512-value group
    const int lane = j & 31;
    float* s = sx + g * Q3K_GROUP + lane;
    const float4* src = reinterpret_cast<const float4*>(x + base + Q3K_VPL * j);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 v = src[q];
        s[(4 * q + 0) * 32] = v.x;
        s[(4 * q + 1) * 32] = v.y;
        s[(4 * q + 2) * 32] = v.z;
        s[(4 * q + 3) * 32] = v.w;
    }
}

// Sixteen-column contribution of one Q3_K row to `acc`.
static __device__ __forceinline__ void q3k_acc(const unsigned char* __restrict__ rp,
                                               int p, const float* xv, float* acc)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int n = within >> 7;
    const int j = (within >> 5) & 3;
    const int half = (within >> 4) & 1;

    const unsigned char* b = rp + (size_t)blk * Q3K_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b + 108));
    const float scale = d * (float)q3k_scale(b, n * 8 + j * 2 + half);

    const unsigned char* qp = b + 32 + n * 32 + half * 16;
    const unsigned char* hp = b + half * 16;
    const int shift = 2 * j;
    const int bit = 1 << (n * 4 + j);

    float a = *acc;
#pragma unroll
    for (int i = 0; i < Q3K_VPL; i += 4) {
        const unsigned int qw = q3k_ld32(qp + i);
        const unsigned int hw = q3k_ld32(hp + i);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int qv = (int)((qw >> (8 * t + shift)) & 3u);
            const int sub = ((hw >> (8 * t)) & (unsigned)bit) ? 0 : 4;
            a = fmaf(scale * (float)(qv - sub), xv[i + t], a);
        }
    }
    *acc = a;
}

// One row/warp when RPW is 1, two when it is Q3K_ROWS_PER_WARP.
template <int RPW>
static __device__ __forceinline__ void q3k_tile(
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
    const int nblocks = k / Q3K_VALUES;
    const int row_tile = Q3K_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * Q3K_BLOCK_BYTES;
        acc[rr] = 0.0f;
    }

    for (int base = 0; base < k; base += Q3K_K_TILE) {
        const int n = min(Q3K_K_TILE, k - base);
        const int nchunks = n / Q3K_VPL;
        for (int j = tid; j < nchunks; j += Q3K_BLOCK) q3k_stage_group(x, sx, base, j);
        __syncthreads();

        for (int c = lane * Q3K_VPL; c < n; c += Q3K_GROUP) {
            const int g = c / Q3K_GROUP;
            const int s = g * Q3K_GROUP + lane;
            float xv[Q3K_VPL];
#pragma unroll
            for (int comp = 0; comp < Q3K_VPL; ++comp) xv[comp] = sx[s + comp * 32];
            const int p = base + c;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) q3k_acc(rp[rr], p, xv, &acc[rr]);
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

extern "C" __global__ void gemv_q3_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[Q3K_K_TILE];
    q3k_tile<Q3K_ROWS_PER_WARP>(w, x, y, m, k, sx, blockIdx.x);
}

extern "C" __global__ void gemv_q3_k_r1(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[Q3K_K_TILE];
    q3k_tile<1>(w, x, y, m, k, sx, blockIdx.x);
}

// ---------------------------------------------------------------------------
// Batched GEMV: B token rows share one pass over the weight. Same reuse as
// `gemv_q4_k_b`; each row keeps its own accumulator and walks k in exactly the
// order `gemv_q3_k` does, so row b is bit-identical to a single-sequence launch
// on x[b].
// ---------------------------------------------------------------------------

#define Q3KB_K_TILE_BIG 1024
#define Q3KB_K_TILE_SMALL 512
#define Q3KB_ROWS_PER_WARP 2

// Dequantise this lane's 16 values of one row into `wv`.
static __device__ __forceinline__ void q3k_load(const unsigned char* __restrict__ rp,
                                                int p, float* wv)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int n = within >> 7;
    const int j = (within >> 5) & 3;
    const int half = (within >> 4) & 1;

    const unsigned char* b = rp + (size_t)blk * Q3K_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b + 108));
    const float scale = d * (float)q3k_scale(b, n * 8 + j * 2 + half);

    const unsigned char* qp = b + 32 + n * 32 + half * 16;
    const unsigned char* hp = b + half * 16;
    const int shift = 2 * j;
    const int bit = 1 << (n * 4 + j);

#pragma unroll
    for (int i = 0; i < Q3K_VPL; i += 4) {
        const unsigned int qw = q3k_ld32(qp + i);
        const unsigned int hw = q3k_ld32(hp + i);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int qv = (int)((qw >> (8 * t + shift)) & 3u);
            const int sub = ((hw >> (8 * t)) & (unsigned)bit) ? 0 : 4;
            wv[i + t] = scale * (float)(qv - sub);
        }
    }
}

template <int B, int RPW, int TILE>
static __device__ __forceinline__ void q3k_tile_b(
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
    const int nblocks = k / Q3K_VALUES;
    const int row_tile = Q3K_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW][B];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * Q3K_BLOCK_BYTES;
#pragma unroll
        for (int bi = 0; bi < B; ++bi) acc[rr][bi] = 0.0f;
    }

    for (int base = 0; base < k; base += TILE) {
        const int n = min(TILE, k - base);
        const int nchunks = n / Q3K_VPL;
        for (int j = tid; j < nchunks; j += Q3K_BLOCK) {
#pragma unroll
            for (int bi = 0; bi < B; ++bi)
                q3k_stage_group(x + (size_t)bi * k, sx + bi * TILE, base, j);
        }
        __syncthreads();

        for (int c = lane * Q3K_VPL; c < n; c += Q3K_GROUP) {
            const int g = c / Q3K_GROUP;
            const int s = g * Q3K_GROUP + lane;
            const int p = base + c;
            float wv[RPW][Q3K_VPL];
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) q3k_load(rp[rr], p, wv[rr]);
#pragma unroll
            for (int bi = 0; bi < B; ++bi) {
                const float* src = sx + bi * TILE + s;
                float xv[Q3K_VPL];
#pragma unroll
                for (int i = 0; i < Q3K_VPL; ++i) xv[i] = src[i * 32];
#pragma unroll
                for (int rr = 0; rr < RPW; ++rr) {
                    float a = acc[rr][bi];
#pragma unroll
                    for (int i = 0; i < Q3K_VPL; ++i) a = fmaf(wv[rr][i], xv[i], a);
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
static __device__ __forceinline__ void q3k_entry_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    extern __shared__ __align__(16) float hipml_sx_b[];
    q3k_tile_b<B, Q3KB_ROWS_PER_WARP, TILE>(w, x, y, m, k, hipml_sx_b, blockIdx.x);
}

extern "C" __global__ void gemv_q3_k_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int batch)
{
    if (batch <= 4) {
        switch (batch) {
        case 1: q3k_entry_b<1, Q3KB_K_TILE_BIG>(w, x, y, m, k); break;
        case 2: q3k_entry_b<2, Q3KB_K_TILE_BIG>(w, x, y, m, k); break;
        case 3: q3k_entry_b<3, Q3KB_K_TILE_BIG>(w, x, y, m, k); break;
        case 4: q3k_entry_b<4, Q3KB_K_TILE_BIG>(w, x, y, m, k); break;
        }
    } else {
        switch (batch) {
        case 5: q3k_entry_b<5, Q3KB_K_TILE_SMALL>(w, x, y, m, k); break;
        case 6: q3k_entry_b<6, Q3KB_K_TILE_SMALL>(w, x, y, m, k); break;
        case 7: q3k_entry_b<7, Q3KB_K_TILE_SMALL>(w, x, y, m, k); break;
        default: q3k_entry_b<8, Q3KB_K_TILE_SMALL>(w, x, y, m, k); break;
        }
    }
}
