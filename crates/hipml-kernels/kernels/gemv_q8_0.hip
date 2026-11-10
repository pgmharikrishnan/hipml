// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's K-quant / q8_0 layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// y[row] = dequant_q8_0(W[row,:]) . x
//
// Q8_0 is a 34-byte block: one fp16 scale and 32 int8 values. It is the cheap
// format in the file (64 tensors), so the decode-stream bench needs a kernel for
// it to make the implied tok/s comparable with Q4_K/Q6_K.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q80_VALUES 32
#define Q80_BLOCK_BYTES 34

// One row, whole block. External linkage so gemv_jobs can call it: there the
// caller walks `row = tile, tile + gridDim.x, ...` inside its own job grid.
__device__ __forceinline__ void q80_row(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int row)
{
    if (row >= m) return;

    const int nblocks = k / Q80_VALUES;
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q80_BLOCK_BYTES;

    float acc = 0.0f;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        const unsigned char* blk = r + (size_t)b * Q80_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const signed char* q = reinterpret_cast<const signed char*>(blk + 2);
        const float* xr = x + (size_t)b * 32;
        float a = 0.0f;
#pragma unroll
        for (int l = 0; l < 32; ++l) a += (float)q[l] * xr[l];
        acc += d * a;
    }

    __shared__ float q80_s[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) q80_s[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (threadIdx.x < 8) ? q80_s[threadIdx.x] : 0.0f;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (threadIdx.x == 0) y[row] = v;
    }
}

extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    q80_row(w, x, y, m, k, blockIdx.x);
}

// ---------------------------------------------------------------------------
// Batched Q8_0 GEMV: one block per output row, B x rows in the same block, so a
// 34-byte weight block is loaded once and applied to all B tokens.
//
// The per-row arithmetic is *identical* to gemv_q8_0: thread t still owns
// blocks t, t+256, ... and still sums `d * sum_l q[l]*x[l]` in that order, and
// the two-stage warp/block reduction is the same. Only the redundant weight
// load is removed, so row b is bit-identical to a single-sequence launch.
// ---------------------------------------------------------------------------

template <int B>
static __device__ __forceinline__ void q80_row_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int row)
{
    if (row >= m) return;

    const int nblocks = k / Q80_VALUES;
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q80_BLOCK_BYTES;

    float acc[B];
#pragma unroll
    for (int bi = 0; bi < B; ++bi) acc[bi] = 0.0f;

    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        const unsigned char* blk = r + (size_t)b * Q80_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const signed char* q = reinterpret_cast<const signed char*>(blk + 2);
        float qf[32];
#pragma unroll
        for (int l = 0; l < 32; ++l) qf[l] = (float)q[l];
#pragma unroll
        for (int bi = 0; bi < B; ++bi) {
            const float* xr = x + (size_t)bi * k + (size_t)b * 32;
            float a = 0.0f;
#pragma unroll
            for (int l = 0; l < 32; ++l) a += qf[l] * xr[l];
            acc[bi] += d * a;
        }
    }

    __shared__ float q80_sb[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int bi = 0; bi < B; ++bi) {
        float a = acc[bi];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
        if (lane == 0) q80_sb[warp] = a;
        __syncthreads();
        if (warp == 0) {
            float v = (threadIdx.x < 8) ? q80_sb[threadIdx.x] : 0.0f;
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) v += __shfl_down(v, off);
            if (threadIdx.x == 0) y[(size_t)bi * m + row] = v;
        }
        __syncthreads();
    }
}

extern "C" __global__ void gemv_q8_0_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int batch)
{
    switch (batch) {
    case 1: q80_row_b<1>(w, x, y, m, k, blockIdx.x); break;
    case 2: q80_row_b<2>(w, x, y, m, k, blockIdx.x); break;
    case 3: q80_row_b<3>(w, x, y, m, k, blockIdx.x); break;
    case 4: q80_row_b<4>(w, x, y, m, k, blockIdx.x); break;
    case 5: q80_row_b<5>(w, x, y, m, k, blockIdx.x); break;
    case 6: q80_row_b<6>(w, x, y, m, k, blockIdx.x); break;
    case 7: q80_row_b<7>(w, x, y, m, k, blockIdx.x); break;
    default: q80_row_b<8>(w, x, y, m, k, blockIdx.x); break;
    }
}
