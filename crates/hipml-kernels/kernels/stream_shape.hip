// SPDX-License-Identifier: Apache-2.0
// Pure streaming read with the GEMV's own grid geometry.
//
// `stream_read` streams a tensor as one flat span, which is the best case for
// the memory path but not the shape a GEMV sees. These entries keep the row
// tile: 8 warps, two output rows per warp, and each warp walks its rows in
// 512-byte lane-interleaved steps exactly like `q4k_tile`'s `for (c = lane *
// VPL; c < n; c += GROUP)` loop. No dequant, no shared staging, no reduction
// beyond one value per block, so the result is the access pattern's ceiling for
// that tensor shape.
//
// Two widths:
//   _w4   four-byte loads, which is what a block format whose row stride is not
//         a multiple of 16 forces (Q5_K 2200 B rows, IQ4_XS 360 B, Q6_K 4200 B);
//   _w16  sixteen-byte loads, the width a 16/128-byte row stride would allow.
// Same bytes either way; the pair is the repack hypothesis measured directly.

#include <hip/hip_runtime.h>

#define SS_ROW_TILE 16
#define SS_WARPS 8
#define SS_RPW 2
#define SS_LANE_STEP 512

template <bool WIDE>
static __device__ __forceinline__ float ss_scan(const unsigned char* __restrict__ rp,
                                                int row_bytes,
                                                int lane)
{
    unsigned acc = 0;
    if (WIDE) {
        const int nchunk = row_bytes / 16;
        for (int c = lane; c < nchunk; c += 32) {
            const uint4 v = *reinterpret_cast<const uint4*>(rp + (size_t)c * 16);
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    } else {
        const int ndw = row_bytes / 4;
        for (int c = lane; c < ndw; c += 32) {
            acc ^= *reinterpret_cast<const unsigned int*>(rp + (size_t)c * 4);
        }
        const int rem = row_bytes & 3;
        if (lane < rem) acc ^= (unsigned)rp[row_bytes - 1 - lane];
    }
    // Fold per lane so the loop cannot be eliminated: the value is unused
    // downstream, but the XOR chain keeps every load live.
    return (float)(acc & 0xFFFFu);
}

extern "C" __global__ void stream_shape_w4(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int row_bytes)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float acc = 0.0f;
    (void)x;
    (void)k;
#pragma unroll
    for (int rr = 0; rr < SS_RPW; ++rr) {
        const int row = blockIdx.x * SS_ROW_TILE + warp * SS_RPW + rr;
        const int r = min(row, m - 1);
        acc += ss_scan<false>(w + (size_t)r * (size_t)row_bytes, row_bytes, lane);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) y[blockIdx.x * SS_WARPS + warp] = acc;
}

extern "C" __global__ void stream_shape_w16(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int row_bytes)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float acc = 0.0f;
    (void)x;
    (void)k;
#pragma unroll
    for (int rr = 0; rr < SS_RPW; ++rr) {
        const int row = blockIdx.x * SS_ROW_TILE + warp * SS_RPW + rr;
        const int r = min(row, m - 1);
        acc += ss_scan<true>(w + (size_t)r * (size_t)row_bytes, row_bytes, lane);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) y[blockIdx.x * SS_WARPS + warp] = acc;
}

// One row per warp (8 rows per block): the `_r1` geometry, for the grids whose
// 16-row tile leaves the machine under-occupied.
#define SS_ROW_TILE_R1 8

extern "C" __global__ void stream_shape_w4_r1(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int row_bytes)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    (void)x;
    (void)k;
    const int row = blockIdx.x * SS_ROW_TILE_R1 + warp;
    const int r = min(row, m - 1);
    float acc = ss_scan<false>(w + (size_t)r * (size_t)row_bytes, row_bytes, lane);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) y[blockIdx.x * SS_WARPS + warp] = acc;
}

// Degenerate: the same bytes with no row structure at all, one block per
// `blocks` grid slots (grid-stride over float4). Fixes the shape ceiling when
// the reader is allowed to ignore rows.
extern "C" __global__ void stream_shape_flat(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int row_bytes)
{
    const unsigned long long nvec = (unsigned long long)m * (unsigned long long)row_bytes / 16;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (; i < nvec; i += stride) {
        const uint4 v = *reinterpret_cast<const uint4*>(w + i * 16);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    float a = (float)(acc & 0xFFFFu);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
    if ((threadIdx.x & 31) == 0) y[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = a;
    (void)k;
}
