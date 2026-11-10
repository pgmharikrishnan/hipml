// SPDX-License-Identifier: Apache-2.0
// Small-m multi-row f32 GEMM: one warp per (token, output row).
//
// `gemm_f32` is a decode-shaped kernel: one block per output row and a
// block-wide reduction for every (token, k-tile), so a prefill chunk's
// `alpha`/`beta` projections (48 rows x 512 tokens) cost more than the 48x
// weight. Here a warp owns one dot product, lanes read consecutive k values
// (coalesced 128 B), and a 5-step shuffle replaces the barriers.
//
// Kernel: gemm_rows_f32. Reference: `gemm_f32` / ops::gemm_f32.

#include <hip/hip_runtime.h>

#define GR_WARPS 8
#define GR_THREADS (GR_WARPS * 32)

extern "C" __global__ void gemm_rows_f32(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int tokens,
    int m,
    int k)
{
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int idx = blockIdx.x * GR_WARPS + warp;
    if (idx >= tokens * m) return;
    const int t = idx / m;
    const int row = idx - t * m;

    const float* xr = x + (size_t)t * k;
    const float* wr = w + (size_t)row * k;
    float acc = 0.0f;
    for (int i = lane; i < k; i += 32) acc += xr[i] * wr[i];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffffffffffULL, acc, off);
    if (lane == 0) y[(size_t)t * m + row] = acc;
}
