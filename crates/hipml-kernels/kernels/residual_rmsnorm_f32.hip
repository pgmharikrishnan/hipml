// SPDX-License-Identifier: Apache-2.0
// Fused residual add + row RMS normalisation:
//
//   res[i] = a[i] + b[i]
//   y[i]   = res[i] / sqrt(mean(res^2) + eps) * w[i]
//
// This is `residual_add_f32` followed by `rmsnorm_f32`, fused so the row is
// never read back from global memory. The arithmetic order is the two-kernel
// order exactly: the same `a + b` per element, the same ascending per-thread
// sum of squares, the same shared-memory tree, and `v * scale * w` in the same
// order. The result is therefore bit-identical to the pair it replaces, which
// is what lets the token stream stay fixed.
//
// One block per row, `MAXK * BLOCK` columns at most; the host only selects this
// entry when `n` fits (the prefill width is 5120, so `MAXK = 32` is ample).

#include <hip/hip_runtime.h>

#define BLOCK 256
#define MAXK 32

extern "C" __global__ void residual_rmsnorm_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ w,
    float* __restrict__ res,
    float* __restrict__ y,
    int n,
    float eps)
{
    const int row = blockIdx.x;
    const float* ar = a + (size_t)row * (size_t)n;
    const float* br = b + (size_t)row * (size_t)n;
    float* rr = res + (size_t)row * (size_t)n;
    float* yr = y + (size_t)row * (size_t)n;

    // Keep this row's summands in registers: the second loop needs the same
    // values the first one added, and recomputing them from `res` would read
    // the row back from memory (the pass this kernel exists to remove).
    float v[MAXK];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int i = (int)threadIdx.x + k * BLOCK;
        if (i < n) {
            const float t = ar[i] + br[i];
            rr[i] = t;
            v[k] = t;
            acc += t * t;
        }
    }

    __shared__ float s[BLOCK];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }

    __shared__ float inv;
    if (threadIdx.x == 0) {
        const float mean_sq = s[0] / (float)n;
        inv = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float scale = inv;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int i = (int)threadIdx.x + k * BLOCK;
        if (i < n) {
            yr[i] = v[k] * scale * w[i];
        }
    }
}
