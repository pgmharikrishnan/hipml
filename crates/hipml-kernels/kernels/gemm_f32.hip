// SPDX-License-Identifier: Apache-2.0
// One block per output column, staging each weight tile once for all R rows.
// y[r][row] = sum_j W[row][j] * x[r][j]
//
// Shared tiles support any R without a fixed register array, at the cost of a
// reduction and synchronization per row and tile. Aligned weights use float4;
// unaligned rows and the tail use scalar loads.

#include <hip/hip_runtime.h>

extern "C" __global__ void gemm_f32(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int r,
    int m,
    int k)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= m) return;
    const float* weights = w + (size_t)row * k;
    __shared__ float tile[256];
    __shared__ float sums[256];

    for (int batch = tid; batch < r; batch += 256) y[(size_t)batch * m + row] = 0.0f;
    __syncthreads();
    for (int base = 0; base < k; base += 256) {
        if (k % 4 == 0 && base + 256 <= k) {
            if (tid < 64) {
                float4 v = reinterpret_cast<const float4*>(weights + base)[tid];
                tile[4 * tid] = v.x;
                tile[4 * tid + 1] = v.y;
                tile[4 * tid + 2] = v.z;
                tile[4 * tid + 3] = v.w;
            }
        } else {
            tile[tid] = base + tid < k ? weights[base + tid] : 0.0f;
        }
        __syncthreads();
        for (int batch = 0; batch < r; ++batch) {
            sums[tid] = base + tid < k ? tile[tid] * x[(size_t)batch * k + base + tid] : 0.0f;
            __syncthreads();
            for (int off = 128; off > 0; off >>= 1) {
                if (tid < off) sums[tid] += sums[tid + off];
                __syncthreads();
            }
            if (tid == 0) y[(size_t)batch * m + row] += sums[0];
            __syncthreads();
        }
    }
}
