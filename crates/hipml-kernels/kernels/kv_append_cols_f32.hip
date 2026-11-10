// SPDX-License-Identifier: Apache-2.0
// Append `rows` token rows to a position-major f32 KV cache pair in one launch.
//
// k_cache and v_cache are [n_ctx][width]; the row for column `r` goes to
// position `pos0 + r`. Same bytes as `kv_append_f32`, which appends one row at
// `step[0]`; the verify path has no device step buffer to read, so the first
// position and the row count are kernel arguments.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_append_cols_f32(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int pos0,
    int rows,
    int width)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 2 * width) return;
    const int r = i / (2 * width);
    const int j = i - r * (2 * width);
    const size_t base = (size_t)(pos0 + r) * (size_t)width;
    const size_t src = (size_t)r * (size_t)width;
    if (j < width) {
        k_cache[base + j] = k[src + j];
    } else {
        v_cache[base + (j - width)] = v[src + (j - width)];
    }
}
