// SPDX-License-Identifier: Apache-2.0
// Residual connection: out[i] = a[i] + b[i]

#include <hip/hip_runtime.h>

extern "C" __global__ void residual_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}
