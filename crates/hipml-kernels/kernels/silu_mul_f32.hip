// SPDX-License-Identifier: Apache-2.0
// SwiGLU activation: out[i] = silu(gate[i]) * up[i]

#include <hip/hip_runtime.h>

extern "C" __global__ void silu_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}
