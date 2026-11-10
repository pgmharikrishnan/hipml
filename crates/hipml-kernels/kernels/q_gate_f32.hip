// SPDX-License-Identifier: Apache-2.0
// Attention output gate: y[i] = x[i] * sigmoid(gate[i]).
//
// Kernel: q_gate_f32, elementwise. y may alias x.

#include <hip/hip_runtime.h>

extern "C" __global__ void q_gate_f32(
    const float* x,
    const float* __restrict__ gate,
    float* y,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] * (1.0f / (1.0f + expf(-gate[i])));
}
