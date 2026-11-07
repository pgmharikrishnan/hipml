// In-place SiLU, the activation qwen35 applies to the conv1d output.
//
// Kernel: ssm_silu_inplace_f32. Reference: the `a / (1 + exp(-a))` loop in
// delta::ssm_layer_f32.

#include <hip/hip_runtime.h>

extern "C" __global__ void ssm_silu_inplace_f32(float* __restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = v / (1.0f + expf(-v));
}
