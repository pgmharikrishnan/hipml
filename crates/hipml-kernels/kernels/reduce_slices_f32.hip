// Sum the k-slice partials of a split prefill GEMM into `y`, in slice order so
// the result is deterministic: y[i] += s[0][i] + s[1][i] + ... (`n - 1` slices).
// One launch instead of `n - 1` residual adds: the SSM projections run 4704
// times on a 100K prompt, where seven tiny launches each would cost more than
// the sum.

#include <hip/hip_runtime.h>

extern "C" __global__ void reduce_slices_f32(
    float* __restrict__ y,
    const float* __restrict__ slices,
    int n,
    int len)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    float a = y[i];
    for (int s = 0; s < n; ++s) a += slices[(size_t)s * len + i];
    y[i] = a;
}
