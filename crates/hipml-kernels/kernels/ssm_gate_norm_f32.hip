// Gated RMS norm of the recurrence output.
//
//   y[h][i] = rms_norm(o[h], ssm_norm) * silu(z[h][i])
//
// One block per value head; the RMS sum is block-wide, the gain and the gate
// are elementwise. `ssm_norm` is [head_v] and is shared by every head.
//
// Kernel: ssm_gate_norm_f32. Reference: the per-head rmsnorm + silu_mul in
// delta::ssm_layer_f32.

#include <hip/hip_runtime.h>

#define GN_BLOCK 256

extern "C" __global__ void ssm_gate_norm_f32(
    const float* __restrict__ o,
    const float* __restrict__ norm,
    const float* __restrict__ z,
    float* __restrict__ y,
    int heads,
    int dim,
    float eps)
{
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const float* src = o + (size_t)h * dim;

    float acc = 0.0f;
    for (int i = t; i < dim; i += GN_BLOCK) {
        acc = __fmaf_rn(src[i], src[i], acc);
    }

    __shared__ float s[GN_BLOCK];
    s[t] = acc;
    __syncthreads();
    for (int off = GN_BLOCK / 2; off > 0; off >>= 1) {
        if (t < off) s[t] += s[t + off];
        __syncthreads();
    }

    __shared__ float scale;
    if (t == 0) {
        scale = rsqrtf(s[0] / (float)dim + eps);
    }
    __syncthreads();

    float* dst = y + (size_t)h * dim;
    const float* zz = z + (size_t)h * dim;
    for (int i = t; i < dim; i += GN_BLOCK) {
        const float g = zz[i];
        const float silu = g / (1.0f + expf(-g));
        dst[i] = src[i] * scale * norm[i] * silu;
    }
}
