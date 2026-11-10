// SPDX-License-Identifier: Apache-2.0
// Per-head L2 normalisation of q and k.
//
// `build_gdn_l2_norm` is `rms_norm(x, eps/n) * 1/sqrt(n)`, which is exactly
// `x / sqrt(sum(x^2) + eps)`; the head dimension is the reduction length.
//
// One block per head. In-place is safe: every thread reads only its own
// elements before the reduction and writes them after the barrier. The two
// pointers may alias, so neither carries __restrict__.
//
// Kernel: l2norm_f32. Reference: delta::l2norm_f32.

#include <hip/hip_runtime.h>

#define L2_BLOCK 256

extern "C" __global__ void l2norm_f32(
    const float* x,
    float* y,
    int heads,
    int dim,
    float eps)
{
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const float* src = x + (size_t)h * dim;

    float acc = 0.0f;
    for (int i = t; i < dim; i += L2_BLOCK) {
        acc = __fmaf_rn(src[i], src[i], acc);
    }

    __shared__ float s[L2_BLOCK];
    s[t] = acc;
    __syncthreads();
    for (int off = L2_BLOCK / 2; off > 0; off >>= 1) {
        if (t < off) s[t] += s[t + off];
        __syncthreads();
    }

    __shared__ float scale;
    if (t == 0) scale = rsqrtf(s[0] + eps);
    __syncthreads();

    for (int i = t; i < dim; i += L2_BLOCK) {
        y[(size_t)h * dim + i] = src[i] * scale;
    }
}
