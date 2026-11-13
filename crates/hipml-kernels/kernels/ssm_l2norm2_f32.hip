// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, tag b12126 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// Per-head L2 normalisation of the q and k slices of the conv output, in one
// launch.
//
// `build_gdn_l2_norm` is `rms_norm(x, eps/n) * 1/sqrt(n)`, which is exactly
// `x / sqrt(sum(x^2) + eps)`; the head dimension is the reduction length.
// The two slices sit at `off0` and `off1` of the same row, so block `h` of the
// first `heads` normalises q head `h` and block `heads + h` normalises k head
// `h`. Same block geometry and same reduction tree as `l2norm_f32`, so each
// head is bit-identical to the two-launch form.
//
// Kernel: ssm_l2norm2_f32. Reference: delta::l2norm_f32, applied twice.

#include <hip/hip_runtime.h>

#define L2B_BLOCK 256

extern "C" __global__ void ssm_l2norm2_f32(
    float* x,
    int off0,
    int off1,
    int heads,
    int dim,
    float eps)
{
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    float* src = x + (size_t)(b < heads ? off0 : off1) +
                 (size_t)(b < heads ? b : b - heads) * (size_t)dim;

    float acc = 0.0f;
    for (int i = t; i < dim; i += L2B_BLOCK) {
        acc = __fmaf_rn(src[i], src[i], acc);
    }

    __shared__ float s[L2B_BLOCK];
    s[t] = acc;
    __syncthreads();
    for (int off = L2B_BLOCK / 2; off > 0; off >>= 1) {
        if (t < off) s[t] += s[t + off];
        __syncthreads();
    }

    __shared__ float scale;
    if (t == 0) scale = rsqrtf(s[0] + eps);
    __syncthreads();

    for (int i = t; i < dim; i += L2B_BLOCK) {
        src[i] = src[i] * scale;
    }
}
