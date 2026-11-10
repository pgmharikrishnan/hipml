// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, tag b12126 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// The GDN time-step gate and beta, from the raw alpha/beta projections.
//
//   g[h]    = softplus(alpha[h] + dt_bias[h]) * ssm_a[h]
//   beta[h] = sigmoid(beta_raw[h])
//
// ssm_a is already negative in the file, so `g` is a log decay.
//
// Kernel: ssm_gates_f32. Reference: the `g`/`beta` construction in
// delta::ssm_layer_f32.

#include <hip/hip_runtime.h>

extern "C" __global__ void ssm_gates_f32(
    const float* __restrict__ alpha,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    float* __restrict__ g,
    float* __restrict__ beta,
    int n)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n) return;

    // Same branch as ggml_compute_softplus_f32 (softplus_f32 in ops.rs).
    const float z = alpha[h] + dt_bias[h];
    const float sp = z > 20.0f ? z : logf(1.0f + expf(z));
    g[h] = sp * a[h];
    beta[h] = 1.0f / (1.0f + expf(-beta_raw[h]));
}
