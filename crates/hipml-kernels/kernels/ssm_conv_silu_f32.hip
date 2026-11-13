// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, tag b12126 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// Causal depthwise conv1d followed by the SiLU the decoder applies to it.
//
// The decode layer used to run `ssm_conv1d_f32` and then `ssm_silu_inplace_f32`
// over the same `conv_ch` row, two in-order launches for one pass over one
// buffer. `ssm_conv1d_f32` stays as the parity target and for the chunked
// prefill; this entry is the decode layer's, and it is the same arithmetic:
// the same `__fmaf_rn` chain over the conv taps, then the same
// `v / (1 + expf(-v))` on that result, so the row is bit-identical.
//
// Kernel: ssm_conv_silu_f32. Reference: delta::ssm_conv1d_f32 then the
// `a / (1 + exp(-a))` loop in delta::ssm_layer_f32.

#include <hip/hip_runtime.h>

extern "C" __global__ void ssm_conv_silu_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ state,
    float* __restrict__ y,
    int channels,
    int d_conv)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    const int hist = d_conv - 1;
    const float* wc = w + (size_t)c * d_conv;

    float acc = 0.0f;
    for (int k = 0; k < hist; ++k) {
        acc = __fmaf_rn(state[(size_t)k * channels + c], wc[k], acc);
    }
    acc = __fmaf_rn(x[c], wc[hist], acc);
    y[c] = acc / (1.0f + expf(-acc));

    // Slide left, then append the current token. Read-before-write is per
    // thread, so the copy cannot race another channel.
    for (int k = 0; k + 1 < hist; ++k) {
        state[(size_t)k * channels + c] = state[(size_t)(k + 1) * channels + c];
    }
    if (hist > 0) {
        state[(size_t)(hist - 1) * channels + c] = x[c];
    }
}
