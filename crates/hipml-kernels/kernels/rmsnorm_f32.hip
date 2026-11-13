// SPDX-License-Identifier: Apache-2.0
// Row-wise RMS normalisation: y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
//
// One block per row; the block reduces the sum of squares, then scales.
//
// This is the only norm on the decode path and it runs 129 times a token, so its
// *duration* is the whole cost: one block means 63 of 64 CUs are idle and the
// next launch on the stream cannot start until this one finishes. The two loops
// below are therefore written to put every load a thread owns in flight before
// consuming any of them, rather than walking the row one strided load at a time.
//
// The arithmetic is unchanged. The sum of squares is still accumulated in the
// strided order (`x[t]`, `x[t + BLOCK]`, …) and the shared-memory tree is
// untouched, so the result is bit-identical to the previous form; the scale pass
// is elementwise and its per-element expression is the same. `MAXK * BLOCK` is
// the widest row this keeps in registers; wider rows fall through to the strided
// tail, which is the old loop.

#include <hip/hip_runtime.h>

#define BLOCK 256

// 20 x 256 = 5120, the hidden width of every Qwen3.8 build.
#define MAXK 20

extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int n,
    float eps)
{
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * (size_t)n;
    const float* wr = w;
    float* yr = y + (size_t)row * (size_t)n;

    // Pass 1: issue this thread's loads, then sum them in the strided order.
    // A k whose index is past `n` contributes 0*0, which is exact.
    float v[MAXK];
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int i = (int)threadIdx.x + k * BLOCK;
        v[k] = (i < n) ? xr[i] : 0.0f;
    }
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        acc += v[k] * v[k];
    }
    for (int i = (int)threadIdx.x + MAXK * BLOCK; i < n; i += BLOCK) {
        acc += xr[i] * xr[i];
    }

    __shared__ float s[BLOCK];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }

    __shared__ float inv;
    if (threadIdx.x == 0) {
        const float mean_sq = s[0] / (float)n;
        inv = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();

    // Pass 2: elementwise, so the width can be widened. A row start is 16-byte
    // aligned whenever `n` is a multiple of 4, which is the only case taken.
    const float scale = inv;
    if ((n & 3) == 0) {
        const float4* x4 = (const float4*)xr;
        const float4* w4 = (const float4*)wr;
        float4* y4 = (float4*)yr;
        const int n4 = n >> 2;
        for (int i = (int)threadIdx.x; i < n4; i += BLOCK) {
            const float4 a = x4[i];
            const float4 c = w4[i];
            y4[i] = make_float4(a.x * scale * c.x, a.y * scale * c.y,
                                a.z * scale * c.z, a.w * scale * c.w);
        }
    } else {
        for (int i = (int)threadIdx.x; i < n; i += BLOCK) {
            yr[i] = xr[i] * scale * wr[i];
        }
    }
}
