// Fused residual add + row RMS normalisation.
//
//   x1[i] = a[i] + b[i]
//   y[i]  = x1[i] / sqrt(mean(x1^2) + eps) * w[i]
//
// This replaces the `residual_add_f32` + `rmsnorm_f32` pair the decode graph
// runs twice per layer. The arithmetic is the same expression in the same
// order: the sum-of-squares loop is rmsnorm's (thread `i`, stride BLOCK) and
// the tree reduction is its tree, so `y` is bit-identical to normalising the
// materialised residual. The saving is one launch per residual point, not a
// different result.
//
// One block: n is 5120 here (20 elements per thread), so a single block's
// memory time is well under the launch it removes. x1 stays in L1 for the
// second pass.

#include <hip/hip_runtime.h>

#define RR_BLOCK 256

extern "C" __global__ void residual_rmsnorm_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ x1,
    const float* __restrict__ w,
    float* __restrict__ y,
    int n,
    float eps)
{
    const int tid = threadIdx.x;

    float acc = 0.0f;
    for (int i = tid; i < n; i += RR_BLOCK) {
        const float v = a[i] + b[i];
        x1[i] = v;
        acc += v * v;
    }

    __shared__ float s[RR_BLOCK];
    s[tid] = acc;
    __syncthreads();
    for (int off = RR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }

    __shared__ float inv;
    if (tid == 0) {
        const float mean_sq = s[0] / (float)n;
        inv = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float scale = inv;
    for (int i = tid; i < n; i += RR_BLOCK) {
        y[i] = x1[i] * scale * w[i];
    }
}
