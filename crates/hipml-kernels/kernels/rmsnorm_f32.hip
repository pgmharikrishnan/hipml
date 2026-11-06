// Row-wise RMS normalisation: y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
//
// One block per row; the block reduces the sum of squares, then scales.

#include <hip/hip_runtime.h>

#define BLOCK 256

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

    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLOCK) {
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

    const float scale = inv;
    for (int i = threadIdx.x; i < n; i += BLOCK) {
        yr[i] = xr[i] * scale * wr[i];
    }
}
