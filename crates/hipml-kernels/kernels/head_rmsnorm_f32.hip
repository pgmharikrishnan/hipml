// Per-head RMS norm over the last axis: y[h][d] = x[h][d] / sqrt(mean + eps) * w[d]
//
// One block per head. `x` and `y` may alias: the reduction reads every element
// before the writer touches its own, and each thread only writes indices it
// read. Deliberately no `__restrict__` on x/y for that reason.

#include <hip/hip_runtime.h>

#define HRN_BLOCK 256

extern "C" __global__ void head_rmsnorm_f32(
    const float* x,
    const float* __restrict__ w,
    float* y,
    int n_heads,
    int head_dim,
    float eps)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const float* xr = x + (size_t)h * (size_t)head_dim;
    float* yr = y + (size_t)h * (size_t)head_dim;

    float acc = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += HRN_BLOCK) {
        acc += xr[d] * xr[d];
    }

    __shared__ float red[HRN_BLOCK];
    __shared__ float inv;
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int off = HRN_BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float mean_sq = red[0] / (float)head_dim;
        inv = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float scale = inv;
    for (int d = threadIdx.x; d < head_dim; d += HRN_BLOCK) {
        yr[d] = xr[d] * scale * w[d];
    }
}
