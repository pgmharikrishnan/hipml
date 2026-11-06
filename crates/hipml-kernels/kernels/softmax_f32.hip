// Row-wise softmax with max subtraction.
//
// One block per row: max reduction, exp+sum reduction, then write.

#include <hip/hip_runtime.h>

#define BLOCK 256

extern "C" __global__ void softmax_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n)
{
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * (size_t)n;
    float* yr = y + (size_t)row * (size_t)n;

    __shared__ float s[BLOCK];

    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += BLOCK) m = fmaxf(m, xr[i]);
    s[threadIdx.x] = m;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + off]);
        __syncthreads();
    }
    __shared__ float maxv;
    if (threadIdx.x == 0) maxv = s[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLOCK) sum += expf(xr[i] - maxv);
    __syncthreads();
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    __shared__ float inv;
    if (threadIdx.x == 0) inv = 1.0f / s[0];
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += BLOCK) {
        yr[i] = expf(xr[i] - maxv) * inv;
    }
}
