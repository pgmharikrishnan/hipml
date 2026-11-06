// One block per output row, one thread per column stride.
//
// y[row] = sum_j W[row][j] * x[j]
//
// Block size is fixed at 256 to match the shared-memory reduction below.

#include <hip/hip_runtime.h>

#define BLOCK 256

extern "C" __global__ void gemv_f32(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    const int row = blockIdx.x;
    if (row >= m) return;

    const float* r = w + (size_t)row * (size_t)k;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < k; j += BLOCK) {
        acc += r[j] * x[j];
    }

    __shared__ float s[BLOCK];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = s[0];
}
