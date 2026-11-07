// y[row] = dequant_q8_0(W[row,:]) . x
//
// Q8_0 is a 34-byte block: one fp16 scale and 32 int8 values. It is the cheap
// format in the file (64 tensors), so the decode-stream bench needs a kernel for
// it to make the implied tok/s comparable with Q4_K/Q6_K.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q80_VALUES 32
#define Q80_BLOCK_BYTES 34

extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    const int row = blockIdx.x;
    if (row >= m) return;

    const int nblocks = k / Q80_VALUES;
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q80_BLOCK_BYTES;

    float acc = 0.0f;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        const unsigned char* blk = r + (size_t)b * Q80_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const signed char* q = reinterpret_cast<const signed char*>(blk + 2);
        const float* xr = x + (size_t)b * 32;
        float a = 0.0f;
#pragma unroll
        for (int l = 0; l < 32; ++l) a += (float)q[l] * xr[l];
        acc += d * a;
    }

    __shared__ float q80_s[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) q80_s[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (threadIdx.x < 8) ? q80_s[threadIdx.x] : 0.0f;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (threadIdx.x == 0) y[row] = v;
    }
}
