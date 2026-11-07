// y[row] = dequant_q6_k(W[row,:]) . x
//
// One block per output row, 256 threads, whole 16-value sub-blocks per thread.
// Q6_K ships the low four bits in ql, the high two bits split across four qh
// planes, and 16 int8 sub-block scales. Sub-block j covers positions j*16..+15
// and uses scale[j]; its q comes from ql[l] (low or high nibble) plus the
// 2-bit qh plane selected by j's group.
//
// No shared staging of the weights: the only shared memory is the 8-float
// cross-warp reduction.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q6K_VALUES 256
#define Q6K_BLOCK_BYTES 210

extern "C" __global__ void gemv_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    const int row = blockIdx.x;
    if (row >= m) return;

    const int nblocks = k / Q6K_VALUES;
    const int nsub = nblocks * 16;
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q6K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
        const int blk = sb >> 4;
        const int j = sb & 15; // sub-block 0..15
        const unsigned char* b = r + (size_t)blk * Q6K_BLOCK_BYTES;

        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        const float scale = d * (float)reinterpret_cast<const signed char*>(b + 192)[j];

        const int half = j >> 3; // first or second 128-value half
        const int s = j & 7;
        const int group = s >> 1; // q1..q4
        const int is = s & 1; // which 16-value slice

        const unsigned char* ql = b + half * 64 + ((group & 1) ? 32 : 0);
        const unsigned char* qh = b + 128 + half * 32;
        const int l0 = is * 16;
        const bool high_nibble = group >= 2;
        const int qh_shift = 2 * group;

        const float* xr = x + (size_t)sb * 16;
        float a = 0.0f;
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            const unsigned int qb = ql[l0 + l];
            const unsigned int nib = high_nibble ? (qb >> 4) : (qb & 0xF);
            const unsigned int hi = (qh[l0 + l] >> qh_shift) & 3;
            const int q = (int)(nib | (hi << 4)) - 32;
            a += scale * (float)q * xr[l];
        }
        acc += a;
    }

    __shared__ float q6k_s[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) q6k_s[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (threadIdx.x < 8) ? q6k_s[threadIdx.x] : 0.0f;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (threadIdx.x == 0) y[row] = v;
    }
}
