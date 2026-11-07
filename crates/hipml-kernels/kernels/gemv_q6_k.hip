// y[row] = dequant_q6_k(W[row,:]) . x
//
// One block per output row, 256 threads. A thread owns a whole 32-value chunk
// (two adjacent 16-value scale sub-blocks): it reads 32 ql bytes, 32 qh bytes
// and the two int8 scales, then produces 32 dot-product terms.
//
// Q6_K super-block (210 bytes, 256 values):
//   ql(128) qh(64) scales(16) d(2)
// A 32-value chunk at group g (0..3) of half h (0..1) starts at position
// h*128 + g*32. Groups 0/1 use the low ql nibble, 2/3 the high nibble, and the
// qh plane is selected by 2*g. Its two scales are scales[h*8 + 2g + {0,1}].
//
// No shared staging of the weights: only the 8-float cross-warp reduction.

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
    const int nchunks = nblocks * 8; // 32 values per chunk
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q6K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int c = threadIdx.x; c < nchunks; c += blockDim.x) {
        const int blk = c >> 3;
        const int in_block = c & 7;
        const int half = in_block >> 2; // first or second 128-value half
        const int group = in_block & 3; // 0..3
        const unsigned char* b = r + (size_t)blk * Q6K_BLOCK_BYTES;

        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        const signed char* scales = reinterpret_cast<const signed char*>(b + 192);
        const int sbase = half * 8 + group * 2;
        const float scale0 = d * (float)scales[sbase];
        const float scale1 = d * (float)scales[sbase + 1];

        const unsigned char* ql =
            b + half * 64 + ((group & 1) ? 32 : 0);
        const unsigned char* qh = b + 128 + half * 32;
        const bool high_nibble = group >= 2;
        const int qh_shift = 2 * group;

        const float* xr = x + (size_t)half * 128 + (size_t)group * 32;
        float a0 = 0.0f;
        float a1 = 0.0f;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const unsigned int qb = ql[l];
            const unsigned int nib = high_nibble ? (qb >> 4) : (qb & 0xF);
            const unsigned int hi = (qh[l] >> qh_shift) & 3;
            const int q = (int)(nib | (hi << 4)) - 32;
            if (l < 16)
                a0 += (scale0 * (float)q) * xr[l];
            else
                a1 += (scale1 * (float)q) * xr[l];
        }
        acc += a0 + a1;
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
