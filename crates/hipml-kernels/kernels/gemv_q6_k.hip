// y[row] = dequant_q6_k(W[row,:]) . x
//
// One block per output row, 256 threads, one thread per column. Thread t owns
// column t of every 256-value super-block, so x is coalesced and each warp is
// exactly one 32-value group: the ql and qh byte indices increment with the
// lane, giving 32-consecutive-byte reads.
//
// Q6_K super-block (210 bytes, 256 values):
//   ql(128) qh(64) scales(16) d(2)
// A thread at position t:
//   half  = t/128, group = (t%128)/32, l = t%32
//   sub   = group*2 + l/16              -> scale[half*8 + sub]
//   ql ix = half*64 + (group&1)*32 + l  (low nibble for group 0/1, high for 2/3)
//   qh ix = 128 + half*32 + l           (2-bit plane 2*group)
//
// Only the 8-float cross-warp reduction touches shared memory.

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

    const int t = threadIdx.x;
    const int half = t >> 7;
    const int in_half = t & 127;
    const int group = in_half >> 5;
    const int l = in_half & 31;
    const int sub = group * 2 + (l >> 4);
    const int scale_ix = half * 8 + sub;
    const bool high_nibble = group >= 2;
    const int qh_shift = 2 * group;
    const int ql_ix = half * 64 + ((group & 1) ? 32 : 0) + l;
    const int qh_ix = 128 + half * 32 + l;

    const int nblocks = k / Q6K_VALUES;
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q6K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int blk = 0; blk < nblocks; ++blk) {
        const unsigned char* b = r + (size_t)blk * Q6K_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        const float scale =
            d * (float)reinterpret_cast<const signed char*>(b + 192)[scale_ix];

        const unsigned int qb = b[ql_ix];
        const unsigned int nib = high_nibble ? (qb >> 4) : (qb & 0xF);
        const unsigned int hi = (b[qh_ix] >> qh_shift) & 3;
        const int q = (int)(nib | (hi << 4)) - 32;

        acc += scale * (float)q * x[(size_t)blk * Q6K_VALUES + t];
    }

    __shared__ float q6k_s[8];
    const int warp = t >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (l == 0) q6k_s[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (t < 8) ? q6k_s[t] : 0.0f;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (t == 0) y[row] = v;
    }
}
