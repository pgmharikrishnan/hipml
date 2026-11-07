// y[row] = dequant_q6_k(W[row,:]) . x
//
// One block per output row, 256 threads, 8 consecutive columns per thread. The
// chunk is one 8-value slice of a 16-value scale sub-block, so a single int8
// scale covers all 8 values. x is two float4s; ql/qh are four uint16s each
// (Q6_K rows are 210*nblocks bytes, only 2-byte aligned, so wider loads are not
// safe).
//
// Q6_K super-block (210 bytes, 256 values):
//   ql(128) qh(64) scales(16) d(2)
// A chunk at sub-block j (0..15), slice p (0..1):
//   half = j/8, s = j%8, group = s/2, is = s%2
//   base = j*16 + p*8                     (position inside the super-block)
//   ql   = half*64 + (group&1)*32 + is*16 + p*8 + i
//   qh   = 128 + half*32 + is*16 + p*8 + i   (2-bit plane 2*group)
//   scale= scales[half*8 + s]
//
// Only the 8-float cross-warp reduction touches shared memory.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q6K_VALUES 256
#define Q6K_BLOCK_BYTES 210
#define Q6K_CHUNKS 32 // 16 sub-blocks x 2 slices of 8 values

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
    const int nchunks = nblocks * Q6K_CHUNKS;
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q6K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int c = threadIdx.x; c < nchunks; c += blockDim.x) {
        const int blk = c >> 5;
        const int rem = c & 31;
        const int sub = rem >> 1; // 0..15
        const int part = rem & 1; // first or second 8 values

        const unsigned char* b = r + (size_t)blk * Q6K_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        const int half = sub >> 3;
        const int s = sub & 7;
        const int group = s >> 1;
        const int is = s & 1;
        const float scale =
            d * (float)reinterpret_cast<const signed char*>(b + 192)[half * 8 + s];

        const int l0 = is * 16 + part * 8;
        const unsigned char* ql = b + half * 64 + ((group & 1) ? 32 : 0) + l0;
        const unsigned char* qh = b + 128 + half * 32 + l0;
        const int qh_shift = 2 * group;
        const bool high_nibble = group >= 2;

        // 8 bytes each, 2-byte aligned -> four uint16 loads.
        unsigned short qlw[4];
        unsigned short qhw[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            qlw[w] = __ldcs(reinterpret_cast<const unsigned short*>(ql + 2 * w));
            qhw[w] = __ldcs(reinterpret_cast<const unsigned short*>(qh + 2 * w));
        }

        const float* xp = x + (size_t)blk * Q6K_VALUES + (size_t)sub * 16 + part * 8;
        const float4 xa = *reinterpret_cast<const float4*>(xp);
        const float4 xb = *reinterpret_cast<const float4*>(xp + 4);
        const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};

#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int i = 2 * w + kk;
                const unsigned int qb = (qlw[w] >> (8 * kk)) & 0xFF;
                const unsigned int hb = (qhw[w] >> (8 * kk)) & 0xFF;
                const unsigned int nib = high_nibble ? (qb >> 4) : (qb & 0xF);
                const unsigned int hi = (hb >> qh_shift) & 3;
                const int v = (int)(nib | (hi << 4)) - 32;
                acc += scale * (float)v * xv[i];
            }
        }
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
