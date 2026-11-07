// y[row] = dequant_q4_k(W[row,:]) . x
//
// One block per output row, 256 threads, one thread per column. Thread t owns
// column t of every 256-value super-block, so:
//   * x[blk*256 + t] is coalesced across the warp;
//   * the 32 threads of a warp are exactly one 32-value sub-block, so the
//     scale/min load is a warp-uniform broadcast;
//   * the quant byte index is constant within a block, so the warp reads 32
//     consecutive qs bytes (one sector).
//
// Q4_K super-block (144 bytes, 256 values):
//   d(2) dmin(2) scales(12) qs(128)
// Sub-block j uses scale[j]/min[j] from the packed 6-bit header; even j reads
// the low nibble of qs[(j/2)*32 ..], odd j the high nibble.
//
// No shared staging of the weights; the only shared memory is the 8-float
// cross-warp reduction.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q4K_VALUES 256
#define Q4K_BLOCK_BYTES 144

static __device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* q,
                                                     int* sc, int* mn)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

extern "C" __global__ void gemv_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    const int row = blockIdx.x;
    if (row >= m) return;

    const int t = threadIdx.x; // column within a super-block (0..255)
    const int j = t >> 5; // sub-block 0..7; uniform within a warp
    const int lane = t & 31; // byte within the 32-byte group
    const int group = j >> 1;
    const bool high_nibble = (j & 1) != 0;

    const int nblocks = k / Q4K_VALUES;
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q4K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int blk = 0; blk < nblocks; ++blk) {
        const unsigned char* b = r + (size_t)blk * Q4K_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
        int sc, mn;
        q4k_scale_min(j, b + 4, &sc, &mn);

        const unsigned int byte = b[16 + group * 32 + lane];
        const int q = high_nibble ? (int)(byte >> 4) : (int)(byte & 0xF);
        acc += (d * (float)sc * (float)q - dmin * (float)mn) *
               x[(size_t)blk * Q4K_VALUES + t];
    }

    __shared__ float q4k_s[8];
    const int warp = t >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) q4k_s[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (t < 8) ? q4k_s[t] : 0.0f;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (t == 0) y[row] = v;
    }
}
