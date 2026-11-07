// y[row] = dequant_q4_k(W[row,:]) . x
//
// One block per output row, 256 threads. A thread owns 8 consecutive columns of
// one 32-value sub-block, so:
//   * x is read as two float4s (32 bytes, coalesced across the warp);
//   * the 8 quant bytes are one uint2 (Q4_K rows are 144-byte multiples, so the
//     load is 8-byte aligned);
//   * one scale/min pair covers all 8 values.
// A block has 32 such chunks (8 sub-blocks x 4), so k=5120 gives 640 chunks and
// all 256 threads stay busy.
//
// Q4_K super-block (144 bytes, 256 values):
//   d(2) dmin(2) scales(12) qs(128)
// Sub-block j uses scale[j]/min[j]; even j reads the low nibble of
// qs[(j/2)*32 ..], odd j the high nibble.
//
// Only the 8-float cross-warp reduction touches shared memory.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q4K_VALUES 256
#define Q4K_BLOCK_BYTES 144
#define Q4K_CHUNKS 32 // 8 sub-blocks x 4 chunks of 8 values

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

    const int nblocks = k / Q4K_VALUES;
    const int nchunks = nblocks * Q4K_CHUNKS;
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q4K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int c = threadIdx.x; c < nchunks; c += blockDim.x) {
        const int blk = c >> 5;
        const int rem = c & 31;
        const int sub = rem >> 2; // sub-block 0..7
        const int part = rem & 3; // 8-value slice of the sub-block

        const unsigned char* b = r + (size_t)blk * Q4K_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
        int sc, mn;
        q4k_scale_min(sub, b + 4, &sc, &mn);
        const float scale = d * (float)sc;
        const float minv = dmin * (float)mn;

        union Q4Bytes {
            uint2 v;
            unsigned char b[8];
        } q;
        q.v = __ldcs(reinterpret_cast<const uint2*>(b + 16 + (sub >> 1) * 32 + part * 8));

        const float* xp = x + (size_t)blk * Q4K_VALUES + (size_t)sub * 32 + part * 8;
        const float4 xa = *reinterpret_cast<const float4*>(xp);
        const float4 xb = *reinterpret_cast<const float4*>(xp + 4);
        const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};

        const bool high_nibble = (sub & 1) != 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned int byte = q.b[i];
            const int v = high_nibble ? (int)(byte >> 4) : (int)(byte & 0xF);
            acc += (scale * (float)v - minv) * xv[i];
        }
    }

    __shared__ float q4k_s[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) q4k_s[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (threadIdx.x < 8) ? q4k_s[threadIdx.x] : 0.0f;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (threadIdx.x == 0) y[row] = v;
    }
}
