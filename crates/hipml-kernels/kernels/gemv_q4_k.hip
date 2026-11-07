// y[row] = dequant_q4_k(W[row,:]) . x
//
// One block per output row, 256 threads. Each thread owns whole 32-value
// sub-blocks, so it loads one 6-bit pair of scale/min per sub-block, unpacks the
// four-bit quants straight from global memory and FMAs into a register. The
// weights are never staged in shared memory; the only shared bytes are the
// eight per-warp sums used for the final reduction.
//
// Q4_K super-block (144 bytes, 256 values):
//   d(2) dmin(2) scales(12) qs(128)
// Sub-block j uses scale[j]/min[j] from the packed 6-bit header and reads the
// low nibble of qs[(j/2)*32 ..] when j is even, the high nibble when j is odd.

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

    const int nblocks = k / Q4K_VALUES;
    const int nsub = nblocks * 8;
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q4K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
        const int blk = sb >> 3;
        const int j = sb & 7;
        const unsigned char* b = r + (size_t)blk * Q4K_BLOCK_BYTES;

        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
        int sc, mn;
        q4k_scale_min(j, b + 4, &sc, &mn);
        const float scale = d * (float)sc;
        const float minv = dmin * (float)mn;

        const unsigned char* g = b + 16 + (j >> 1) * 32;
        const float* xr = x + (size_t)sb * 32;
        float a = 0.0f;
        if ((j & 1) == 0) {
#pragma unroll
            for (int l = 0; l < 32; ++l)
                a += (scale * (float)(g[l] & 0xF) - minv) * xr[l];
        } else {
#pragma unroll
            for (int l = 0; l < 32; ++l)
                a += (scale * (float)(g[l] >> 4) - minv) * xr[l];
        }
        acc += a;
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
