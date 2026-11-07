// y[row] = dequant_q4_k(W[row,:]) . x
//
// One block per output row, 256 threads. A thread owns a whole 64-value group
// (two adjacent 32-value sub-blocks) and loads its 32 bytes of quants once as
// two uint4s. Both nibbles of every byte belong to that thread, so there is no
// duplicate weight read — the even sub-block is the low nibble, the odd one the
// high nibble, each with its own 6-bit scale/min.
//
// Q4_K super-block (144 bytes, 256 values):
//   d(2) dmin(2) scales(12) qs(128)
//
// Weights are never staged in shared memory; the only shared bytes are the
// eight per-warp sums used for the final reduction.

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
    const int ngroups = nblocks * 4; // 64 values per group
    const unsigned char* r =
        w + (size_t)row * (size_t)nblocks * Q4K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int g = threadIdx.x; g < ngroups; g += blockDim.x) {
        const int blk = g >> 2;
        const int grp = g & 3;
        const unsigned char* b = r + (size_t)blk * Q4K_BLOCK_BYTES;

        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));

        const int j0 = grp * 2;
        int sc0, mn0, sc1, mn1;
        q4k_scale_min(j0, b + 4, &sc0, &mn0);
        q4k_scale_min(j0 + 1, b + 4, &sc1, &mn1);
        const float s0 = d * (float)sc0;
        const float m0 = dmin * (float)mn0;
        const float s1 = d * (float)sc1;
        const float m1 = dmin * (float)mn1;

        // 32 bytes = two 16-byte vector loads; byte i has the even value in
        // its low nibble and the odd value in its high nibble. Q4_K blocks are
        // 144 bytes and rows are 144*nblocks, so these loads are 16-byte aligned.
        union Q4Group {
            struct {
                uint4 a;
                uint4 b;
            } v;
            unsigned char bytes[32];
        } q;
        const unsigned char* qs = b + 16 + grp * 32;
        q.v.a = *reinterpret_cast<const uint4*>(qs);
        q.v.b = *reinterpret_cast<const uint4*>(qs + 16);

        const float* xr = x + (size_t)g * 64;
        float a0 = 0.0f;
        float a1 = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const unsigned int byte = q.bytes[i];
            a0 += (s0 * (float)(byte & 0xF) - m0) * xr[i];
            a1 += (s1 * (float)(byte >> 4) - m1) * xr[32 + i];
        }
        acc += a0 + a1;
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
