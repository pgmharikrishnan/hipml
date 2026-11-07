// Dequantise Q4_K super-blocks to f32.
//
// One block of 256 threads per super-block; thread `l` writes value `l` using
// the same 32-value sub-block mapping the GEMV kernels use:
//
//   d(2) dmin(2) scales(12) qs(128)
//   y = d*scale[sub]*q - dmin*min[sub]
//
// This is the embedding-row path: a token id selects one 5120-value row, so the
// loader launches this over that row's 20 super-blocks only. It is not on the
// decode hot path; the GEMV kernels dequantise inline.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define DQ4K_VALUES 256
#define DQ4K_BLOCK_BYTES 144

static __device__ __forceinline__ void dq4k_scale_min(int j,
                                                      const unsigned char* q,
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

extern "C" __global__ void dequant_q4_k_f32(
    const unsigned char* __restrict__ w,
    float* __restrict__ out,
    int nblocks)
{
    const int b = blockIdx.x;
    const int l = threadIdx.x;
    if (b >= nblocks || l >= DQ4K_VALUES) return;

    const unsigned char* bp = w + (size_t)b * DQ4K_BLOCK_BYTES;
    const int sub = l >> 5;
    const int within = l & 31;

    const float d = __half2float(*reinterpret_cast<const __half*>(bp));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(bp + 2));
    int sc, mn;
    dq4k_scale_min(sub, bp + 4, &sc, &mn);
    const float scale = d * (float)sc;
    const float minv = dmin * (float)mn;

    const unsigned char qb = bp[16 + (sub >> 1) * 32 + within];
    const int q = (sub & 1) ? (int)(qb >> 4) : (int)(qb & 0xF);
    out[(size_t)b * DQ4K_VALUES + l] = scale * (float)q - minv;
}
