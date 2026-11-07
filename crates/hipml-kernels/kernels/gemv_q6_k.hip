// y[row] = dequant_q6_k(W[row,:]) . x
//
// Row-tile GEMV: one block owns ROW_TILE output rows (one warp per row) and
// stages the x k-tile in shared memory once, so x is read once per tile instead
// of once per row. This is the same structure that took the f32 GEMM from
// 92.7 GB/s to 627 GB/s.
//
// Each lane owns 4 consecutive columns, read as one float4 from shared. Its
// weight bytes are four ql bytes and four qh bytes (2-byte aligned in Q6_K, so
// two uint16 loads each). A 4-value chunk never crosses a 16-value scale
// sub-block, so one int8 scale covers it.
//
// Q6_K super-block (210 bytes, 256 values):
//   ql(128) qh(64) scales(16) d(2)
// Math is unchanged: q = (ql nibble) | ((qh plane) << 4), y = d*scale*(q-32).
//
// Two barriers per k-tile, none per row; the reduction is a warp shuffle tree.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q6K_VALUES 256
#define Q6K_BLOCK_BYTES 210
#define BLOCK 256
#define WARPS (BLOCK / 32)
#define ROW_TILE WARPS
#define K_TILE 512

// Mirrors the manifest's KernelKind::RowTile { rows = ROW_TILE, block = BLOCK };
// change both together.

extern "C" __global__ void gemv_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[K_TILE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * ROW_TILE + warp;
    const int r = min(row, m - 1);
    const int nblocks = k / Q6K_VALUES;
    const unsigned char* rp =
        w + (size_t)r * (size_t)nblocks * Q6K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int base = 0; base < k; base += K_TILE) {
        const int n = min(K_TILE, k - base);

        if ((n & 3) == 0) {
            const int nv = n >> 2;
            for (int j = tid; j < nv; j += BLOCK)
                reinterpret_cast<float4*>(sx)[j] =
                    reinterpret_cast<const float4*>(x + base)[j];
        } else {
            for (int j = tid; j < n; j += BLOCK) sx[j] = x[base + j];
        }
        __syncthreads();

        for (int c = lane * 4; c < n; c += 128) {
            const int p = base + c;
            const int blk = p >> 8;
            const int within = p & 255;
            const int half = within >> 7;
            const int ph = within & 127;
            const int group = ph >> 5;
            const int l = ph & 31;

            const unsigned char* b = rp + (size_t)blk * Q6K_BLOCK_BYTES;
            const float d =
                __half2float(*reinterpret_cast<const __half*>(b + 208));
            const int sub = group * 2 + (l >> 4);
            const float scale =
                d * (float)reinterpret_cast<const signed char*>(b + 192)[half * 8 + sub];

            const int ql_off = half * 64 + ((group & 1) ? 32 : 0) + l;
            const int qh_off = 128 + half * 32 + l;
            const unsigned short ql0 =
                *reinterpret_cast<const unsigned short*>(b + ql_off);
            const unsigned short ql1 =
                *reinterpret_cast<const unsigned short*>(b + ql_off + 2);
            const unsigned short qh0 =
                *reinterpret_cast<const unsigned short*>(b + qh_off);
            const unsigned short qh1 =
                *reinterpret_cast<const unsigned short*>(b + qh_off + 2);
            const unsigned int qlw = (unsigned int)ql0 | ((unsigned int)ql1 << 16);
            const unsigned int qhw = (unsigned int)qh0 | ((unsigned int)qh1 << 16);

            const float4 xv = reinterpret_cast<const float4*>(sx)[c >> 2];
            const int shift = 2 * group;
            const bool high = group >= 2;

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int qb = (qlw >> (8 * i)) & 0xFFu;
                const unsigned int hb = (qhw >> (8 * i)) & 0xFFu;
                const unsigned int nib = high ? (qb >> 4) : (qb & 0xF);
                const unsigned int hi = (hb >> shift) & 3u;
                const int v = (int)(nib | (hi << 4)) - 32;
                const float xf = (i == 0) ? xv.x
                                : (i == 1) ? xv.y
                                : (i == 2) ? xv.z
                                           : xv.w;
                acc = fmaf(scale * (float)v, xf, acc);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0 && row < m) y[row] = acc;
}
