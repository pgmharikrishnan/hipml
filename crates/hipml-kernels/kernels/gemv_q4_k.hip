// y[row] = dequant_q4_k(W[row,:]) . x
//
// Row-tile GEMV: one block owns ROW_TILE output rows (one warp per row) and
// stages the x k-tile in shared memory once. Every warp then reuses that tile,
// so x is read from global/L2 once per ROW_TILE rows instead of once per row.
// With 4-bit weights the old per-row kernel read 5-8x more x bytes than weight
// bytes and was L2-bound; this is the fix.
//
// Each lane owns 4 consecutive columns, read as one float4 from shared. Its
// weight bytes are one 32-bit word (four nibbles) from the Q4_K row. Lanes 0-7
// cover sub-block 0, 8-15 sub-block 1 (the high nibble of the same bytes), and
// so on, so a warp's weight load spans two 32-byte groups with no waste.
//
// Q4_K super-block (144 bytes, 256 values):
//   d(2) dmin(2) scales(12) qs(128)
// Math is unchanged: y = d*scale[j]*q - dmin*min[j] for sub-block j.
//
// Two barriers per k-tile, none per row. The only reduction is a warp shuffle
// tree: 32 lanes are the only partials left.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q4K_VALUES 256
#define Q4K_BLOCK_BYTES 144
#define BLOCK 256
#define WARPS (BLOCK / 32)
#define ROW_TILE WARPS
#define K_TILE 512

// Mirrors the manifest's KernelKind::RowTile { rows = ROW_TILE, block = BLOCK };
// change both together.

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
    __shared__ __align__(16) float sx[K_TILE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * ROW_TILE + warp;
    const int r = min(row, m - 1); // clamp, do not store when row >= m
    const int nblocks = k / Q4K_VALUES;
    const unsigned char* rp =
        w + (size_t)r * (size_t)nblocks * Q4K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int base = 0; base < k; base += K_TILE) {
        const int n = min(K_TILE, k - base);

        // Stage x[base, base+n) once for all ROW_TILE rows.
        if ((n & 3) == 0) {
            const int nv = n >> 2;
            for (int j = tid; j < nv; j += BLOCK)
                reinterpret_cast<float4*>(sx)[j] =
                    reinterpret_cast<const float4*>(x + base)[j];
        } else {
            for (int j = tid; j < n; j += BLOCK) sx[j] = x[base + j];
        }
        __syncthreads();

        // One warp per row; each lane owns four consecutive columns.
        for (int c = lane * 4; c < n; c += 128) {
            const int p = base + c;
            const int blk = p >> 8; // p / 256
            const int within = p & 255;
            const int sub = within >> 5;
            const int part = (within & 31) >> 2;

            const unsigned char* b = rp + (size_t)blk * Q4K_BLOCK_BYTES;
            const float d = __half2float(*reinterpret_cast<const __half*>(b));
            const float dmin =
                __half2float(*reinterpret_cast<const __half*>(b + 2));
            int sc, mn;
            q4k_scale_min(sub, b + 4, &sc, &mn);
            const float scale = d * (float)sc;
            const float minv = dmin * (float)mn;

            const unsigned int qw = *reinterpret_cast<const unsigned int*>(
                b + 16 + (sub >> 1) * 32 + part * 4);
            const float4 xv = reinterpret_cast<const float4*>(sx)[c >> 2];
            const bool high = (sub & 1) != 0;

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int byte = (qw >> (8 * i)) & 0xFFu;
                const int v = high ? (int)(byte >> 4) : (int)(byte & 0xF);
                const float xf = (i == 0) ? xv.x
                                : (i == 1) ? xv.y
                                : (i == 2) ? xv.z
                                           : xv.w;
                acc = fmaf(scale * (float)v - minv, xf, acc);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0 && row < m) y[row] = acc;
}
