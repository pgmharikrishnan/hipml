// y[row] = dequant_q6_k(W[row,:]) . x
//
// Row-tile GEMV: one block owns ROW_TILE output rows (one warp per row) and
// stages the x k-tile in shared memory once, so x is read once per tile instead
// of once per row. Same structure that took the f32 batched GEMM to 627 GB/s.
//
// The shared tile is de-interleaved by 256-value group: component `c` of a group
// lives at sx[g*256 + c*32 + lane], so a lane's eight scalar reads hit eight
// distinct banks and the warp has no bank conflicts.
//
// Each lane owns 8 consecutive columns. In Q6_K those are eight ql bytes and
// eight qh bytes (2-byte aligned, so four uint16 loads each) and one int8
// scale: a chunk never crosses a 16-value scale sub-block.
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

static __device__ __forceinline__ void stage_group6(const float* __restrict__ x,
                                                    float* __restrict__ sx,
                                                    int base, int j)
{
    const float4 a = reinterpret_cast<const float4*>(x + base)[2 * j];
    const float4 b = reinterpret_cast<const float4*>(x + base)[2 * j + 1];
    const int g = j >> 5;
    const int lane = j & 31;
    float* s = sx + g * 256 + lane;
    s[0 * 32] = a.x;
    s[1 * 32] = a.y;
    s[2 * 32] = a.z;
    s[3 * 32] = a.w;
    s[4 * 32] = b.x;
    s[5 * 32] = b.y;
    s[6 * 32] = b.z;
    s[7 * 32] = b.w;
}

static __device__ __forceinline__ unsigned int join16(unsigned short a,
                                                      unsigned short b)
{
    return (unsigned int)a | ((unsigned int)b << 16);
}

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
        const int nchunks = n >> 3;
        for (int j = tid; j < nchunks; j += BLOCK) stage_group6(x, sx, base, j);
        __syncthreads();

        for (int c = lane * 8; c < n; c += 256) {
            const int g = c >> 8;
            const int s = g * 256 + lane;
            float xv[8];
#pragma unroll
            for (int comp = 0; comp < 8; ++comp) xv[comp] = sx[s + comp * 32];

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
            const unsigned int qlw0 = join16(
                *reinterpret_cast<const unsigned short*>(b + ql_off),
                *reinterpret_cast<const unsigned short*>(b + ql_off + 2));
            const unsigned int qlw1 = join16(
                *reinterpret_cast<const unsigned short*>(b + ql_off + 4),
                *reinterpret_cast<const unsigned short*>(b + ql_off + 6));
            const unsigned int qhw0 = join16(
                *reinterpret_cast<const unsigned short*>(b + qh_off),
                *reinterpret_cast<const unsigned short*>(b + qh_off + 2));
            const unsigned int qhw1 = join16(
                *reinterpret_cast<const unsigned short*>(b + qh_off + 4),
                *reinterpret_cast<const unsigned short*>(b + qh_off + 6));

            const int shift = 2 * group;
            const bool high = group >= 2;

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned int qw = (i < 4) ? qlw0 : qlw1;
                const unsigned int hw = (i < 4) ? qhw0 : qhw1;
                const unsigned int qb = (qw >> (8 * (i & 3))) & 0xFFu;
                const unsigned int hb = (hw >> (8 * (i & 3))) & 0xFFu;
                const unsigned int nib = high ? (qb >> 4) : (qb & 0xF);
                const unsigned int hi = (hb >> shift) & 3u;
                const int v = (int)(nib | (hi << 4)) - 32;
                acc = fmaf(scale * (float)v, xv[i], acc);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0 && row < m) y[row] = acc;
}
