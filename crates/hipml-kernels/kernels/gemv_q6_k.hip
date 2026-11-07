// y[row] = dequant_q6_k(W[row,:]) . x
//
// Row-tile GEMV: one block owns ROW_TILE output rows and stages the x k-tile in
// shared memory once. Each warp owns ROWS_PER_WARP rows and keeps one
// accumulator per row, so a value read from shared is applied to
// ROWS_PER_WARP weight rows before it is discarded.
//
// Each lane owns VALUES_PER_LANE (16) consecutive columns, which is exactly one
// Q6_K int8 scale sub-block: one scale and 16 ql + 16 qh bytes (2-byte aligned,
// so eight uint16 loads each).
//
// The shared tile is de-interleaved by 512-value group: component `c` of a group
// lives at sx[g*512 + c*32 + lane], so a lane's 16 scalar reads hit 16 distinct
// banks and the warp has no bank conflicts.
//
// Q6_K super-block (210 bytes, 256 values):
//   ql(128) qh(64) scales(16) d(2)
// Math is unchanged: q = (ql nibble) | ((qh plane) << 4), y = d*scale*(q-32).
//
// Two barriers per k-tile; the only reduction is a warp shuffle tree per row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q6K_VALUES 256
#define Q6K_BLOCK_BYTES 210
#define BLOCK 256
#define WARPS (BLOCK / 32)
#define ROWS_PER_WARP 2
#define ROW_TILE (WARPS * ROWS_PER_WARP)
#define VALUES_PER_LANE 16
#define GROUP_VALUES (VALUES_PER_LANE * 32)
#define K_TILE 2048

// Mirrors the manifest's KernelKind::RowTile { rows = ROW_TILE, block = BLOCK };
// change both together.

static __device__ __forceinline__ void stage_group6(const float* __restrict__ x,
                                                    float* __restrict__ sx,
                                                    int base, int j)
{
    const int g = j >> 5;
    const int lane = j & 31;
    float* s = sx + g * GROUP_VALUES + lane;
    const float4* src = reinterpret_cast<const float4*>(x + base + VALUES_PER_LANE * j);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 v = src[q];
        s[(4 * q + 0) * 32] = v.x;
        s[(4 * q + 1) * 32] = v.y;
        s[(4 * q + 2) * 32] = v.z;
        s[(4 * q + 3) * 32] = v.w;
    }
}

static __device__ __forceinline__ unsigned int join16(unsigned short a,
                                                      unsigned short b)
{
    return (unsigned int)a | ((unsigned int)b << 16);
}

// Sixteen-column contribution of one Q6_K row to `acc`.
static __device__ __forceinline__ void q6k_acc(const unsigned char* __restrict__ rp,
                                               int p, const float* xv, float* acc)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int half = within >> 7;
    const int ph = within & 127;
    const int group = ph >> 5;
    const int l = ph & 31; // 0 or 16

    const unsigned char* b = rp + (size_t)blk * Q6K_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
    const int sub = group * 2 + (l >> 4);
    const float scale =
        d * (float)reinterpret_cast<const signed char*>(b + 192)[half * 8 + sub];

    const int ql_off = half * 64 + ((group & 1) ? 32 : 0) + l;
    const int qh_off = 128 + half * 32 + l;
    unsigned int qlw[4];
    unsigned int qhw[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        qlw[w] = join16(
            *reinterpret_cast<const unsigned short*>(b + ql_off + 4 * w),
            *reinterpret_cast<const unsigned short*>(b + ql_off + 4 * w + 2));
        qhw[w] = join16(
            *reinterpret_cast<const unsigned short*>(b + qh_off + 4 * w),
            *reinterpret_cast<const unsigned short*>(b + qh_off + 4 * w + 2));
    }

    const int shift = 2 * group;
    const bool high = group >= 2;

    float a = *acc;
#pragma unroll
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const unsigned int qb = (qlw[i >> 2] >> (8 * (i & 3))) & 0xFFu;
        const unsigned int hb = (qhw[i >> 2] >> (8 * (i & 3))) & 0xFFu;
        const unsigned int nib = high ? (qb >> 4) : (qb & 0xF);
        const unsigned int hi = (hb >> shift) & 3u;
        const int v = (int)(nib | (hi << 4)) - 32;
        a = fmaf(scale * (float)v, xv[i], a);
    }
    *acc = a;
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
    const int nblocks = k / Q6K_VALUES;

    int row[ROWS_PER_WARP];
    int r[ROWS_PER_WARP];
    const unsigned char* rp[ROWS_PER_WARP];
    float acc[ROWS_PER_WARP];
#pragma unroll
    for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
        row[rr] = blockIdx.x * ROW_TILE + warp * ROWS_PER_WARP + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * Q6K_BLOCK_BYTES;
        acc[rr] = 0.0f;
    }

    for (int base = 0; base < k; base += K_TILE) {
        const int n = min(K_TILE, k - base);
        const int nchunks = n / VALUES_PER_LANE;
        for (int j = tid; j < nchunks; j += BLOCK) stage_group6(x, sx, base, j);
        __syncthreads();

        for (int c = lane * VALUES_PER_LANE; c < n; c += GROUP_VALUES) {
            const int g = c / GROUP_VALUES;
            const int s = g * GROUP_VALUES + lane;
            float xv[VALUES_PER_LANE];
#pragma unroll
            for (int comp = 0; comp < VALUES_PER_LANE; ++comp) xv[comp] = sx[s + comp * 32];
            const int p = base + c;
#pragma unroll
            for (int rr = 0; rr < ROWS_PER_WARP; ++rr) q6k_acc(rp[rr], p, xv, &acc[rr]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
        float a = acc[rr];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
        if (lane == 0 && row[rr] < m) y[row[rr]] = a;
    }
}
